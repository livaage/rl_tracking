from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete

from rl_tracking.utils.logger import get_logger

logger = get_logger()

COLS = [
    "hit_id",
    "x",
    "y",
    "z",
    "volume_id",
    "layer_id",
    "module_id",
    "particle_id",
    "tx",
    "ty",
    "tz",
    "tpx",
    "tpy",
    "tpz",
    "weight",
    "particle_type",
    "vx",
    "vy",
    "vz",
    "px",
    "py",
    "pz",
    "q",
    "nhits",
    "r",
    "pt",
    "unique_layer_id",
    "theta",
    "phi",
]

STATE_FEATURE_NAMES_FULL = [
    "last_hit_z_norm",
    "last_hit_x_norm",
    "last_hit_y_norm",
    "delta_z_norm",
    "delta_x_norm",
    "delta_y_norm",
    "predicted_delta_z_norm",
    "predicted_delta_x_norm",
    "predicted_delta_y_norm",
    "target_layer_norm",
    "remaining_ratio",
    "slope_xz_norm",
    "slope_yz_norm",
    "intercept_x_norm",
    "intercept_y_norm",
    "seed_pt_norm",
    "seed_p_norm",
]

STATE_FEATURE_NAMES_REDUCED = STATE_FEATURE_NAMES_FULL[3:]


@dataclass(frozen=True)
class LayerSurface:
    surface_type: str  # "cylinder" or "disk"
    radius: Optional[float] = None
    z: Optional[float] = None
    r_min: Optional[float] = None
    r_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None


class StraightLineTrackingEnv:
    """Environment that propagates a straight 3D line through detector layers."""

    MAX_CANDIDATES = 20
    SEED_LENGTH = 3
    DISTANCE_SCALE = 5.0  # centimetres used for distance reward shaping

    def __init__(
        self,
        data_loader,
        particle_filters: Optional[Dict] = None,
        hit_filters: Optional[Dict] = None,
        use_distance_reward: bool = False,
        deterministic: bool = False,
        hit_feature_mode: str = "absolute",
        state_feature_mode: str = "full",
        correct_nonprimary_bonus: float = 0.5,
    ):
        self.data_loader = iter(data_loader)
        self.particle_filters = particle_filters or {}
        self.hit_filters = hit_filters or {}
        self.use_distance_reward = use_distance_reward
        self.deterministic = deterministic
        self.hit_feature_mode = (hit_feature_mode or "absolute").lower()
        if self.hit_feature_mode not in {"absolute", "relative"}:
            logger.warning(
                "Unsupported hit_feature_mode '%s'; defaulting to 'absolute'.",
                hit_feature_mode,
            )
            self.hit_feature_mode = "absolute"
        self.state_feature_mode = (state_feature_mode or "full").lower()
        if self.state_feature_mode not in {"full", "reduced"}:
            logger.warning(
                "Unsupported state_feature_mode '%s'; defaulting to 'full'.",
                state_feature_mode,
            )
            self.state_feature_mode = "full"
        self.correct_nonprimary_bonus = max(0.0, float(correct_nonprimary_bonus))
        self.state_feature_names = (
            STATE_FEATURE_NAMES_FULL
            if self.state_feature_mode == "full"
            else STATE_FEATURE_NAMES_REDUCED
        )
        self.state_feature_dim = len(self.state_feature_names)
        self.state_feature_set = getattr(self, "state_feature_set", "unspecified")

        self.layer_surfaces = self._load_layer_geometry()

        self.all_hits: Optional[pd.DataFrame] = None
        self.pids_to_explore: list[int] = []
        self.particle_index: int = 0

        self.truth_hits: Optional[pd.DataFrame] = None
        self.accepted_hits: Optional[pd.DataFrame] = None
        self.seed_hit_ids: set[int] = set()
        self.current_target_index: int = 0
        self.current_track: Optional[pd.DataFrame] = None
        self.hit_number: int = 0

        self.current_candidates: Optional[pd.DataFrame] = None
        self.current_candidate_mask: Optional[np.ndarray] = None
        self.current_truth_hit_ids: set[int] = set()
        self.current_rewards: list[float] = []
        self.current_predicted_point: Optional[np.ndarray] = None
        self.current_info: Dict[str, object] = {}
        self.current_target_volume: Optional[int] = None
        self.current_target_layer_raw: Optional[int] = None
        self.current_target_unique_layer: Optional[int] = None
        self.current_remaining_hits: int = 0
        self.candidate_distance_max = 100.0  # millimetres (~10 cm) tolerance for candidate gathering

        self.distance_history: list[float] = []
        self.distance_sum: float = 0.0
        self.distance_count: int = 0

        self.total_selections = 0
        self.correct_selections = 0
        self.rank_selection_counts = Counter()
        self.rank_correct_counts = Counter()
        self.last_selected_hit: Optional[pd.Series] = None
        self.last_selected_action_index: Optional[int] = None
        self.last_selected_hit_id: Optional[int] = None
        self.episode_seed_layer_max: Optional[float] = None
        self.episode_layers_post_seed: List[float] = []
        self.episode_layers_with_correct_hit: Set[float] = set()
        self.episode_layer_failures: Counter = Counter()
        self.episode_layer_failure_details: List[Dict[str, object]] = []
        self.last_episode_stats: Optional[Dict[str, float]] = None
        self.truth_hit_total: int = 0
        self.seed_pt_estimate: float = 0.0
        self.seed_p_estimate: float = 0.0
        self.seed_direction_vector: np.ndarray = np.zeros(3, dtype=np.float64)
        self.state_feature_metadata: Dict[str, object] = {
            "mode": self.state_feature_mode,
            "set": self.state_feature_set,
            "features": list(self.state_feature_names),
        }
        self._last_candidate_stats: Dict[str, object] = {}
        self._last_candidate_stats: Dict[str, object] = {}

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.state_feature_dim,), dtype=np.float32
        )
        self.action_space = Discrete(self.MAX_CANDIDATES)

        self.load_next_file()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_info_path() -> Path:
        base_dir = Path(__file__).resolve().parent.parent
        primary = base_dir / "physics" / "layer_info.csv"
        if primary.exists():
            return primary

        fallback_candidates = [
            base_dir / "training" / "layer_info.csv",
            base_dir / "layer_info.csv",
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                logger.warning(
                    "Using fallback layer_info.csv from %s because the primary physics asset is missing.",
                    candidate,
                )
                return candidate

        searched = [str(primary)] + [str(path) for path in fallback_candidates]
        raise FileNotFoundError(
            "Could not locate 'layer_info.csv' required for detector geometry. "
            f"Searched: {', '.join(searched)}. Ensure the geometry assets are installed with the package."
        )

    def _load_layer_geometry(self) -> Dict[int, LayerSurface]:
        path = self._layer_info_path()
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        geometry: Dict[int, LayerSurface] = {}

        for idx, row in df.iterrows():
            layer_id = int(round(float(idx)))
            z_min = float(row[("z", "min")])
            z_max = float(row[("z", "max")])
            z_median = float(row[("z", "median")])
            r_min = float(row[("r", "min")])
            r_max = float(row[("r", "max")])
            r_median = float(row[("r", "median")])

            if abs(z_max - z_min) < 5.0:
                geometry[layer_id] = LayerSurface(
                    surface_type="disk",
                    z=z_median,
                    r_min=r_min,
                    r_max=r_max,
                )
            else:
                geometry[layer_id] = LayerSurface(
                    surface_type="cylinder",
                    radius=r_median,
                    z_min=z_min,
                    z_max=z_max,
                )
        return geometry

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _apply_particle_filters(self, hits: pd.DataFrame) -> pd.DataFrame:
        filtered = hits
        pt_cut = self.particle_filters.get("pt")
        nhits_min = self.particle_filters.get("nhits_min")
        nhits_max = self.particle_filters.get("nhits_max")

        if pt_cut is not None:
            filtered = filtered[filtered["pt"] > pt_cut]
        if nhits_min is not None:
            filtered = filtered[filtered["nhits"] >= nhits_min]
        if nhits_max is not None:
            filtered = filtered[filtered["nhits"] <= nhits_max]

        layers_per_particle = filtered.groupby("particle_id")[
            "unique_layer_id"
        ].nunique()
        required_layers = self.SEED_LENGTH + 1
        keep = layers_per_particle[layers_per_particle >= required_layers].index

        if len(keep) == 0:
            logger.warning(
                "apply_particle_filters() removed all particles: none had >= %d layers after filtering.",
                required_layers,
            )
            return filtered.iloc[0:0]

        dropped = set(filtered["particle_id"].unique()) - set(keep)
        if dropped:
            logger.info(
                "apply_particle_filters() dropped %d particles with insufficient layers (required %d).",
                len(dropped),
                required_layers,
            )

        return filtered[filtered["particle_id"].isin(keep)]

    def load_next_file(self) -> None:
        for sample in self.data_loader:
            if isinstance(sample, (list, tuple)):
                sample_tensor = sample[0]
            else:
                sample_tensor = sample

            sample_tensor = sample_tensor.squeeze(0)
            array = sample_tensor.view(sample_tensor.shape[0], sample_tensor.shape[1])
            hits = pd.DataFrame(array.numpy(), columns=COLS)
            hits["hit_id"] = hits["hit_id"].astype(int)
            hits["particle_id"] = hits["particle_id"].astype(int)
            hits["unique_layer_id"] = hits["unique_layer_id"].astype(int)
            hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)

            hits = self._apply_particle_filters(hits)
            if hits.empty:
                logger.warning(
                    "load_next_file() filtered event produced no valid hits; skipping event."
                )
                continue

            self.all_hits = hits
            particle_ids = hits["particle_id"].unique().astype(int).tolist()
            if self.deterministic:
                particle_ids.sort()
            else:
                np.random.shuffle(particle_ids)

            self.pids_to_explore = particle_ids
            self.particle_index = 0
            logger.info(
                "Loaded event with %d hits and %d particles",
                len(hits),
                len(particle_ids),
            )
            return

        self.all_hits = None
        self.pids_to_explore = []
        logger.warning("load_next_file() exhausted data loader with no remaining events.")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset environment to a valid track, skipping empty ones."""
        attempts = 0

        while True:
            if self.all_hits is None or self.particle_index >= len(self.pids_to_explore):
                self.load_next_file()

            if self.all_hits is None or not self.pids_to_explore:
                logger.warning("reset() could not find any particles to explore in loaded file.")
                return None, {}

            if attempts >= len(self.pids_to_explore):
                logger.warning(
                    "reset() exhausted available particles without finding a valid next state."
                )
                return None, {}

            particle_id = self.pids_to_explore[self.particle_index]
            self.particle_index += 1
            attempts += 1

            track = self.all_hits[self.all_hits["particle_id"] == particle_id].copy()
            track = track.sort_values(
                ["unique_layer_id", "r", "z"], kind="stable"
            ).reset_index(drop=True)
            track = track.drop_duplicates(
                subset="unique_layer_id", keep="first"
            ).reset_index(drop=True)

            if len(track) < self.SEED_LENGTH:
                logger.warning(
                    "reset() skipping particle %s: only %d hits available (need %d).",
                    particle_id,
                    len(track),
                    self.SEED_LENGTH,
                )
                continue

            self.truth_hits = track
            self.truth_hit_total = len(track)
            self.current_track = track

            seed_len = min(self.SEED_LENGTH, len(track))
            self.accepted_hits = track.iloc[:seed_len].copy()
            self.seed_hit_ids = set(self.accepted_hits["hit_id"].astype(int))
            self.current_target_index = seed_len
            self.hit_number = seed_len

            self.current_candidates = None
            self.current_candidate_mask = None
            self.current_truth_hit_ids = set()
            self.current_rewards = []
            self.current_predicted_point = None
            self.distance_history.clear()
            self.distance_sum = 0.0
            self.distance_count = 0
            self.current_target_volume = None
            self.current_target_layer_raw = None
            self.current_target_unique_layer = None
            self.current_remaining_hits = 0

            self.total_selections = 0
            self.correct_selections = 0
            self.rank_selection_counts.clear()
            self.rank_correct_counts.clear()
            self.last_selected_hit = None
            self.last_episode_stats = None
            seed_layer_ids = set(self.accepted_hits["unique_layer_id"].astype(int))
            self.episode_seed_layer_max = (
                float(max(seed_layer_ids)) if seed_layer_ids else None
            )
            all_layers = sorted(self.current_track["unique_layer_id"].unique())
            if self.episode_seed_layer_max is not None:
                self.episode_layers_post_seed = [
                    float(layer) for layer in all_layers if layer > self.episode_seed_layer_max
                ]
            else:
                self.episode_layers_post_seed = [float(layer) for layer in all_layers]
            self.episode_layers_with_correct_hit = set()
            self.episode_layer_failures = Counter()
            self.episode_layer_failure_details = []
            self._update_seed_momentum_estimate()
            
            state, info = self.get_current_state()
            if state is not None:
                return state, info

            logger.warning(
                "reset() skipping particle %s: no valid state produced (attempt %d).",
                particle_id,
                attempts,
            )

    # ------------------------------------------------------------------
    # Straight-line propagation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a straight line using PCA; returns point on line and unit direction."""
        if len(points) == 0:
            raise ValueError("Cannot fit line to empty point set")

        centroid = points.mean(axis=0)
        if len(points) == 1:
            direction = points[0] - centroid
        else:
            centered = points - centroid
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, -1]

        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = direction / norm

        if len(points) >= 2 and np.dot(direction, points[-1] - points[0]) < 0:
            direction = -direction

        return centroid, direction

    def _intersect_layer(
        self,
        point_on_line: np.ndarray,
        direction: np.ndarray,
        layer_id: int,
        layer_hits: pd.DataFrame,
    ) -> np.ndarray:
        if layer_hits.empty:
            return point_on_line + direction

        r_min = float(layer_hits["r"].min())
        r_max = float(layer_hits["r"].max())
        z_min = float(layer_hits["z"].min())
        z_max = float(layer_hits["z"].max())

        if abs(z_max - z_min) < 5.0:
            surface = LayerSurface(
                surface_type="disk",
                z=float(layer_hits["z"].median()),
                r_min=r_min,
                r_max=r_max,
            )
            intersection = self._intersect_disk(point_on_line, direction, surface)
        else:
            surface = LayerSurface(
                surface_type="cylinder",
                radius=float(layer_hits["r"].median()),
                z_min=z_min,
                z_max=z_max,
            )
            intersection = self._intersect_cylinder(point_on_line, direction, surface)

        if intersection is None:
            return point_on_line + direction
        return intersection

    @staticmethod
    def _intersect_disk(
        origin: np.ndarray, direction: np.ndarray, surface: LayerSurface
    ) -> Optional[np.ndarray]:
        if surface.z is None or abs(direction[2]) < 1e-9:
            return None

        t = (surface.z - origin[2]) / direction[2]
        if t <= 1e-6:
            return None

        point = origin + t * direction
        r = np.hypot(point[0], point[1])
        if surface.r_min is not None and r < surface.r_min:
            return None
        if surface.r_max is not None and r > surface.r_max:
            return None
        return point

    @staticmethod
    def _intersect_cylinder(
        origin: np.ndarray, direction: np.ndarray, surface: LayerSurface
    ) -> Optional[np.ndarray]:
        radius = surface.radius
        if radius is None:
            return None

        dx, dy = direction[0], direction[1]
        a = dx**2 + dy**2
        if a < 1e-12:
            return None

        x0, y0 = origin[0], origin[1]
        b = 2 * (x0 * dx + y0 * dy)
        c = x0**2 + y0**2 - radius**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        t_values = [t for t in (t1, t2) if t > 1e-6]
        if not t_values:
            return None
        t = min(t_values)
        point = origin + t * direction

        if surface.z_min is not None and point[2] < surface.z_min:
            point[2] = surface.z_min
        if surface.z_max is not None and point[2] > surface.z_max:
            point[2] = surface.z_max

        return point

    def _prepare_candidates(
        self,
        predicted_point: np.ndarray,
        layer_hits: pd.DataFrame,
        truth_hit_ids: Set[int],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        columns = ["hit_id", "x", "y", "z", "r", "distance"]
        truth_in_layer_hits = False
        truth_after_used_filter = False
        truth_after_distance_filter = False
        truth_in_candidates = False
        if layer_hits.empty:
            self._last_candidate_stats = {
                "initial_count": 0,
                "pre_filter_count": 0,
                "post_filter_count": 0,
                "distance_threshold": self.candidate_distance_max,
                "eta": None,
                "candidate_min_distance": None,
                "candidate_max_distance": None,
                "candidate_mean_distance": None,
                "candidate_distance_sample": [],
                "truth_in_layer_hits": False,
                "truth_after_used_filter": False,
                "truth_after_distance_filter": False,
                "truth_in_candidates_after_cap": False,
            }
            return pd.DataFrame(columns=columns), np.zeros(self.MAX_CANDIDATES, dtype=bool)

        # Ensure core columns are present even if upstream transformations dropped them
        for col in ["hit_id", "x", "y", "z", "r"]:
            if col not in layer_hits.columns:
                layer_hits[col] = self.all_hits.loc[layer_hits.index, col].values

        if truth_hit_ids:
            layer_hit_ids_before_used = set(layer_hits["hit_id"].astype(int))
            truth_in_layer_hits = bool(truth_hit_ids & layer_hit_ids_before_used)

        used_ids = set(self.accepted_hits["hit_id"].astype(int))
        layer_hits = layer_hits[~layer_hits["hit_id"].isin(used_ids)]
        if layer_hits.empty:
            self._last_candidate_stats = {
                "initial_count": 0,
                "pre_filter_count": 0,
                "post_filter_count": 0,
                "distance_threshold": self.candidate_distance_max,
                "eta": None,
                "candidate_min_distance": None,
                "candidate_max_distance": None,
                "candidate_mean_distance": None,
                "candidate_distance_sample": [],
                "truth_in_layer_hits": truth_in_layer_hits,
                "truth_after_used_filter": False,
                "truth_after_distance_filter": False,
                "truth_in_candidates_after_cap": False,
            }
            return pd.DataFrame(columns=columns), np.zeros(self.MAX_CANDIDATES, dtype=bool)

        layer_hits = layer_hits.copy()
        initial_count = len(layer_hits)
        if initial_count == 0:
            self._last_candidate_stats = {
                "initial_count": 0,
                "pre_filter_count": 0,
                "post_filter_count": 0,
                "distance_threshold": self.candidate_distance_max,
                "eta": None,
                "candidate_min_distance": None,
                "candidate_max_distance": None,
                "candidate_mean_distance": None,
                "candidate_distance_sample": [],
                "truth_in_layer_hits": truth_in_layer_hits,
                "truth_after_used_filter": truth_after_used_filter,
                "truth_after_distance_filter": False,
                "truth_in_candidates_after_cap": False,
            }
            return pd.DataFrame(columns=columns), np.zeros(self.MAX_CANDIDATES, dtype=bool)
        if truth_hit_ids:
            layer_hit_ids_after_used = set(layer_hits["hit_id"].astype(int))
            truth_after_used_filter = bool(truth_hit_ids & layer_hit_ids_after_used)
        coords = layer_hits[["x", "y", "z"]].to_numpy()
        distances = np.linalg.norm(coords - predicted_point[None, :], axis=1)
        layer_hits["distance"] = distances
        eta = abs(float(layer_hits["eta"].median())) if "eta" in layer_hits.columns else 0.0
        #eta_scale = min(1.0 + 0.3 * eta, 3.0)
        #distance_threshold = self.candidate_distance_max * eta_scale
        distance_threshold = 3000
        pre_filter_count = len(layer_hits)

        layer_hits = layer_hits[layer_hits["distance"] <= distance_threshold]
        post_filter_count = len(layer_hits)
        if truth_hit_ids:
            layer_hit_ids_after_distance = set(layer_hits["hit_id"].astype(int))
            truth_after_distance_filter = bool(truth_hit_ids & layer_hit_ids_after_distance)
        if layer_hits.empty:
            self._last_candidate_stats = {
                "initial_count": initial_count,
                "pre_filter_count": pre_filter_count,
                "post_filter_count": post_filter_count,
                "distance_threshold": distance_threshold,
                "eta": eta,
                "candidate_min_distance": None,
                "candidate_max_distance": None,
                "candidate_mean_distance": None,
                "candidate_distance_sample": [],
                "truth_in_layer_hits": truth_in_layer_hits,
                "truth_after_used_filter": truth_after_used_filter,
                "truth_after_distance_filter": truth_after_distance_filter,
                "truth_in_candidates_after_cap": False,
            }
            return pd.DataFrame(columns=columns), np.zeros(self.MAX_CANDIDATES, dtype=bool)

        layer_hits = layer_hits.sort_values("distance")
        candidates = layer_hits.head(self.MAX_CANDIDATES).reset_index(drop=True)
        mask = np.zeros(self.MAX_CANDIDATES, dtype=bool)
        mask[: len(candidates)] = True
        candidate_distances = candidates["distance"].to_numpy(dtype=float)
        if candidate_distances.size > 0:
            min_dist = float(np.min(candidate_distances))
            max_dist = float(np.max(candidate_distances))
            mean_dist = float(np.mean(candidate_distances))
            distance_sample = candidate_distances[
                : min(5, candidate_distances.size)
            ].tolist()
        else:
            min_dist = max_dist = mean_dist = None
            distance_sample = []
        if truth_hit_ids:
            candidate_ids_set = set(candidates["hit_id"].astype(int))
            truth_in_candidates = bool(truth_hit_ids & candidate_ids_set)
        self._last_candidate_stats = {
            "initial_count": initial_count,
            "pre_filter_count": pre_filter_count,
            "post_filter_count": len(candidates),
            "distance_threshold": distance_threshold,
            "eta": eta,
            "candidate_min_distance": min_dist,
            "candidate_max_distance": max_dist,
            "candidate_mean_distance": mean_dist,
            "candidate_distance_sample": distance_sample,
            "truth_in_layer_hits": truth_in_layer_hits,
            "truth_after_used_filter": truth_after_used_filter,
            "truth_after_distance_filter": truth_after_distance_filter,
            "truth_in_candidates_after_cap": truth_in_candidates,
        }
        return candidates, mask

    def _update_seed_momentum_estimate(self) -> None:
        if self.accepted_hits is None or len(self.accepted_hits) < 2:
            self.seed_direction_vector = np.zeros(3, dtype=np.float64)
            self.seed_pt_estimate = 0.0
            self.seed_p_estimate = 0.0
            return

        start = (
            self.accepted_hits.iloc[0][["x", "y", "z"]]
            .astype(float)
            .to_numpy()
        )
        end = (
            self.accepted_hits.iloc[-1][["x", "y", "z"]]
            .astype(float)
            .to_numpy()
        )
        direction = end - start
        self.seed_direction_vector = direction
        total_momentum = float(np.linalg.norm(direction))
        transverse_momentum = float(np.linalg.norm(direction[:2]))
        self.seed_p_estimate = total_momentum
        self.seed_pt_estimate = transverse_momentum

    def _compute_rewards(self, candidates: pd.DataFrame) -> list[float]:
        rewards: list[float] = []
        for idx, (_, row) in enumerate(candidates.iterrows()):
            hit_id = int(row["hit_id"])
            if hit_id in self.current_truth_hit_ids:
                reward = 1.0
                if idx > 0:
                    reward += self.correct_nonprimary_bonus
                rewards.append(reward)
            elif self.use_distance_reward:
                distance = float(row["distance"])
                reward = max(0.0, 1.0 - distance / self.DISTANCE_SCALE)
                rewards.append(reward)
            else:
                rewards.append(0.0)
        return rewards

    def _current_observation(self) -> np.ndarray:
        last_hit = self.accepted_hits.iloc[-1]
        z = float(last_hit["z"])
        x = float(last_hit["x"])
        y = float(last_hit["y"])

        coord_scale = 300.0
        delta_scale = 50.0
        layer_scale = 50.0
        slope_scale = 5.0
        momentum_scale = 100.0

        feature_values = {name: 0.0 for name in STATE_FEATURE_NAMES_FULL}

        feature_values["last_hit_z_norm"] = z / coord_scale
        feature_values["last_hit_x_norm"] = x / coord_scale
        feature_values["last_hit_y_norm"] = y / coord_scale

        if len(self.accepted_hits) >= 2:
            prev = self.accepted_hits.iloc[-2]
            prev_z = float(prev["z"])
            prev_x = float(prev["x"])
            prev_y = float(prev["y"])
            feature_values["delta_z_norm"] = (z - prev_z) / delta_scale
            feature_values["delta_x_norm"] = (x - prev_x) / delta_scale
            feature_values["delta_y_norm"] = (y - prev_y) / delta_scale

        if self.current_predicted_point is not None:
            pred_x, pred_y, pred_z = self.current_predicted_point.tolist()
            feature_values["predicted_delta_z_norm"] = (pred_z - z) / delta_scale
            feature_values["predicted_delta_x_norm"] = (pred_x - x) / delta_scale
            feature_values["predicted_delta_y_norm"] = (pred_y - y) / delta_scale

        unique_layer = float(self.current_target_unique_layer or 0)
        remaining_ratio = 0.0
        if self.truth_hit_total > 0:
            remaining_ratio = self.current_remaining_hits / float(self.truth_hit_total)

        feature_values["target_layer_norm"] = unique_layer / layer_scale
        feature_values["remaining_ratio"] = remaining_ratio

        points = self.accepted_hits[["x", "y", "z"]].to_numpy()
        line_origin, line_direction = self._fit_line(points)
        dz = line_direction[2]
        if abs(dz) < 1e-9:
            slope_xz = 0.0
            slope_yz = 0.0
        else:
            slope_xz = line_direction[0] / dz
            slope_yz = line_direction[1] / dz
        intercept_x = float(line_origin[0] - slope_xz * line_origin[2])
        intercept_y = float(line_origin[1] - slope_yz * line_origin[2])

        feature_values["slope_xz_norm"] = slope_xz / slope_scale
        feature_values["slope_yz_norm"] = slope_yz / slope_scale
        feature_values["intercept_x_norm"] = intercept_x / coord_scale
        feature_values["intercept_y_norm"] = intercept_y / coord_scale

        feature_values["seed_pt_norm"] = self.seed_pt_estimate / momentum_scale
        feature_values["seed_p_norm"] = self.seed_p_estimate / momentum_scale

        features = [feature_values[name] for name in self.state_feature_names]
        obs = np.array(features, dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_state(self):
        if (
            self.truth_hits is None
            or self.accepted_hits is None
            or self.current_target_index >= len(self.truth_hits)
        ):
            logger.warning(
                "get_current_state() unable to proceed: truth_hits=%s, accepted_hits=%s, target_index=%d.",
                None if self.truth_hits is None else len(self.truth_hits),
                None if self.accepted_hits is None else len(self.accepted_hits),
                self.current_target_index,
            )
            logger.warning(
                "get_current_state() called with no remaining truth hits (index=%d, total=%s).",
                self.current_target_index,
                None if self.truth_hits is None else len(self.truth_hits),
            )
            return None, {}

        target_row = self.truth_hits.iloc[self.current_target_index]
        target_layer = int(target_row["unique_layer_id"])
        target_volume = int(target_row["volume_id"])
        target_layer_id = int(target_row["layer_id"])
        self.current_target_volume = target_volume
        self.current_target_layer_raw = target_layer_id
        self.current_target_unique_layer = target_layer

        points = self.accepted_hits[["x", "y", "z"]].to_numpy()
        line_origin, line_direction = self._fit_line(points)

        layer_hits_full = self.all_hits[
            (self.all_hits["unique_layer_id"] == target_layer)
            & (self.all_hits["volume_id"] == target_volume)
            & (self.all_hits["layer_id"] == target_layer_id)
        ]

        if layer_hits_full.empty:
            logger.warning(
                "get_current_state() particle %s layer %s (volume %s) has no hits; skipping.",
                int(target_row.get("particle_id", -1)),
                target_layer,
                target_volume,
            )
            return None, {}

        truth_hits = self.truth_hits[self.truth_hits["unique_layer_id"] == target_layer]
        truth_hit_ids = set(truth_hits["hit_id"].astype(int).tolist())

        predicted_point = self._intersect_layer(
            line_origin, line_direction, target_layer, layer_hits_full
        )

        candidates, mask = self._prepare_candidates(
            predicted_point, layer_hits_full, truth_hit_ids
        )

        remaining = max(len(self.truth_hits) - self.current_target_index, 0)
        self.current_remaining_hits = remaining
        candidate_hit_ids = (
            set(candidates["hit_id"].astype(int).tolist())
            if not candidates.empty
            else set()
        )

        min_truth_distance = None
        max_truth_distance = None
        mean_truth_distance = None
        if not truth_hits.empty:
            truth_coords = truth_hits[["x", "y", "z"]].to_numpy()
            truth_distances = np.linalg.norm(
                truth_coords - predicted_point[None, :], axis=1
            )
            min_truth_distance = float(np.min(truth_distances))
            max_truth_distance = float(np.max(truth_distances))
            mean_truth_distance = float(np.mean(truth_distances))

        if truth_hit_ids:
            stats = getattr(self, "_last_candidate_stats", {}) or {}
            failure_detail: Dict[str, object] = {
                "layer": float(target_layer),
                "eta": float(truth_hits["eta"].median())
                if "eta" in truth_hits.columns and not truth_hits.empty
                else None,
                "min_truth_distance": min_truth_distance,
                "max_truth_distance": max_truth_distance,
                "mean_truth_distance": mean_truth_distance,
                "distance_threshold": float(
                    stats.get("distance_threshold", self.candidate_distance_max)
                ),
                "pre_filter_count": int(stats.get("pre_filter_count", 0)),
                "post_filter_count": int(stats.get("post_filter_count", len(candidates))),
                "truth_hit_count": int(len(truth_hits)),
                "missing_truth_hits": int(len(truth_hit_ids - candidate_hit_ids)),
                "candidate_min_distance": stats.get("candidate_min_distance"),
                "candidate_max_distance": stats.get("candidate_max_distance"),
                "candidate_mean_distance": stats.get("candidate_mean_distance"),
                "candidate_distance_sample": stats.get(
                    "candidate_distance_sample", []
                ),
            "truth_in_layer_hits": bool(stats.get("truth_in_layer_hits", False)),
            "truth_after_used_filter": bool(
                stats.get("truth_after_used_filter", False)
            ),
            "truth_after_distance_filter": bool(
                stats.get("truth_after_distance_filter", False)
            ),
            "truth_in_candidates_after_cap": bool(
                stats.get("truth_in_candidates_after_cap", False)
            ),
            }

            if not candidate_hit_ids:
                failure_type = "no_candidates"
                self.episode_layer_failures[failure_type] += 1
                failure_detail["failure_type"] = failure_type
                self.episode_layer_failure_details.append(failure_detail)
            elif candidate_hit_ids.isdisjoint(truth_hit_ids):
                failure_type = "truth_not_in_candidates"
                self.episode_layer_failures[failure_type] += 1
                failure_detail["failure_type"] = failure_type
                self.episode_layer_failure_details.append(failure_detail)
            else:
                missing_truth = len(truth_hit_ids - candidate_hit_ids)
                if missing_truth > 0:
                    failure_type = "truth_partially_missing"
                    self.episode_layer_failures[failure_type] += 1
                    failure_detail["failure_type"] = failure_type
                    self.episode_layer_failure_details.append(failure_detail)
        self.current_candidates = candidates
        self.current_candidate_mask = mask
        self.current_truth_hit_ids = truth_hit_ids
        self.current_predicted_point = predicted_point
        self.current_rewards = self._compute_rewards(candidates)

        if self.use_distance_reward and not candidates.empty:
            self.distance_history.extend(candidates["distance"].tolist())

        hit_features = np.zeros((self.MAX_CANDIDATES, 4), dtype=np.float32)
        pred_x, pred_y, pred_z = predicted_point.tolist()
        pred_r = np.hypot(pred_x, pred_y)
        for idx in range(len(candidates)):
            row = candidates.iloc[idx]
            if self.hit_feature_mode == "relative":
                values = np.array(
                    [
                        row["x"] - pred_x,
                        row["y"] - pred_y,
                        row["z"] - pred_z,
                        row["r"] - pred_r,
                    ],
                    dtype=np.float32,
                )
            else:
                values = np.array(
                    [row["x"], row["y"], row["z"], row["r"]],
                    dtype=np.float32,
                )
            hit_features[idx] = values

        info = {
            "hit_features": hit_features,
            "hit_mask": mask.copy(),
            "rewards": list(self.current_rewards),
            "candidate_hit_ids": candidates["hit_id"].astype(int).tolist(),
            "correct_hit_ids": truth_hit_ids,
            "target_layer_id": target_layer,
            "predicted_point": predicted_point.astype(float),
            "state_feature_names": list(self.state_feature_names),
            "state_feature_set": self.state_feature_set,
            "state_feature_metadata": dict(self.state_feature_metadata),
        }
        self.current_info = info
        return self._current_observation(), info

    def step(self, action: int):
        if self.current_candidates is None:
            logger.warning("step() called before get_current_state(); resetting.")
            state, info = self.reset()
            return state, 0.0, False, False, info

        if not self.current_candidates.empty:
            clamped_action = int(np.clip(action, 0, len(self.current_candidates) - 1))
            self.last_selected_action_index = clamped_action
            selected = self.current_candidates.iloc[clamped_action]
            selected_hit_id = int(selected["hit_id"])
            reward = float(self.current_rewards[clamped_action])
            self.last_selected_hit_id = selected_hit_id
            is_correct = selected_hit_id in self.current_truth_hit_ids
        else:
            selected = None
            selected_hit_id = None
            reward = 0.0
            is_correct = False
            self.last_selected_hit_id = None

        # if selected is not None:
        #     self.accepted_hits = pd.concat(
        #         [self.accepted_hits, selected.to_frame().T], ignore_index=True
        #     )

        if selected_hit_id is not None and selected_hit_id not in self.seed_hit_ids:
            self.total_selections += 1
            if is_correct:
                self.correct_selections += 1

        if self.current_candidates is not None and not self.current_candidates.empty:
            clamped_action = int(np.clip(action, 0, len(self.current_candidates) - 1))
            self.rank_selection_counts[clamped_action] += 1
            if is_correct:
                self.rank_correct_counts[clamped_action] += 1

        self.last_selected_hit = selected
        if (
            is_correct
            and self.episode_seed_layer_max is not None
            and selected is not None
            and "unique_layer_id" in selected
        ):
            selected_layer = float(selected["unique_layer_id"])
            if selected_layer > self.episode_seed_layer_max:
                self.episode_layers_with_correct_hit.add(selected_layer)

        next_state, info = self.get_current_state()
        done = False
        self.current_target_index += 1
        self.hit_number = self.current_target_index

        if self.current_target_index >= len(self.truth_hits) or self.current_candidates.empty:
            done = True

        if done:
            reason = (
                "finished_track"
                if self.current_target_index >= len(self.truth_hits)
                else "no_candidates"
            )
            expected_hits = max(self.truth_hit_total - self.SEED_LENGTH, 0)
            episode_stats = {
                "reason": reason,
                "total_selections": self.total_selections,
                "correct_selections": self.correct_selections,
                "expected_hits": float(expected_hits),
                "layer_total": float(len(self.episode_layers_post_seed)),
                "layer_correct": float(len(self.episode_layers_with_correct_hit)),
                "layer_failures": dict(self.episode_layer_failures),
                "layer_failure_details": list(self.episode_layer_failure_details),
            }
            context = getattr(self, "context_tag", "unknown")
            logger.info(
                "[%s] Episode done: reason=%s, selections=%d, correct=%d, expected=%d",
                context,
                reason,
                self.total_selections,
                self.correct_selections,
                expected_hits,
            )
            next_state, info = self.reset()
            self.last_episode_stats = episode_stats
        else:
            next_state, info = self.get_current_state()

        self.current_info = info
        truncated = False
        return next_state, reward, done, truncated, info


class TrackingEnv(StraightLineTrackingEnv):
    """Backward-compatible alias that supports legacy keyword arguments."""

    def __init__(self, data_loader, *args, use_truth_path_plan: bool = False, **kwargs):
        self.use_truth_path_plan = use_truth_path_plan
        super().__init__(data_loader, *args, **kwargs)
