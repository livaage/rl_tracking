from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    ):
        self.data_loader = iter(data_loader)
        self.particle_filters = particle_filters or {}
        self.hit_filters = hit_filters or {}
        self.use_distance_reward = use_distance_reward
        self.deterministic = deterministic

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
        self.candidate_distance_max = 10.0  # cm (~0.1 m) tolerance for candidate gathering

        self.distance_history: list[float] = []
        self.distance_sum: float = 0.0
        self.distance_count: int = 0

        self.total_selections = 0
        self.correct_selections = 0
        self.rank_selection_counts = Counter()
        self.rank_correct_counts = Counter()
        self.last_selected_hit: Optional[pd.Series] = None
        self.last_episode_stats: Optional[Dict[str, float]] = None
        self.truth_hit_total: int = 0

        self.observation_space = Box(
            low=-200.0, high=200.0, shape=(4,), dtype=np.float32
        )
        self.action_space = Discrete(self.MAX_CANDIDATES)

        self.load_next_file()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_info_path() -> Path:
        return Path(__file__).resolve().parent.parent / "physics" / "layer_info.csv"

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
        keep = layers_per_particle[layers_per_particle >= self.SEED_LENGTH].index
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

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        if self.all_hits is None or self.particle_index >= len(self.pids_to_explore):
            self.load_next_file()

        if self.all_hits is None or not self.pids_to_explore:
            return None, {}

        particle_id = self.pids_to_explore[self.particle_index]
        self.particle_index += 1

        track = self.all_hits[self.all_hits["particle_id"] == particle_id].copy()
        track = track.sort_values(
            ["unique_layer_id", "r", "z"], kind="stable"
        ).reset_index(drop=True)
        track = track.drop_duplicates(subset="unique_layer_id", keep="first").reset_index(
            drop=True
        )
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

        self.total_selections = 0
        self.correct_selections = 0
        self.rank_selection_counts.clear()
        self.rank_correct_counts.clear()
        self.last_selected_hit = None
        self.last_episode_stats = None

        return self.get_current_state()

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
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if layer_hits.empty:
            return pd.DataFrame(), np.zeros(self.MAX_CANDIDATES, dtype=bool)

        used_ids = set(self.accepted_hits["hit_id"].astype(int))
        layer_hits = layer_hits[~layer_hits["hit_id"].isin(used_ids)]
        if layer_hits.empty:
            return pd.DataFrame(), np.zeros(self.MAX_CANDIDATES, dtype=bool)

        layer_hits = layer_hits.copy()
        coords = layer_hits[["x", "y", "z"]].to_numpy()
        distances = np.linalg.norm(coords - predicted_point[None, :], axis=1)
        layer_hits["distance"] = distances
        layer_hits = layer_hits[layer_hits["distance"] <= self.candidate_distance_max]
        if layer_hits.empty:
            return pd.DataFrame(), np.zeros(self.MAX_CANDIDATES, dtype=bool)

        layer_hits = layer_hits.sort_values("distance")
        candidates = layer_hits.head(self.MAX_CANDIDATES).reset_index(drop=True)
        mask = np.zeros(self.MAX_CANDIDATES, dtype=bool)
        mask[: len(candidates)] = True
        return candidates, mask

    def _compute_rewards(self, candidates: pd.DataFrame) -> list[float]:
        rewards: list[float] = []
        for _, row in candidates.iterrows():
            hit_id = int(row["hit_id"])
            if hit_id in self.current_truth_hit_ids:
                rewards.append(1.0)
            elif self.use_distance_reward:
                distance = float(row["distance"])
                reward = max(0.0, 1.0 - distance / self.DISTANCE_SCALE)
                rewards.append(reward)
            else:
                rewards.append(0.0)
        return rewards

    def _current_observation(self) -> np.ndarray:
        last_hit = self.accepted_hits.iloc[-1]
        r = float(last_hit.get("r", np.hypot(last_hit["x"], last_hit["y"])))
        obs = np.array(
            [last_hit["z"], r, last_hit["x"], last_hit["y"]],
            dtype=np.float32,
        )
        return np.clip(obs, -200.0, 200.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_state(self):
        if (
            self.truth_hits is None
            or self.accepted_hits is None
            or self.current_target_index >= len(self.truth_hits)
        ):
            return None, {}

        target_row = self.truth_hits.iloc[self.current_target_index]
        target_layer = int(target_row["unique_layer_id"])

        points = self.accepted_hits[["x", "y", "z"]].to_numpy()
        line_origin, line_direction = self._fit_line(points)

        layer_hits_full = self.all_hits[
            (self.all_hits["unique_layer_id"] == target_layer)
            & (self.all_hits["volume_id"] == target_row["volume_id"])
            & (self.all_hits["layer_id"] == target_row["layer_id"])
        ]

        if layer_hits_full.empty:
            return None, {}

        predicted_point = self._intersect_layer(
            line_origin, line_direction, target_layer, layer_hits_full
        )

        candidates, mask = self._prepare_candidates(
            predicted_point, layer_hits_full
        )

        truth_hits = self.truth_hits[self.truth_hits["unique_layer_id"] == target_layer]
        truth_hit_ids = set(truth_hits["hit_id"].astype(int).tolist())

        self.current_candidates = candidates
        self.current_candidate_mask = mask
        self.current_truth_hit_ids = truth_hit_ids
        self.current_predicted_point = predicted_point
        self.current_rewards = self._compute_rewards(candidates)

        if self.use_distance_reward and not candidates.empty:
            self.distance_history.extend(candidates["distance"].tolist())

        hit_features = np.zeros((self.MAX_CANDIDATES, 4), dtype=np.float32)
        for idx in range(len(candidates)):
            row = candidates.iloc[idx]
            hit_features[idx] = np.array(
                [row["x"], row["y"], row["z"], row["r"]],
                dtype=np.float32,
            )

        info = {
            "hit_features": hit_features,
            "hit_mask": mask.copy(),
            "rewards": list(self.current_rewards),
            "candidate_hit_ids": candidates["hit_id"].astype(int).tolist(),
            "correct_hit_ids": truth_hit_ids,
            "target_layer_id": target_layer,
            "predicted_point": predicted_point.astype(float),
        }
        self.current_info = info
        return self._current_observation(), info

    def step(self, action: int):
        if self.current_candidates is None:
            logger.warning("step() called before get_current_state(); resetting.")
            return self.reset()

        if not self.current_candidates.empty:
            clamped_action = int(np.clip(action, 0, len(self.current_candidates) - 1))
            selected = self.current_candidates.iloc[clamped_action]
            selected_hit_id = int(selected["hit_id"])
            reward = float(self.current_rewards[clamped_action])
            is_correct = selected_hit_id in self.current_truth_hit_ids
        else:
            selected = None
            selected_hit_id = None
            reward = 0.0
            is_correct = False

        if selected is not None:
            self.accepted_hits = pd.concat(
                [self.accepted_hits, selected.to_frame().T], ignore_index=True
            )

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
        self.current_target_index += 1
        self.hit_number = self.current_target_index

        done = False
        if (
            self.current_target_index >= len(self.truth_hits)
            or self.current_candidates.empty
        ):
            done = True

        if done:
            reason = (
                "finished_track"
                if self.current_target_index >= len(self.truth_hits)
                else "no_candidates"
            )
            expected_hits = max(self.truth_hit_total - self.SEED_LENGTH, 0)
            self.last_episode_stats = {
                "reason": reason,
                "total_selections": self.total_selections,
                "correct_selections": self.correct_selections,
                "expected_hits": float(expected_hits),
            }
            next_state, info = self.reset()
        else:
            next_state, info = self.get_current_state()

        self.current_info = info
        truncated = False
        return next_state, reward, done, truncated, info


class TrackingEnv(StraightLineTrackingEnv):
    """Backward-compatible alias for legacy imports."""
