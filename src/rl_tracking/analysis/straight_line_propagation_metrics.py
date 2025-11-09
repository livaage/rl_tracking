from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from trackml.dataset import load_event


@dataclass
class StepMetrics:
    truth_rank: int
    num_candidates: int
    distance_to_truth: float


@dataclass
class Summary:
    total_truth_layers: int = 0
    evaluated_layers: int = 0
    truth_in_candidates: int = 0
    tracks_considered: int = 0
    rank_counts: Counter[int] = None
    distances: list[float] = None

    def __post_init__(self) -> None:
        if self.rank_counts is None:
            self.rank_counts = Counter()
        if self.distances is None:
            self.distances = []

    def update(self, metrics: StepMetrics, top_k: int) -> None:
        self.evaluated_layers += 1
        if metrics.truth_rank <= top_k:
            self.truth_in_candidates += 1
        bucket = metrics.truth_rank if metrics.truth_rank <= top_k else top_k + 1
        self.rank_counts[bucket] += 1
        self.distances.append(metrics.distance_to_truth)

    def extend_distances(self, values: Iterable[float]) -> None:
        self.distances.extend(values)


def _load_layer_remapping() -> pd.DataFrame:
    base = Path(__file__).resolve().parent.parent
    candidates = [
        base / "geometry" / "tml_layer_remap.csv",
        base / "training" / "tml_layer_remap.csv",
        base / "tml_layer_remap.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(
                candidate,
                usecols=["volume_id", "layer_id", "unique_layer_id"],
            ).drop_duplicates()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate tml_layer_remap.csv. Locations checked: {searched}"
    )


_LAYER_REMAPPING = _load_layer_remapping()


@dataclass(frozen=True)
class Surface:
    surface_type: str
    radius: Optional[float] = None
    z: Optional[float] = None
    r_min: Optional[float] = None
    r_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None


CANDIDATE_DISTANCE_MAX = 10.0  # cm


def _fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def _intersect_disk(origin: np.ndarray, direction: np.ndarray, surface: Surface) -> Optional[np.ndarray]:
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


def _intersect_cylinder(origin: np.ndarray, direction: np.ndarray, surface: Surface) -> Optional[np.ndarray]:
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
    t_candidates = [
        t for t in ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)) if t > 1e-6
    ]
    if not t_candidates:
        return None
    t = min(t_candidates)
    point = origin + t * direction
    if surface.z_min is not None and point[2] < surface.z_min:
        point[2] = surface.z_min
    if surface.z_max is not None and point[2] > surface.z_max:
        point[2] = surface.z_max
    return point


def _intersect_layer(origin: np.ndarray, direction: np.ndarray, layer_hits: pd.DataFrame) -> np.ndarray:
    if layer_hits.empty:
        return origin + direction

    r_min = float(layer_hits["r"].min())
    r_max = float(layer_hits["r"].max())
    z_min = float(layer_hits["z"].min())
    z_max = float(layer_hits["z"].max())

    if abs(z_max - z_min) < 5.0:
        surface = Surface(
            surface_type="disk",
            z=float(layer_hits["z"].median()),
            r_min=r_min,
            r_max=r_max,
        )
        intersection = _intersect_disk(origin, direction, surface)
    else:
        surface = Surface(
            surface_type="cylinder",
            radius=float(layer_hits["r"].median()),
            z_min=z_min,
            z_max=z_max,
        )
        intersection = _intersect_cylinder(origin, direction, surface)

    if intersection is None:
        return origin + direction
    return intersection


def _predict_next(history: Sequence[np.ndarray], layer_hits: pd.DataFrame) -> np.ndarray:
    history_arr = np.asarray(history, dtype=np.float64)
    if history_arr.shape[0] < 2:
        direction = history_arr[-1] - history_arr[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = direction / norm
        return history_arr[-1] + direction
    origin, direction = _fit_line(history_arr)
    return _intersect_layer(origin, direction, layer_hits)


def load_event_hits(prefix: Path) -> pd.DataFrame:
    hits, cells, particles, truth = load_event(prefix)
    merged = (
        hits.merge(truth, on="hit_id", how="inner")
        .merge(particles, on="particle_id", how="left")
        .rename(columns={"tx": "tx", "ty": "ty", "tz": "tz"})
    )
    merged["x"] *= 0.1
    merged["y"] *= 0.1
    merged["z"] *= 0.1
    merged["r"] = np.sqrt(merged["x"] ** 2 + merged["y"] ** 2)
    merged["pt"] = np.sqrt(merged["px"] ** 2 + merged["py"] ** 2)
    merged = merged.merge(_LAYER_REMAPPING, on=["volume_id", "layer_id"], how="left")
    merged = merged.sort_values(
        ["particle_id", "unique_layer_id", "r", "z"],
        kind="stable",
    ).reset_index(drop=True)
    return merged


def enumerate_steps(
    hits: pd.DataFrame,
    *,
    min_history: int,
    max_candidates: int,
    pt_cut: float,
    max_tracks: int | None,
) -> tuple[Summary, int, bool]:
    summary = Summary()
    missing_layers = 0
    top_k = max_candidates

    grouped = hits.groupby("particle_id", sort=False)
    for particle_id, group in grouped:
        group = group.drop_duplicates(subset="unique_layer_id", keep="first")
        group = group.sort_values(["unique_layer_id", "r", "z"], kind="stable")
        if max_tracks is not None and summary.tracks_considered >= max_tracks:
            return summary, missing_layers, True
        if group["pt"].iloc[0] < pt_cut:
            continue
        group = group.dropna(subset=["unique_layer_id"])
        if len(group) < min_history + 1:
            continue

        summary.tracks_considered += 1
        positions = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
        layers = group["unique_layer_id"].to_numpy(dtype=np.int64)
        volumes = group["volume_id"].to_numpy(dtype=np.int64)
        layer_ids = group["layer_id"].to_numpy(dtype=np.int64)
        hit_ids = group["hit_id"].to_numpy(dtype=np.int64)

        truth_steps = len(group) - min_history
        summary.total_truth_layers += truth_steps

        for idx in range(min_history, len(group)):
            target_layer = layers[idx]
            truth_hit_id = hit_ids[idx]

            layer_hits = hits[
                (hits["unique_layer_id"] == target_layer)
                & (hits["volume_id"] == volumes[idx])
                & (hits["layer_id"] == layer_ids[idx])
            ]
            if layer_hits.empty:
                missing_layers += 1
                continue

            history = positions[idx - min_history : idx]
            prediction = _predict_next(history, layer_hits)

            candidate_positions = layer_hits[["x", "y", "z"]].to_numpy(dtype=np.float64)
            candidate_hit_ids = layer_hits["hit_id"].to_numpy(dtype=np.int64)

            dists = np.linalg.norm(candidate_positions - prediction[None, :], axis=1)
            valid_mask = dists <= CANDIDATE_DISTANCE_MAX
            if not valid_mask.any():
                continue
            candidate_positions = candidate_positions[valid_mask]
            candidate_hit_ids = candidate_hit_ids[valid_mask]
            dists = dists[valid_mask]
            order = np.argsort(dists, kind="stable")
            selected = order[:max_candidates]

            selected_hit_ids = candidate_hit_ids[selected]
            selected_distances = dists[selected]

            truth_indices = np.where(selected_hit_ids == truth_hit_id)[0]
            if truth_indices.size > 0:
                truth_rank = int(truth_indices[0]) + 1
                distance_to_truth = float(selected_distances[truth_indices[0]])
            else:
                truth_rank = max_candidates + 1
                truth_pos = hits.loc[hits["hit_id"] == truth_hit_id, ["x", "y", "z"]].to_numpy()
                if truth_pos.size == 0:
                    distance_to_truth = float("nan")
                else:
                    distance_to_truth = float(np.linalg.norm(prediction - truth_pos[0]))

            summary.update(
                StepMetrics(
                    truth_rank=truth_rank,
                    num_candidates=len(selected_hit_ids),
                    distance_to_truth=distance_to_truth,
                ),
                top_k=top_k,
            )

    return summary, missing_layers, False


def format_distribution(counts: Counter[int], total: int, top_k: int) -> str:
    lines = []
    for rank in range(1, top_k + 1):
        count = counts.get(rank, 0)
        pct = count / total * 100 if total else 0.0
        lines.append(f"    Rank {rank:2d}: {count:6d} ({pct:5.1f}%)")
    overflow = counts.get(top_k + 1, 0)
    pct_over = overflow / total * 100 if total else 0.0
    lines.append(f"    >{top_k:2d}: {overflow:6d} ({pct_over:5.1f}%)")
    return "\n".join(lines)


def collect_event_prefixes(directory: Path, limit: int | None) -> list[Path]:
    hits_files = sorted(directory.glob("event*-hits.csv.gz"))
    prefixes = []
    for hits_file in hits_files:
        prefix = hits_file.with_suffix("")
        prefix = prefix.with_suffix("")
        # Remove trailing "-hits"
        prefixes.append(Path(str(prefix).rsplit("-", 1)[0]))
        if limit is not None and len(prefixes) >= limit:
            break
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser(description="Straight-line propagation diagnostics.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing TrackML event CSVs (hits/truth/particles).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5,
        help="Maximum number of events to analyse from the directory.",
    )
    parser.add_argument(
        "--pt-cuts",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="pt thresholds (GeV) to apply when choosing seed tracks.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=3,
        help="Number of seed hits used for the straight-line fit.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Number of candidate hits to keep per layer (matches environment).",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=100,
        help="Maximum number of tracks (particles) to evaluate per pt cut.",
    )
    args = parser.parse_args()

    prefixes = collect_event_prefixes(args.data_dir, args.max_events)
    if not prefixes:
        raise RuntimeError(f"No event prefixes found under {args.data_dir}")

    print(f"Analysing {len(prefixes)} events from {args.data_dir}")
    for prefix in prefixes:
        print(f"  - {prefix}")
    print()

    hits_by_event: dict[Path, pd.DataFrame] = {}
    for prefix in prefixes:
        hits_by_event[prefix] = load_event_hits(prefix)

    for pt_cut in args.pt_cuts:
        summary = Summary()
        missing_layers_total = 0
        reached_limit = False
        for prefix in prefixes:
            event_summary, missing_layers, limit_hit = enumerate_steps(
                hits_by_event[prefix],
                min_history=args.min_history,
                max_candidates=args.max_candidates,
                pt_cut=pt_cut,
                max_tracks=args.max_tracks,
            )
            summary.total_truth_layers += event_summary.total_truth_layers
            summary.evaluated_layers += event_summary.evaluated_layers
            summary.truth_in_candidates += event_summary.truth_in_candidates
            summary.tracks_considered += event_summary.tracks_considered
            summary.rank_counts.update(event_summary.rank_counts)
            summary.extend_distances(event_summary.distances)
            missing_layers_total += missing_layers
            if limit_hit:
                reached_limit = True
                break

        print(f"pt cut >= {pt_cut:.2f} GeV")
        print("---------------------------")
        print(f"Total truth layers (post-seed): {summary.total_truth_layers}")
        print(f"Evaluated layers:              {summary.evaluated_layers}")
        coverage = (
            summary.evaluated_layers / summary.total_truth_layers * 100
            if summary.total_truth_layers
            else 0.0
        )
        print(f"Coverage:                      {coverage:5.1f}%")
        print(f"Layers with truth in top {args.max_candidates}: {summary.truth_in_candidates}")
        inclusion_rate = (
            summary.truth_in_candidates / summary.evaluated_layers * 100
            if summary.evaluated_layers
            else 0.0
        )
        print(f"Truth inclusion rate:          {inclusion_rate:5.1f}%")
        print(f"Missing layer lookups:         {missing_layers_total}")
        print(f"Tracks evaluated:              {summary.tracks_considered}")
        if reached_limit:
            print("Reached track limit; additional tracks skipped.")
        print()
        print("Truth rank distribution:")
        print(
            format_distribution(
                summary.rank_counts,
                summary.evaluated_layers,
                args.max_candidates,
            )
        )
        print()
        distances = [d for d in summary.distances if np.isfinite(d)]
        if distances:
            avg_distance = float(np.mean(distances))
            std_distance = float(np.std(distances))
            print(f"Mean |prediction - truth| distance: {avg_distance:.3f} cm")
            print(f"Std  |prediction - truth| distance: {std_distance:.3f} cm")
        else:
            print("No finite distance measurements available.")
        print()


if __name__ == "__main__":
    main()
