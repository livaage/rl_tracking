"""Straight-line propagation diagnostics mirroring the RL environment.

This script evaluates the pure geometry-based straight-line propagator that
`StraightLineTrackingEnv` uses for candidate generation. It measures

1. How often the truth hit appears in the candidate list
2. The rank distribution of the truth hit among candidates
3. Basic distance statistics between the predicted point and the truth hit

Usage mirrors `test_track_propagation.py`: point it at a TrackML event directory
or allow it to sample a random event from a dataset root.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.logger import get_logger, set_log_file

logger = get_logger()

DEFAULT_BASE_DIR = Path("/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data")
DEFAULT_PART = "part_5"

SEED_LENGTH = 3
MAX_CANDIDATES = 10
CANDIDATE_DISTANCE_MAX = 100.0  # centimetres (matches StraightLineTrackingEnv)


def _event_prefix_from_hits(hits_file: Path) -> Path:
    name = hits_file.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".zip"):
        name = name[:-4]
    if not name.endswith("-hits.csv"):
        raise ValueError(f"Unrecognised hits file pattern: {hits_file}")
    prefix = name[:-len("-hits.csv")]
    return hits_file.parent / prefix


def select_random_event(base_dir: Path, part: str = DEFAULT_PART, seed: int | None = None) -> Path:
    part_dir = base_dir / part
    if not part_dir.exists():
        raise FileNotFoundError(f"Part directory not found: {part_dir}")

    patterns = ["event*-hits.csv", "event*-hits.csv.gz", "event*-hits.csv.zip"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(part_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No event files found in {part_dir}")

    rng = np.random.default_rng(seed)
    hits_file = Path(rng.choice(candidates))
    return _event_prefix_from_hits(hits_file)


@dataclass(frozen=True)
class LayerSurface:
    surface_type: str
    radius: Optional[float] = None
    z: Optional[float] = None
    r_min: Optional[float] = None
    r_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal-component line fit matching StraightLineTrackingEnv."""
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


def _intersect_disk(origin: np.ndarray, direction: np.ndarray, surface: LayerSurface) -> Optional[np.ndarray]:
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


def _intersect_cylinder(origin: np.ndarray, direction: np.ndarray, surface: LayerSurface) -> Optional[np.ndarray]:
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
        surface = LayerSurface(
            surface_type="disk",
            z=float(layer_hits["z"].median()),
            r_min=r_min,
            r_max=r_max,
        )
        intersection = _intersect_disk(origin, direction, surface)
    else:
        surface = LayerSurface(
            surface_type="cylinder",
            radius=float(layer_hits["r"].median()),
            z_min=z_min,
            z_max=z_max,
        )
        intersection = _intersect_cylinder(origin, direction, surface)

    if intersection is None:
        return origin + direction
    return intersection


def _prepare_candidates(
    predicted_point: np.ndarray,
    layer_hits: pd.DataFrame,
    used_hit_ids: set[int],
) -> pd.DataFrame:
    if layer_hits.empty:
        return pd.DataFrame(columns=["hit_id", "distance"])

    candidates = layer_hits[~layer_hits["hit_id"].isin(used_hit_ids)].copy()
    if candidates.empty:
        return candidates

    coords = candidates[["x", "y", "z"]].to_numpy(dtype=np.float64)
    distances = np.linalg.norm(coords - predicted_point[None, :], axis=1)
    candidates["distance"] = distances
    candidates = candidates[candidates["distance"] <= CANDIDATE_DISTANCE_MAX]
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("distance", kind="stable")
    return candidates.head(MAX_CANDIDATES).reset_index(drop=True)


def _apply_particle_filters(hits_df: pd.DataFrame, particle_filters: dict | None) -> pd.DataFrame:
    if not particle_filters:
        return hits_df

    filtered = hits_df
    pt_cut = particle_filters.get("pt")
    nhits_min = particle_filters.get("nhits_min")
    nhits_max = particle_filters.get("nhits_max")

    if pt_cut is not None:
        filtered = filtered[filtered["pt"] > pt_cut]
    if nhits_min is not None:
        filtered = filtered[filtered["nhits"] >= nhits_min]
    if nhits_max is not None:
        filtered = filtered[filtered["nhits"] <= nhits_max]

    return filtered


def _preprocess_event(event_dir: Path) -> pd.DataFrame:
    processor = EventProcessor(data_dir=event_dir.parent)
    processor.process(event_dir)
    hits_df = processor.hits.copy()
    if hits_df is None or hits_df.empty:
        raise RuntimeError("No hits found after processing event")

    # Ensure numeric types we rely on
    for col in ["hit_id", "particle_id", "unique_layer_id", "volume_id", "layer_id"]:
        hits_df[col] = hits_df[col].astype(int)
    return hits_df


def test_straight_line_propagation(
    event_dir: Path,
    num_tracks: int = 100,
    particle_filters: Optional[dict] = None,
) -> dict:
    log_file = Path("test_straight_line_propagation.log")
    set_log_file(log_file)
    logger.info("Starting straight-line propagation test on %s", event_dir)

    hits_df = _preprocess_event(event_dir)
    hits_df = _apply_particle_filters(hits_df, particle_filters)

    grouped = hits_df.groupby("particle_id", sort=False)
    particle_ids = list(grouped.groups.keys())
    if not particle_ids:
        raise RuntimeError("No particles available after filtering")

    rng = np.random.default_rng()
    if len(particle_ids) > num_tracks:
        particle_ids = rng.choice(particle_ids, size=num_tracks, replace=False)

    stats = {
        "total_tracks": 0,
        "evaluated_layers": 0,
        "truth_in_candidates": 0,
        "truth_is_best": 0,
        "rank_counts": Counter(),
        "distance_samples": [],
        "tracks": [],
    }

    for particle_id in particle_ids:
        track = grouped.get_group(particle_id).copy()
        track = track.sort_values(["unique_layer_id", "r", "z"], kind="stable")
        track = track.drop_duplicates(subset="unique_layer_id", keep="first")

        if len(track) <= SEED_LENGTH:
            continue

        stats["total_tracks"] += 1
        accepted_hits = track.iloc[:SEED_LENGTH].copy()
        used_hit_ids = set(accepted_hits["hit_id"].astype(int))

        track_summary = {
            "particle_id": int(particle_id),
            "num_hits": int(len(track)),
            "evaluations": [],
        }

        for idx in range(SEED_LENGTH, len(track)):
            target = track.iloc[idx]
            target_layer = int(target["unique_layer_id"])
            target_volume = int(target["volume_id"])
            target_layer_id = int(target["layer_id"])
            truth_hit_id = int(target["hit_id"])

            layer_hits = hits_df[
                (hits_df["unique_layer_id"] == target_layer)
                & (hits_df["volume_id"] == target_volume)
                & (hits_df["layer_id"] == target_layer_id)
            ]

            if layer_hits.empty:
                logger.debug(
                    "Particle %s layer %s (volume %s) has no hits in dataset; skipping step.",
                    particle_id,
                    target_layer,
                    target_volume,
                )
                continue

            points = accepted_hits[["x", "y", "z"]].to_numpy(dtype=np.float64)
            origin, direction = _fit_line(points)
            predicted_point = _intersect_layer(origin, direction, layer_hits)
            candidates = _prepare_candidates(predicted_point, layer_hits, used_hit_ids)

            if candidates.empty:
                track_summary["evaluations"].append(
                    {
                        "target_layer": target_layer,
                        "num_candidates": 0,
                        "truth_in_candidates": False,
                        "truth_rank": None,
                        "distance_to_truth": None,
                    }
                )
                continue

            distances = candidates["distance"].to_numpy(dtype=np.float64)
            hit_ids = candidates["hit_id"].astype(int).to_numpy()
            stats["evaluated_layers"] += 1

            truth_indices = np.where(hit_ids == truth_hit_id)[0]
            if truth_indices.size > 0:
                truth_rank = int(truth_indices[0]) + 1
                truth_in_candidates = True
                distance_to_truth = float(distances[truth_indices[0]])
                stats["truth_in_candidates"] += 1
                stats["rank_counts"][truth_rank] += 1
                stats["distance_samples"].append(distance_to_truth)
                if truth_rank == 1:
                    stats["truth_is_best"] += 1
            else:
                truth_rank = None
                truth_in_candidates = False
                stats["rank_counts"]["miss"] += 1
                distance_to_truth = None

            track_summary["evaluations"].append(
                {
                    "target_layer": target_layer,
                    "num_candidates": int(len(candidates)),
                    "truth_in_candidates": truth_in_candidates,
                    "truth_rank": truth_rank,
                    "distance_to_truth": distance_to_truth,
                }
            )

            # Append the truth hit to keep following layers realistic
            truth_hit_row = layer_hits[layer_hits["hit_id"] == truth_hit_id]
            if not truth_hit_row.empty:
                accepted_hits = pd.concat([accepted_hits, truth_hit_row.iloc[:1]], ignore_index=True)
                used_hit_ids.add(truth_hit_id)

        stats["tracks"].append(track_summary)

    return stats


def print_statistics(stats: dict) -> None:
    if not stats:
        print("No statistics to display")
        return

    total_tracks = stats.get("total_tracks", 0)
    evaluated_layers = stats.get("evaluated_layers", 0)
    truth_in_candidates = stats.get("truth_in_candidates", 0)
    truth_is_best = stats.get("truth_is_best", 0)

    print("\n" + "=" * 60)
    print("STRAIGHT-LINE PROPAGATION TEST RESULTS")
    print("=" * 60)
    print(f"Total tracks evaluated:        {total_tracks}")
    print(f"Layers with candidates:        {evaluated_layers}")
    if evaluated_layers:
        inclusion_pct = truth_in_candidates / evaluated_layers * 100.0
        best_pct = truth_is_best / evaluated_layers * 100.0
        print(f"Truth in candidate set:        {truth_in_candidates} ({inclusion_pct:.2f}%)")
        print(f"Truth ranked first:            {truth_is_best} ({best_pct:.2f}%)")

    rank_counts = stats.get("rank_counts", Counter())
    if rank_counts:
        print("\nTruth rank histogram:")
        ordered = sorted((k, v) for k, v in rank_counts.items() if isinstance(k, int))
        for rank, count in ordered:
            pct = count / evaluated_layers * 100.0 if evaluated_layers else 0.0
            print(f"  Rank {rank:2d}: {count:6d} ({pct:5.2f}%)")
        missed = rank_counts.get("miss", 0)
        if missed:
            miss_pct = missed / evaluated_layers * 100.0 if evaluated_layers else 0.0
            print(f"  Missed:  {missed:6d} ({miss_pct:5.2f}%)")

    distances = stats.get("distance_samples", [])
    finite_distances = [d for d in distances if d is not None and np.isfinite(d)]
    if finite_distances:
        mean_dist = float(np.mean(finite_distances))
        std_dist = float(np.std(finite_distances))
        print("\n|prediction - truth| distance (cm):")
        print(f"  Mean: {mean_dist:.3f}")
        print(f"  Std:  {std_dist:.3f}")

    print("\nDetailed logs saved to: test_straight_line_propagation.log")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test straight-line propagation accuracy")
    parser.add_argument(
        "event_dir",
        nargs="?",
        default=None,
        help="Path to TrackML event directory prefix (e.g., /path/to/event000001000). "
             "If omitted, a random event will be sampled.",
    )
    parser.add_argument("--num-tracks", type=int, default=100, help="Number of tracks to test (default: 100)")
    parser.add_argument("--pt-min", type=float, default=2.0, help="Minimum pT for particles (default: 2.0)")
    parser.add_argument("--nhits-min", type=int, default=3, help="Minimum number of hits (default: 3)")
    parser.add_argument("--nhits-max", type=int, default=20, help="Maximum number of hits (default: 20)")
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR),
                        help="Root directory containing part_* folders (default: Codalab dataset root)")
    parser.add_argument("--part", type=str, default=DEFAULT_PART,
                        help="Dataset part to sample from when event_dir is not provided (default: part_5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for random event selection")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON summary")

    args = parser.parse_args()

    if args.event_dir is None:
        base_dir = Path(args.base_dir)
        try:
            event_dir = select_random_event(base_dir, args.part, args.seed)
            print(f"Selected random event from {args.part}: {event_dir}")
        except FileNotFoundError as exc:
            print(f"Error selecting random event: {exc}")
            raise SystemExit(1)
    else:
        event_dir = Path(args.event_dir)

    particle_filters = {
        "pt": args.pt_min,
        "nhits_min": args.nhits_min,
        "nhits_max": args.nhits_max,
    }

    stats = test_straight_line_propagation(
        event_dir,
        num_tracks=args.num_tracks,
        particle_filters=particle_filters,
    )

    print_statistics(stats)

    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=lambda o: list(o) if isinstance(o, Counter) else o)
            print(f"Saved summary to {output_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save summary: {exc}")



