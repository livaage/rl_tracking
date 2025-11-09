from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trackml.dataset import load_event


@dataclass(slots=True)
class StepObservation:
    rank: int
    num_candidates: int
    volume_id: int
    layer_id: int
    particle_id: int
    step_index: int


@dataclass(slots=True)
class StepDetail(StepObservation):
    history: np.ndarray
    candidates: np.ndarray
    candidate_ids: np.ndarray
    prediction: np.ndarray
    truth_candidate_col: int
    truth_position: np.ndarray
    example_label: str


def _fit_straight_line(history: np.ndarray, next_t: float) -> np.ndarray:
    """Return the extrapolated next point using a per-axis linear fit."""
    t = np.arange(history.shape[0], dtype=np.float64)
    a = np.vstack([t, np.ones_like(t)]).T
    pred = []
    for dim in range(history.shape[1]):
        coeffs, _, _, _ = np.linalg.lstsq(a, history[:, dim], rcond=None)
        pred.append(coeffs[0] * next_t + coeffs[1])
    return np.array(pred, dtype=np.float64)


def _predict_next(history: Sequence[np.ndarray]) -> np.ndarray:
    """Predict the next hit position from a sequence of previous hits."""
    history_arr = np.asarray(history, dtype=np.float64)
    if history_arr.shape[0] < 3:
        # Fall back to a simple extrapolation using the last delta.
        direction = history_arr[-1] - history_arr[-2]
        return history_arr[-1] + direction
    return _fit_straight_line(history_arr, next_t=history_arr.shape[0])


def _prepare_hits(event_prefix: Path) -> pd.DataFrame:
    hits, cells, particles, truth = load_event(event_prefix)
    merged = (
        hits.merge(truth, on="hit_id", how="inner")
        .merge(particles, on="particle_id", how="left")
        .sort_values(
            ["particle_id", "volume_id", "layer_id", "hit_id"],
            kind="stable",
        )
    )
    # Cylindrical radius is convenient for ordering along the track.
    merged["r"] = np.sqrt(merged.x**2 + merged.y**2)
    merged["abs_z"] = merged.z.abs()
    merged = merged.sort_values(
        ["particle_id", "volume_id", "layer_id", "r", "abs_z", "z"],
        kind="stable",
    )
    merged["pt"] = np.sqrt(merged.tpx**2 + merged.tpy**2)
    merged = merged[merged["pt"] > 2].reset_index(drop=True)

    group = merged.groupby("particle_id", sort=False)
    merged[["prev_x", "prev_y", "prev_z"]] = group[["x", "y", "z"]].shift(1)
    merged[["next_x", "next_y", "next_z"]] = group[["x", "y", "z"]].shift(-1)
    merged["next_hit_id"] = group["hit_id"].shift(-1)

    return merged.drop(columns="abs_z")


def evaluate_event(
    event_prefix: Path,
    *,
    min_history: int,
    num_candidates: int,
    example_min_rank: int,
    start_only: bool,
) -> tuple[list[StepObservation], list[StepDetail]]:
    hits = _prepare_hits(event_prefix)

    observations: list[StepObservation] = []
    details: list[StepDetail] = []

    for particle_id, group in hits.groupby("particle_id", sort=False):
        if len(group) < min_history + 1:
            continue

        positions = group[["x", "y", "z"]].to_numpy()
        volumes = group["volume_id"].to_numpy()
        layers = group["layer_id"].to_numpy()

        for idx in range(min_history, len(group)):
            if start_only and idx > min_history:
                break
            history = positions[idx - min_history:idx]
            volume_id = int(volumes[idx])
            layer_id = int(layers[idx])

            target_row_id = group.index[idx]
            layer_hits = hits[
                (hits["volume_id"] == volume_id)
                & (hits["layer_id"] == layer_id)
            ]
            if layer_hits.empty:
                continue

            candidate_positions_full = layer_hits[["x", "y", "z"]].to_numpy()
            layer_indices = layer_hits.index.to_numpy()
            truth_locs = np.where(layer_indices == target_row_id)[0]
            if truth_locs.size == 0:
                continue
            truth_loc = int(truth_locs[0])

            prediction = _predict_next(history)
            dists = np.linalg.norm(
                candidate_positions_full - prediction,
                axis=1,
            )
            order = np.argsort(dists, kind="stable")

            selected_order: list[int] = []
            for o in order:
                selected_order.append(o)
                if len(selected_order) >= num_candidates:
                    break
            if truth_loc not in selected_order:
                selected_order.append(truth_loc)

            selected_order = sorted(
                selected_order,
                key=lambda o: dists[o],
            )

            candidate_positions = candidate_positions_full[selected_order]
            candidate_indices = layer_indices[selected_order]

            truth_positions = np.where(
                np.array(selected_order) == truth_loc
            )[0]
            truth_rank = int(truth_positions[0]) + 1

            detail = StepDetail(
                rank=truth_rank,
                num_candidates=len(candidate_positions),
                volume_id=volume_id,
                layer_id=layer_id,
                particle_id=int(particle_id),
                step_index=idx,
                history=history.copy(),
                candidates=candidate_positions.copy(),
                candidate_ids=candidate_indices.copy(),
                prediction=prediction,
                truth_candidate_col=int(truth_positions[0]),
                truth_position=hits.loc[
                    target_row_id, ["x", "y", "z"]
                ].to_numpy(),
                example_label=(
                    f"{event_prefix.name} – particle {particle_id}, layer "
                    f"{layer_id}"
                ),
            )
            observations.append(detail)
            if truth_rank >= example_min_rank:
                details.append(detail)

    if not details:
        details = observations  # fall back to all details if threshold not met

    return observations, list(details)


def summarise(
    observations: Iterable[StepObservation],
    top_k: int = 20,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for obs in observations:
        rank_bucket = obs.rank if obs.rank <= top_k else top_k + 1
        counts[rank_bucket] += 1
    return counts


def plot_rank_distribution(
    counts: Counter[int],
    top_k: int,
    output_path: Path,
) -> None:
    labels = list(range(1, top_k + 1)) + [f">{top_k}"]
    values = [counts.get(r, 0) for r in range(1, top_k + 1)]
    values.append(counts.get(top_k + 1, 0))

    total = sum(values)
    proportions = [v / total if total else 0 for v in values]

    positions = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(positions, proportions, color="#1f77b4", width=0.9)
    ax.set_ylabel("Fraction of steps")
    ax.set_xlabel(
        "Rank of truth hit under straight-line extrapolation"
    )
    ax.set_ylim(0, max(proportions) * 1.1 if total else 1)
    ax.set_title("Truth Hit Rank Distribution (Straight-Line Predictor)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, prop in zip(bars, proportions):
        if prop > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{prop:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_example_step(example: StepDetail, output_path: Path) -> None:
    if example is None:
        raise RuntimeError("No example step was recorded; cannot plot.")

    hist = example.history
    candidates = example.candidates
    prediction = example.prediction
    truth_pos = example.truth_position
    truth_candidate = candidates[example.truth_candidate_col]

    radii_hist = np.linalg.norm(hist[:, :2], axis=1)
    radii_candidates = np.linalg.norm(candidates[:, :2], axis=1)
    radius_truth = np.linalg.norm(truth_pos[:2])
    radius_pred = np.linalg.norm(prediction[:2])
    z_hist = hist[:, 2]
    z_candidates = candidates[:, 2]
    z_truth = truth_pos[2]
    z_pred = prediction[2]

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(12, 5))

    ax_xy.scatter(
        candidates[:, 0],
        candidates[:, 1],
        c="#bbbbbb",
        label="Candidates",
        alpha=0.6,
    )
    ax_xy.scatter(
        truth_candidate[0],
        truth_candidate[1],
        marker="*",
        c="#d62728",
        s=160,
        label="Truth hit",
        zorder=5,
    )
    ax_xy.scatter(
        prediction[0],
        prediction[1],
        marker="x",
        c="#1f77b4",
        s=120,
        label="Straight-line extrapolation",
        zorder=6,
    )
    ax_xy.plot(
        hist[:, 0],
        hist[:, 1],
        "-o",
        c="#2ca02c",
        label="Seeds",
        zorder=4,
    )
    ax_xy.plot(
        [hist[-1, 0], prediction[0]],
        [hist[-1, 1], prediction[1]],
        "--",
        c="#1f77b4",
        alpha=0.7,
    )
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_xy.set_title(
        f"XY View – {example.example_label} (rank={example.rank})"
    )
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.set_aspect("equal", "box")
    ax_xy.grid(alpha=0.3)

    ax_rz.scatter(z_candidates, radii_candidates, c="#bbbbbb", alpha=0.6)
    ax_rz.scatter(
        z_truth,
        radius_truth,
        marker="*",
        c="#d62728",
        s=160,
        label="Truth hit",
        zorder=5,
    )
    ax_rz.scatter(
        z_pred,
        radius_pred,
        marker="x",
        c="#1f77b4",
        s=120,
        label="Straight-line extrapolation",
        zorder=6,
    )
    ax_rz.plot(
        z_hist,
        radii_hist,
        "-o",
        c="#2ca02c",
        label="Seeds",
        zorder=4,
    )
    ax_rz.plot(
        [z_hist[-1], z_pred],
        [radii_hist[-1], radius_pred],
        "--",
        c="#1f77b4",
        alpha=0.7,
    )
    ax_rz.set_xlabel("z [mm]")
    ax_rz.set_ylabel("r [mm]")
    ax_rz.set_title("R–Z View")
    ax_rz.legend(loc="best", fontsize=8)
    ax_rz.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise how a straight-line extrapolation ranks the truth hit "
            "relative to hits on the next detector layer."
        )
    )
    parser.add_argument(
        "events",
        nargs="+",
        type=Path,
        help="Event prefixes (without -hits/-truth suffixes).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Bucket ranks above this value into a single bar.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=3,
        help="Number of past hits used to define the straight line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("straight_line_rank_distribution.png"),
        help="Where to store the generated plot.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=20,
        help=(
            "Number of compatible hits to consider "
            "(including the truth hit)."
        ),
    )
    parser.add_argument(
        "--example-output",
        type=Path,
        default=Path("straight_line_example.png"),
        help="Where to store the illustrative candidate plot.",
    )
    parser.add_argument(
        "--example-min-rank",
        type=int,
        default=10,
        help=(
            "Prefer examples whose truth rank meets "
            "or exceeds this value."
        ),
    )
    parser.add_argument(
        "--example-target-rank",
        type=int,
        default=None,
        help="If set, pick the example with this rank (exact match).",
    )
    parser.add_argument(
        "--example-target-index",
        type=int,
        default=0,
        help=(
            "When multiple examples match the target rank, pick the "
            "nth occurrence (0-based)."
        ),
    )
    parser.add_argument(
        "--example-random",
        action="store_true",
        help="Pick a random example (after applying rank filters).",
    )
    parser.add_argument(
        "--example-random-seed",
        type=int,
        default=None,
        help="Seed used when choosing a random example.",
    )
    parser.add_argument(
        "--include-all-steps",
        action="store_true",
        help="Evaluate every propagation step instead of just the first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_observations: list[StepObservation] = []
    candidate_examples: list[StepDetail] = []

    for event_path in args.events:
        observations, details = evaluate_event(
            event_path,
            min_history=args.min_history,
            num_candidates=args.num_candidates,
            example_min_rank=args.example_min_rank,
            start_only=not args.include_all_steps,
        )
        all_observations.extend(observations)
        candidate_examples.extend(details)

    if not all_observations:
        raise RuntimeError(
            "No observations were gathered; check event paths and filters."
        )

    counts = summarise(all_observations, top_k=args.top_k)
    plot_rank_distribution(counts, top_k=args.top_k, output_path=args.output)

    chosen_example: StepDetail | None = None
    first_step_matches = [
        detail
        for detail in candidate_examples
        if detail.step_index == args.min_history
    ]

    if args.example_target_rank is not None:
        matches = [
            detail
            for detail in candidate_examples
            if detail.rank == args.example_target_rank
        ]
        if matches:
            index = min(args.example_target_index, len(matches) - 1)
            chosen_example = matches[index]
    else:
        pool_source = first_step_matches or candidate_examples
        pool = [
            detail
            for detail in pool_source
            if detail.rank >= args.example_min_rank
        ]
        if args.example_random and pool:
            rng = random.Random(args.example_random_seed)
            chosen_example = rng.choice(pool)
        elif pool:
            chosen_example = max(pool, key=lambda d: d.rank)

    if chosen_example is None and candidate_examples:
        chosen_example = max(candidate_examples, key=lambda d: d.rank)

    if chosen_example is not None:
        plot_example_step(chosen_example, output_path=args.example_output)
        example_msg = (
            f"Example plot uses truth rank {chosen_example.rank} "
            f"({chosen_example.example_label})."
        )
    else:
        example_msg = (
            "No example plot generated "
            "(truth hit missing from candidate sets)."
        )

    total = sum(counts.values())
    head = sum(counts[r] for r in range(1, min(6, args.top_k + 1)))
    processed_msg = (
        f"Processed {total} propagation steps across "
        f"{len(args.events)} events."
    )
    print(processed_msg)
    print(
        f"Truth hit is top-1 in {counts.get(1, 0) / total:.1%} of steps."
    )
    print(
        f"Truth hit is in top-5 in {head / total:.1%} of steps."
    )
    print(f"Plot saved to {args.output.resolve()}")
    print(example_msg)


if __name__ == "__main__":
    main()
