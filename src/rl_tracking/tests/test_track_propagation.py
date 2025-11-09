# flake8: noqa
"""
Test script to verify track propagation accuracy.

This script tests:
1. What percentage of the time it propagates to the correct layer
2. How often the correct hit is within the selected hits
"""
import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_BASE_DIR = Path("/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data")
DEFAULT_PART = "part_5"
PROPAGATION_TOLERANCE = 1.0


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
    candidates = []
    for pattern in patterns:
        candidates.extend(part_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No event files found in {part_dir}")

    rng = np.random.default_rng(seed)
    hits_file = rng.choice(candidates)
    return _event_prefix_from_hits(hits_file)

try:
    from tqdm import tqdm
except ImportError:
    # tqdm is optional - if not available, just use a simple iterator
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.physics.track import Helix
from rl_tracking.physics.hit_holder import HitHolder
from rl_tracking.utils.logger import get_logger, set_log_file

logger = get_logger()


def test_propagation_on_tracks(event_dir, num_tracks=100, particle_filters=None):
    """
    Test track propagation on a set of tracks.
    
    Args:
        event_dir: Path to TrackML event directory
        num_tracks: Number of tracks to test
        particle_filters: Dict of particle filtering criteria
    
    Returns:
        Dictionary with statistics
    """
    # Set up logging to a test-specific file
    log_file = Path('test_propagation.log')
    set_log_file(log_file)
    logger.info(f"Starting propagation test on {event_dir}")
    
    # Process event
    data_dir = Path(event_dir).parent
    processor = EventProcessor(data_dir=data_dir)
    processor.process(event_dir)
    
    # Access the processed hits DataFrame directly from processor
    hits_df = processor.hits.copy()
    
    if hits_df is None or len(hits_df) == 0:
        logger.error("No hits found after processing event")
        return None
    
    # Apply particle filters if provided
    if particle_filters:
        if particle_filters.get('pt'):
            hits_df = hits_df[hits_df['pt'] > particle_filters['pt']]
        if particle_filters.get('nhits_min'):
            hits_df = hits_df[hits_df['nhits'] >= particle_filters['nhits_min']]
        if particle_filters.get('nhits_max'):
            hits_df = hits_df[hits_df['nhits'] <= particle_filters['nhits_max']]
    
    # Get unique particles
    unique_particles = hits_df['particle_id'].unique()
    
    if len(unique_particles) == 0:
        logger.error("No particles found after filtering")
        return None
    
    # Limit number of tracks to test
    num_tracks = min(num_tracks, len(unique_particles))
    test_particles = np.random.choice(unique_particles, size=num_tracks, replace=False)
    
    logger.info(f"Testing propagation on {num_tracks} tracks")
    
    # Statistics
    stats = {
        'total_tracks': 0,
        'successful_propagations': 0,
        'correct_layer_propagations': 0,
        'correct_hit_in_selected': 0,
        'correct_hit_is_best': 0,
        'layer_mismatches': [],
        'propagation_failures': 0,
        'rank_correct_counts': defaultdict(int),
        'rank_total_counts': defaultdict(int),
        'correct_layer_miss_hit_ids': set(),
        'correct_layer_miss_total': 0,
        'correct_layer_miss_outside_tolerance': 0,
        'correct_layer_miss_within_tolerance': 0,
        'correct_layer_miss_no_truth_hits': 0,
        'correct_layer_miss_no_candidates': 0,
        'correct_layer_miss_distances': [],
        'tracks_tested': []
    }
    
    hit_holder = HitHolder(hits_df)
    
    for particle_id in tqdm(test_particles, desc="Testing tracks"):
        # Get hits for this particle
        particle_hits = hits_df[hits_df['particle_id'] == particle_id].copy()
        
        if len(particle_hits) < 3:
            continue  # Skip tracks with too few hits
        
        stats['total_tracks'] += 1
        
        try:
            # Create helix
            helix = Helix(particle_hits)
            correct_layers = set(particle_hits['unique_layer_id'].unique())
            
            # Track propagation statistics for this track
            track_stats = {
                'particle_id': particle_id,
                'num_hits': len(particle_hits),
                'correct_layers': sorted(correct_layers),
                'propagations': []
            }
            
            # Propagate through layers
            max_propagations = min(10, len(helix.path_plan) - 1)  # Limit to 10 steps
            
            for step in range(max_propagations):
                # Get unexplored layers
                unexplored_layers = helix.path_plan[helix.path_plan > helix.current_layer]
                
                if len(unexplored_layers) == 0:
                    break  # Reached end of path plan
                
                # Try to propagate to next layer
                next_layer = unexplored_layers[0]
                comp_hits, rewards, done, correct_in_comp, correct_is_best = (
                    helix.propagate_one_layer(
                        hit_holder,
                        tolerance_r=PROPAGATION_TOLERANCE,
                    )
                )
                
                # Check if propagation was successful
                if done and len(comp_hits) == 0:
                    stats['propagation_failures'] += 1
                    track_stats['propagations'].append({
                        'target_layer': int(next_layer),
                        'success': False,
                        'correct_layer': next_layer in correct_layers,
                        'num_compatible_hits': 0,
                        'correct_hit_in_selected': False,
                        'correct_hit_is_best': False
                    })
                    break  # Propagation failed
                
                if len(comp_hits) == 0:
                    if next_layer in correct_layers:
                        stats['correct_layer_miss_total'] += 1
                        truth_hits_df = helix.get_correct_hits_df_in_layer(next_layer)
                        if truth_hits_df is not None and len(truth_hits_df) > 0:
                            try:
                                miss_hit_ids = truth_hits_df['hit_id'].astype(int).tolist()
                                stats['correct_layer_miss_hit_ids'].update(miss_hit_ids)
                            except Exception:
                                pass
                            if {"x", "y", "z"}.issubset(truth_hits_df.columns):
                                dx = truth_hits_df["x"] - helix.x
                                dy = truth_hits_df["y"] - helix.y
                                dz = truth_hits_df["z"] - helix.z
                                min_distance = float(np.sqrt(dx**2 + dy**2 + dz**2).min())
                                stats['correct_layer_miss_distances'].append(min_distance)
                                if min_distance > PROPAGATION_TOLERANCE:
                                    stats['correct_layer_miss_outside_tolerance'] += 1
                                else:
                                    stats['correct_layer_miss_within_tolerance'] += 1
                            else:
                                stats['correct_layer_miss_no_truth_hits'] += 1
                        else:
                            stats['correct_layer_miss_no_truth_hits'] += 1
                        stats['correct_layer_miss_no_candidates'] += 1
                    continue  # No compatible hits found, try next layer
                
                stats['successful_propagations'] += 1
                
                # Check if we propagated to the correct layer
                propagated_to_correct_layer = next_layer in correct_layers
                if propagated_to_correct_layer:
                    stats['correct_layer_propagations'] += 1
                else:
                    stats['layer_mismatches'].append({
                        'particle_id': particle_id,
                        'target_layer': int(next_layer),
                        'correct_layers': sorted(correct_layers),
                        'current_layer': int(helix.current_layer)
                    })
                
                correct_rank = None
                if correct_in_comp > 0:
                    stats['correct_hit_in_selected'] += 1
                    try:
                        hit_ids = comp_hits['hit_id'].astype(int).tolist()
                    except Exception:
                        hit_ids = list(comp_hits['hit_id'])
                    for idx, hit_id in enumerate(hit_ids):
                        try:
                            hit_int = int(hit_id)
                        except Exception:
                            continue
                        if hit_int in helix.correct_hit_ids_in_layer:
                            correct_rank = idx
                            stats['rank_total_counts'][idx] = (
                                stats['rank_total_counts'].get(idx, 0) + 1
                            )
                            break
                
                # Check if correct hit is the best hit
                if correct_is_best > 0:
                    stats['correct_hit_is_best'] += 1
                    stats['rank_correct_counts'][0] = (
                        stats['rank_correct_counts'].get(0, 0) + 1
                    )
                elif correct_rank is not None:
                    stats['rank_correct_counts'][correct_rank] = (
                        stats['rank_correct_counts'].get(correct_rank, 0) + 1
                    )
                else:
                    if next_layer in correct_layers:
                        stats['correct_layer_miss_total'] += 1
                        truth_hits_df = helix.correct_hits_df_in_layer
                        if truth_hits_df is not None and len(truth_hits_df) > 0:
                            try:
                                miss_hit_ids = truth_hits_df['hit_id'].astype(int).tolist()
                                stats['correct_layer_miss_hit_ids'].update(miss_hit_ids)
                            except Exception:
                                pass
                            if {"x", "y", "z"}.issubset(truth_hits_df.columns):
                                dx = truth_hits_df["x"] - helix.x
                                dy = truth_hits_df["y"] - helix.y
                                dz = truth_hits_df["z"] - helix.z
                                min_distance = float(np.sqrt(dx**2 + dy**2 + dz**2).min())
                                stats['correct_layer_miss_distances'].append(min_distance)
                                if min_distance > PROPAGATION_TOLERANCE:
                                    stats['correct_layer_miss_outside_tolerance'] += 1
                                else:
                                    stats['correct_layer_miss_within_tolerance'] += 1
                            else:
                                stats['correct_layer_miss_no_truth_hits'] += 1
                        else:
                            stats['correct_layer_miss_no_truth_hits'] += 1
                
                track_stats['propagations'].append({
                    'target_layer': int(next_layer),
                    'success': True,
                    'correct_layer': propagated_to_correct_layer,
                    'num_compatible_hits': len(comp_hits),
                    'correct_hit_in_selected': correct_in_comp > 0,
                    'correct_hit_is_best': correct_is_best > 0,
                    'correct_hit_rank': correct_rank
                })
                
                # Update helix layer
                helix.current_layer = next_layer
                
                if done:
                    break
            
            stats['correct_layer_miss_distances'] = [
                float(value) for value in stats['correct_layer_miss_distances']
            ]
            stats['tracks_tested'].append(track_stats)
            
        except Exception as e:
            logger.error(f"Error testing track {particle_id}: {e}")
            continue
    
    stats['rank_correct_counts'] = dict(stats['rank_correct_counts'])
    stats['rank_total_counts'] = dict(stats['rank_total_counts'])
    stats['correct_layer_miss_hit_ids'] = sorted(
        int(hit_id) for hit_id in stats['correct_layer_miss_hit_ids']
    )
    return stats


def print_statistics(stats):
    """Print formatted statistics."""
    if stats is None:
        print("No statistics to display")
        return
    
    print("\n" + "="*60)
    print("TRACK PROPAGATION TEST RESULTS")
    print("="*60)
    
    total = stats['total_tracks']
    if total == 0:
        print("No tracks were tested")
        return
    
    print(f"\nTotal tracks tested: {total}")
    print(f"Successful propagations: {stats['successful_propagations']}")
    print(f"Propagation failures: {stats['propagation_failures']}")
    
    # Layer accuracy
    if stats['successful_propagations'] > 0:
        layer_accuracy = (stats['correct_layer_propagations'] / stats['successful_propagations']) * 100
        print(f"\nLayer Accuracy:")
        print(f"  Propagated to correct layer: {stats['correct_layer_propagations']} / {stats['successful_propagations']} ({layer_accuracy:.2f}%)")
    
    # Hit selection accuracy
    if stats['successful_propagations'] > 0:
        hit_selection_accuracy = (stats['correct_hit_in_selected'] / stats['successful_propagations']) * 100
        best_hit_accuracy = (stats['correct_hit_is_best'] / stats['successful_propagations']) * 100
        
        print(f"\nHit Selection Accuracy:")
        print(f"  Correct hit in selected hits: {stats['correct_hit_in_selected']} / {stats['successful_propagations']} ({hit_selection_accuracy:.2f}%)")
        print(f"  Correct hit is best hit: {stats['correct_hit_is_best']} / {stats['successful_propagations']} ({best_hit_accuracy:.2f}%)")
    
    rank_totals = stats.get('rank_total_counts', {})
    if rank_totals:
        print("\nCorrect-hit rank distribution:")
        logger.info("Correct-hit rank distribution:")
        for rank in sorted(rank_totals):
            total_rank = rank_totals[rank]
            correct_rank_hits = stats.get('rank_correct_counts', {}).get(rank, 0)
            accuracy_pct = (correct_rank_hits / total_rank * 100) if total_rank else 0.0
            print(f"  Rank {rank}: {correct_rank_hits} / {total_rank} ({accuracy_pct:.2f}%)")
            logger.info(
                "Rank %s: correct %d / %d (%.2f%%)",
                rank,
                correct_rank_hits,
                total_rank,
                accuracy_pct,
            )
    
    missing_total = stats.get('correct_layer_miss_total', 0)
    if missing_total:
        print("\nCorrect-layer miss diagnostics:")
        logger.info("Correct-layer miss diagnostics:")
        print(f"  Misses while on correct layer: {missing_total}")
        logger.info("  Misses while on correct layer: %d", missing_total)
        outside_tol = stats.get('correct_layer_miss_outside_tolerance', 0)
        within_tol = stats.get('correct_layer_miss_within_tolerance', 0)
        no_truth_hits = stats.get('correct_layer_miss_no_truth_hits', 0)
        no_candidates = stats.get('correct_layer_miss_no_candidates', 0)
        print(f"    Outside tolerance: {outside_tol}")
        logger.info("    Outside tolerance: %d", outside_tol)
        print(f"    Within tolerance: {within_tol}")
        logger.info("    Within tolerance: %d", within_tol)
        print(f"    Truth hits unavailable: {no_truth_hits}")
        logger.info("    Truth hits unavailable: %d", no_truth_hits)
        print(f"    No candidates returned: {no_candidates}")
        logger.info("    No candidates returned: %d", no_candidates)
        distances = stats.get('correct_layer_miss_distances', [])
        if distances:
            avg_distance = float(np.mean(distances))
            max_distance = float(np.max(distances))
            print(f"      Distance stats (cm): avg={avg_distance:.2f}, max={max_distance:.2f}")
            logger.info(
                "      Distance stats (cm): avg=%.2f, max=%.2f",
                avg_distance,
                max_distance,
            )
    
    # Layer mismatches
    if stats['layer_mismatches']:
        print(f"\nLayer Mismatches: {len(stats['layer_mismatches'])}")
        print("  Sample mismatches (first 5):")
        for mm in stats['layer_mismatches'][:5]:
            print(f"    Particle {mm['particle_id']}: target={mm['target_layer']}, "
                  f"correct_layers={mm['correct_layers']}, current={mm['current_layer']}")
    
    print("\n" + "="*60)
    print(f"Detailed logs saved to: test_propagation.log")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test track propagation accuracy')
    parser.add_argument(
        'event_dir',
        nargs='?',
        default=None,
        help='Path to TrackML event directory prefix (e.g., /path/to/event000001000). '
             'If omitted, a random event will be sampled.'
    )
    parser.add_argument('--num-tracks', type=int, default=100, help='Number of tracks to test (default: 100)')
    parser.add_argument('--pt-min', type=float, default=2.0, help='Minimum pT for particles (default: 2.0)')
    parser.add_argument('--nhits-min', type=int, default=3, help='Minimum number of hits (default: 3)')
    parser.add_argument('--nhits-max', type=int, default=20, help='Maximum number of hits (default: 20)')
    parser.add_argument('--base-dir', type=str, default=str(DEFAULT_BASE_DIR),
                        help='Root directory containing part_* folders (default: Codalab dataset root)')
    parser.add_argument('--part', type=str, default=DEFAULT_PART,
                        help='Dataset part to sample from when event_dir is not provided (default: part_5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for random event selection')
    parser.add_argument('--missed-hit-ids-output', type=str, default=None,
                        help='Optional path to save JSON array of hit IDs missed while on the correct layer')
    
    args = parser.parse_args()
    
    if args.event_dir is None:
        base_dir = Path(args.base_dir)
        try:
            event_dir = select_random_event(base_dir, args.part, args.seed)
            print(f"Selected random event from {args.part}: {event_dir}")
        except FileNotFoundError as exc:
            print(f"Error selecting random event: {exc}")
            sys.exit(1)
    else:
        event_dir = Path(args.event_dir)
    
    particle_filters = {
        'pt': args.pt_min,
        'nhits_min': args.nhits_min,
        'nhits_max': args.nhits_max
    }
    
    stats = test_propagation_on_tracks(
        event_dir,
        num_tracks=args.num_tracks,
        particle_filters=particle_filters,
    )
    
    print_statistics(stats)
    
    if args.missed_hit_ids_output:
        try:
            output_path = Path(args.missed_hit_ids_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(stats.get('correct_layer_miss_hit_ids', []), f, indent=2)
            print(f"Saved missed hit IDs to {output_path}")
        except Exception as exc:
            print(f"Failed to save missed hit IDs: {exc}")

