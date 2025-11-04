"""
Test script to verify track propagation accuracy.

This script tests:
1. What percentage of the time it propagates to the correct layer
2. How often the correct hit is within the selected hits
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

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
                    helix.propagate_one_layer(hit_holder)
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
                
                # Check if correct hit is in selected hits
                if correct_in_comp > 0:
                    stats['correct_hit_in_selected'] += 1
                
                # Check if correct hit is the best hit
                if correct_is_best > 0:
                    stats['correct_hit_is_best'] += 1
                
                track_stats['propagations'].append({
                    'target_layer': int(next_layer),
                    'success': True,
                    'correct_layer': propagated_to_correct_layer,
                    'num_compatible_hits': len(comp_hits),
                    'correct_hit_in_selected': correct_in_comp > 0,
                    'correct_hit_is_best': correct_is_best > 0
                })
                
                # Update helix layer
                helix.current_layer = next_layer
                
                if done:
                    break
            
            stats['tracks_tested'].append(track_stats)
            
        except Exception as e:
            logger.error(f"Error testing track {particle_id}: {e}")
            continue
    
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
    parser.add_argument('event_dir', type=str, help='Path to TrackML event directory (e.g., /path/to/event000001000)')
    parser.add_argument('--num-tracks', type=int, default=100, help='Number of tracks to test (default: 100)')
    parser.add_argument('--pt-min', type=float, default=2.0, help='Minimum pT for particles (default: 2.0)')
    parser.add_argument('--nhits-min', type=int, default=3, help='Minimum number of hits (default: 3)')
    parser.add_argument('--nhits-max', type=int, default=20, help='Maximum number of hits (default: 20)')
    
    args = parser.parse_args()
    
    particle_filters = {
        'pt': args.pt_min,
        'nhits_min': args.nhits_min,
        'nhits_max': args.nhits_max
    }
    
    stats = test_propagation_on_tracks(
        args.event_dir,
        num_tracks=args.num_tracks,
        particle_filters=particle_filters
    )
    
    print_statistics(stats)

