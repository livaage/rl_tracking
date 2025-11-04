import pandas as pd
import numpy as np
from rl_tracking.utils.logger import get_logger

logger = get_logger()


class HitHolder:
    """
    Wrapper class around a DataFrame to provide hit lookup and close hit finding methods.
    """
    
    def __init__(self, hits_df):
        """
        Initialize with a DataFrame of hits.
        
        Args:
            hits_df: DataFrame containing hits with columns like hit_id, x, y, z, etc.
        """
        self.hits = hits_df
    
    def find_close_hits(self, helix, tolerance_r=2, tolerance_z=0, target_layer=None, debug=True):
        """
        Find hits close to the helix position in 3D space.
        
        Args:
            helix: Helix object with x, y, z, phi attributes
            tolerance_r: Tolerance in radial distance (r = sqrt(x^2+y^2))
            tolerance_z: Tolerance in z direction
            target_layer: Optional layer ID to filter hits by (unique_layer_id)
            debug: If True, print debug information
            
        Returns:
            DataFrame of compatible hits
        """
        # Calculate 3D distance between helix position and hits
        if self.hits is None or len(self.hits) == 0:
            if debug:
                logger.debug("  DEBUG find_close_hits: No hits in hit_holder")
            return pd.DataFrame()
        
        # Start with all hits
        hits = self.hits.copy()
        
        if debug:
            logger.debug(f"\n  DEBUG find_close_hits: Starting with {len(hits)} total hits")
        
        # Filter by target layer if specified
        if target_layer is not None and 'unique_layer_id' in hits.columns:
            layer_hits = hits[hits['unique_layer_id'] == target_layer]
            if debug:
                logger.debug(f"  DEBUG find_close_hits: Target layer {target_layer} has {len(layer_hits)} hits")
                if len(layer_hits) > 0:
                    logger.debug(f"  DEBUG find_close_hits: Sample hit in layer: hit_id={layer_hits.iloc[0]['hit_id']}, x={layer_hits.iloc[0]['x']:.2f}, y={layer_hits.iloc[0]['y']:.2f}, z={layer_hits.iloc[0]['z']:.2f}")
            if len(layer_hits) == 0:
                if debug:
                    logger.debug(f"  DEBUG find_close_hits: No hits in target layer {target_layer}")
                return pd.DataFrame()
            hits = layer_hits
        
        # Get helix position
        helix_x = helix.x
        helix_y = helix.y
        helix_z = helix.z
        
        if debug:
            logger.debug(f"  DEBUG find_close_hits: Helix position: ({helix_x:.2f}, {helix_y:.2f}, {helix_z:.2f})")
            logger.debug(f"  DEBUG find_close_hits: Checking {len(hits)} hits")
        
        # Calculate distances
        dx = hits['x'] - helix_x
        dy = hits['y'] - helix_y
        dz = hits['z'] - helix_z
        
        # Calculate 3D distance
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Filter by tolerance - use a reasonable tolerance (e.g., 5cm for tracking)
        tolerance = tolerance_r if tolerance_r > 0 else 5.0  # 5cm default
        
        if debug:
            logger.debug(f"  DEBUG find_close_hits: Tolerance: {tolerance} cm")
            logger.debug(f"  DEBUG find_close_hits: Distance stats - min={distances.min():.2f}, max={distances.max():.2f}, mean={distances.mean():.2f} cm")
            # Show closest few hits
            closest_idx = distances.nsmallest(3).index
            for idx in closest_idx:
                hit = hits.loc[idx]
                logger.debug(f"  DEBUG find_close_hits: Closest hit: hit_id={hit['hit_id']}, dist={distances.loc[idx]:.2f} cm, pos=({hit['x']:.2f}, {hit['y']:.2f}, {hit['z']:.2f})")
        
        close_mask = distances <= tolerance
        n_close = close_mask.sum()
        
        if debug:
            logger.debug(f"  DEBUG find_close_hits: Found {n_close} hits within tolerance")
        
        if n_close > 0:
            return hits[close_mask].copy()
        else:
            return pd.DataFrame()
    
    def get_hit(self, hit_id):
        """
        Get a specific hit by ID.
        
        Args:
            hit_id: The hit ID
            
        Returns:
            Series or DataFrame row for the hit
        """
        hit = self.hits[self.hits['hit_id'] == hit_id]
        return hit.iloc[0] if len(hit) > 0 else None
    
    def get_reward_binary(self, hit, correct_hit_ids):
        """
        Get a binary reward for a hit (1 if correct, 0 otherwise).
        
        Args:
            hit: A hit (Series or row)
            correct_hit_ids: List of correct hit IDs
            
        Returns:
            1 if hit ID is in correct_hit_ids, 0 otherwise
        """
        if isinstance(hit, pd.Series):
            hit_id = int(hit['hit_id'])
        else:
            hit_id = int(hit.hit_id)
        
        return 1 if hit_id in correct_hit_ids else 0
    
    def get_reward_distance(self, hit, correct_hits_df, helix_pos=None):
        """
        Get a distance-based reward for a hit based on distance to correct hits.
        The reward is inversely proportional to the distance (closer = higher reward).
        
        Args:
            hit: A predicted hit (Series or row) with x, y, z coordinates
            correct_hits_df: DataFrame containing correct hits for this layer
            helix_pos: Optional (x, y, z) tuple of helix propagation position.
                      If None, uses the closest correct hit as reference.
            
        Returns:
            Reward value (higher for closer hits, normalized to [0, 1])
        """
        if correct_hits_df is None or len(correct_hits_df) == 0:
            return 0.0
        
        # Get predicted hit position
        if isinstance(hit, pd.Series):
            pred_x, pred_y, pred_z = hit['x'], hit['y'], hit['z']
        else:
            pred_x, pred_y, pred_z = hit.x, hit.y, hit.z
        
        # Calculate distance to each correct hit and take minimum
        dx = correct_hits_df['x'] - pred_x
        dy = correct_hits_df['y'] - pred_y
        dz = correct_hits_df['z'] - pred_z
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        min_distance = distances.min()
        
        # Convert distance to reward: use exponential decay with better scaling
        # Reward = exp(-distance / scale_factor) * reward_scale
        # Using scale_factor of 2.0 cm for better reward distribution
        # Scaling rewards up to prevent Q-value collapse
        scale_factor = 2.0
        reward_scale = 10.0  # Scale rewards up to make them more meaningful
        
        # Clip very large distances to prevent extremely small rewards
        # For distances > 20cm, use a minimum reward
        if min_distance > 20.0:
            reward = 0.01 * reward_scale  # Minimum reward scaled
        else:
            reward = np.exp(-min_distance / scale_factor) * reward_scale
        
        return float(reward)
    
    def get_distance_to_correct(self, hit, correct_hits_df):
        """
        Get the minimum distance from a predicted hit to any correct hit.
        
        Args:
            hit: A predicted hit (Series or row) with x, y, z coordinates
            correct_hits_df: DataFrame containing correct hits for this layer
            
        Returns:
            Minimum distance in cm (or np.inf if no correct hits)
        """
        if correct_hits_df is None or len(correct_hits_df) == 0:
            return np.inf
        
        # Get predicted hit position
        if isinstance(hit, pd.Series):
            pred_x, pred_y, pred_z = hit['x'], hit['y'], hit['z']
        else:
            pred_x, pred_y, pred_z = hit.x, hit.y, hit.z
        
        # Calculate distance to each correct hit and take minimum
        dx = correct_hits_df['x'] - pred_x
        dy = correct_hits_df['y'] - pred_y
        dz = correct_hits_df['z'] - pred_z
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return float(distances.min())