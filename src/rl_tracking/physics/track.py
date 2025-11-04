import numpy as np
from rl_tracking.physics import seed
import pandas as pd
from rl_tracking.physics import propagate
from rl_tracking.physics import comp_hits_shaper
import json
from pathlib import Path
from rl_tracking.utils.logger import get_logger

logger = get_logger()

# Load data files relative to this module
PHYSICS_DIR = Path(__file__).parent
path_plans = pd.read_pickle(PHYSICS_DIR / 'theta_path_plan.pkl')

with open(PHYSICS_DIR / 'momentum_theta_path_plan_100.json') as f:
    new_path_plans = json.load(f)



class Helix:

# from https://github.com/livcms/mkFit/blob/devel/Propagation.cc

  def __init__(self, p, use_truth_path_plan=False):
   #x, y, z, px, py, pz, Q, layer, prev_x, prev_y, prev_z, prev_prev_x, prev_prev_y, prev_prev_z
   self.p = p
   """ Sets up track from the truth level seed and finds the corresponding path plan """
   self.seed_hits, ix_final_seed_hit = seed.select_seed_hits(p)
   self.pt, self.px, self.py, self.pz, self.Q, self.theta = seed.track_from_seed_hits(self.seed_hits)
   self.ipt = 1/self.pt
   self.phi = np.arctan2(self.py, self.px)
   #for debugging
   self.initial_phi = self.phi

   self.x, self.y, self.z, self.current_layer = p.iloc[ix_final_seed_hit][['x', 'y', 'z', 'unique_layer_id']].values
   self.r0 = np.sqrt(self.x**2+self.y**2)
   
   # Option to use truth-level path plan (all layers in track after seed) instead of lookup
   if use_truth_path_plan:
     # Get all unique layers from the track, sorted
     all_layers = sorted(p['unique_layer_id'].unique())
     # Filter to only layers after the seed (seed is in first 3 layers)
     # The seed ends at current_layer, so we want layers > current_layer
     seed_layers = all_layers[:3]  # First 3 layers are seed
     self.path_plan = np.array([layer for layer in all_layers if layer > self.current_layer])
     logger.debug(f"Using truth-level path plan (all layers after seed): {self.path_plan}")
   else:
     self.path_plan = get_sub_value(new_path_plans, self.pt, self.theta)
     logger.debug(f"Using lookup path plan (based on pt, theta): {self.path_plan}")
   
   # Store the initial starting layer (final seed layer) for reference
   # This is needed to correctly filter expected hits to only post-seed layers
   self.initial_start_layer = float(self.current_layer)
   # Initialize correct hit IDs from the particle data
   self.correct_hit_ids = set(p['hit_id'].values) if 'hit_id' in p.columns else set()
   # Store the full particle DataFrame for distance calculations
   self.particle_df = p
   logger.debug(f"Path plan (unique_layer_ids): {self.path_plan}")
   logger.debug(f"Current layer (unique_layer_id): {self.current_layer}")
   logger.debug(f"Initial start layer (final seed): {self.initial_start_layer}")
  
  def print_helix(self):
    """Print debug information about the helix state."""
    return f"x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, r0={self.r0:.3f}, layer={self.current_layer}"
  
  def get_correct_hitids_in_layer(self, layer):
    """
    Get correct hit IDs for a specific layer.
    
    CRITICAL: This method only returns hits from the CURRENT layer.
    This ensures NO forward-looking bias - the reward only knows about
    hits in the current layer, not future layers.
    
    Args:
        layer: The unique_layer_id to get hits for
        
    Returns:
        List of hit IDs from the specified layer only
    """
    if 'unique_layer_id' in self.p.columns:
      layer_hits = self.p[self.p['unique_layer_id'] == layer]
      if len(layer_hits) > 0 and 'hit_id' in layer_hits.columns:
        return [int(hit_id) for hit_id in layer_hits['hit_id'].values]
    return []
  
  def get_correct_hits_df_in_layer(self, layer):
    """Get DataFrame of correct hits for a specific layer."""
    if 'unique_layer_id' in self.p.columns:
      layer_hits = self.p[self.p['unique_layer_id'] == layer]
      if len(layer_hits) > 0:
        return layer_hits.copy()
    return pd.DataFrame()
  
  def get_closest_hit_to_prop(self, comp_hits):
    """Get the hit closest to the propagation."""
    if len(comp_hits) == 0:
      return None
    # Simple heuristic: return the first hit for now
    # You may want to implement actual distance calculation
    return comp_hits.iloc[0]

  def propagate_one_layer(self, hit_holder, use_distance_reward=False, max_layer_skip=4, tolerance_r=1.0):
    """
    Propagate to the next layer and find compatible hits.
    
    Args:
        hit_holder: HitHolder instance with all hits
        use_distance_reward: Whether to use distance-based rewards
        max_layer_skip: Maximum number of layers to skip ahead when looking for compatible hits.
                       If no compatible hits are found in the next max_layer_skip layers,
                       propagation will fail and return done=True.
                       Default is 4. Increase to be more persistent, decrease to fail faster.
        tolerance_r: Distance tolerance in cm for finding compatible hits.
                    Hits within this 3D distance from the propagated position are considered compatible.
                    Default is 1.0 cm. Increase to find more hits (but may include more false positives).
    """
    correct_in_comp = 0
    correct_is_best = 0
    done = False

    unexplored_layers = self.path_plan[self.path_plan > self.current_layer]

    if len(unexplored_layers) == 0:
      return [], 0, True, 0, 0

    # Attempt to find the next layer with comp hits
    # Try the immediate next layer first (most likely to have hits)
    # If no hits found, try up to max_layer_skip layers ahead
    # But don't give up - advance to next layer and continue if we've tried several
    comp_hits = []
    next_layer = None
    
    # Try immediate next layer first
    if len(unexplored_layers) > 0:
      next_layer = unexplored_layers[0]
      comp_hits = propagate.propagate_and_get_comp_hits(self, next_layer, hit_holder, tolerance_r=tolerance_r)
    
    # If no hits in immediate next layer, try skipping ahead (up to max_layer_skip)
    if len(comp_hits) == 0 and len(unexplored_layers) > 1:
      for i in range(1, min(max_layer_skip, len(unexplored_layers))):
        next_layer = unexplored_layers[i]
        comp_hits = propagate.propagate_and_get_comp_hits(self, next_layer, hit_holder, tolerance_r=tolerance_r)
        if len(comp_hits) > 0:
          break
    
    # If still no hits found, we advance to the next layer anyway (don't give up yet)
    # This allows the track to continue even if some layers don't have compatible hits
    # The environment will handle termination when the path_plan is truly exhausted
    if len(comp_hits) == 0:
      # Advance to next layer in path_plan even if no hits found
      # This prevents premature termination when tolerance is strict
      if next_layer is not None:
        self.current_layer = next_layer
        logger.debug(f"No compatible hits in layer {next_layer}, advancing anyway. Unexplored: {len(unexplored_layers)}")
      # Return empty comp_hits but NOT done=True - let environment decide based on path_plan
      return [], 0, False, 0, 0

    # Update current layer to the layer where we found hits
    self.current_layer = next_layer
    
    # CRITICAL: Only get correct hits for the CURRENT layer (next_layer)
    # This ensures NO forward-looking bias - rewards are based only on current layer
    # The model cannot see future layer information when making decisions
    self.correct_hit_ids_in_layer = self.get_correct_hitids_in_layer(next_layer)
    self.correct_hits_df_in_layer = self.get_correct_hits_df_in_layer(next_layer)
    correct_layer = self.p.unique_layer_id.unique()

    if len(comp_hits) > 0:
      # Shape the comp hits and find the best match
      comp_hits = comp_hits_shaper.reformat_comp_hits(comp_hits, self.correct_hit_ids_in_layer, hit_holder)
      best_hit = self.get_closest_hit_to_prop(comp_hits)

      if int(best_hit.hit_id) in self.correct_hit_ids:
        correct_is_best += 1

      correct_in_comp = sum(
        1 for hit_id in comp_hits.hit_id.values if int(hit_id) in self.correct_hit_ids_in_layer
      )

      # Calculate rewards for each hit - use distance or binary based on flag
      if use_distance_reward:
        rewards = [
          hit_holder.get_reward_distance(comp_hits.iloc[i], self.correct_hits_df_in_layer, helix_pos=(self.x, self.y, self.z))
          for i in range(len(comp_hits))
        ]
      else:
        rewards = [
          hit_holder.get_reward_binary(comp_hits.iloc[i], self.correct_hit_ids_in_layer)
          for i in range(len(comp_hits))
        ]
    else:
      comp_hits = []
      rewards = 0
      done = True

    return comp_hits, rewards, done, correct_in_comp, correct_is_best

def get_closest_value(values, target):
  """Returns the closest value in the list to the target."""
  values_array = np.array(list(map(float, values)))
  return values_array[np.argmin(np.abs(values_array - target))]


def get_sub_value(new_path_plans, pt, theta):
  """Returns the path plan based on closest values of pt and theta."""
  pt_key = str(round(pt, 1))
  theta_key = str(round(theta, 3))

  # Check if pt_key and theta_key exist
  if pt_key in new_path_plans:
    pt_sub = pt_key
  else:
    pt_sub = str(get_closest_value(new_path_plans.keys(), pt))

  if theta_key in new_path_plans[pt_sub]:
    theta_sub = theta_key
  else:
    theta_sub = str(get_closest_value(new_path_plans[pt_sub].keys(), theta))

  # Return the path plan for the closest pt_sub and theta_sub
  return np.array(new_path_plans[pt_sub][theta_sub])
