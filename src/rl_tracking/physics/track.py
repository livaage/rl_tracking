import numpy as np
from rl_tracking.physics import seed
import pandas as pd
from rl_tracking.physics import propagate
from rl_tracking.physics import comp_hits_shaper
import json

path_plans = pd.read_pickle('/Users/liv/rl_tracking/src/rl_tracking/physics/theta_path_plan.pkl')

new_path_plans = open("/Users/liv/rl_tracking/src/rl_tracking/physics/momentum_theta_path_plan_100.json")
new_path_plans = json.load(new_path_plans)



class Helix:

# from https://github.com/livcms/mkFit/blob/devel/Propagation.cc

  def __init__(self, p):
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
   self.path_plan = get_sub_value(new_path_plans, self.pt, self.theta)
   print(self.path_plan)

  def propagate_one_layer(self, hit_holder):
    correct_in_comp = 0
    correct_is_best = 0
    done = False

    unexplored_layers = self.path_plan[self.path_plan > self.current_layer]

    if len(unexplored_layers) == 0:
      return [], 0, True, 0, 0

    # Attempt to find the next layer with comp hits, try up to 4 layers
    comp_hits = []
    for i in range(min(4, len(unexplored_layers))):
      next_layer = unexplored_layers[i]
      comp_hits = propagate.propagate_and_get_comp_hits(self, next_layer, hit_holder)
      if len(comp_hits) > 0:
        break
    else:
      # If no compatible hits are found after 4 attempts, return with an error message
      print("still no compatible hits")
      print("helix is ", self.print_helix())
      return [], 0, True, 0, 0

    self.correct_hit_ids_in_layer = self.get_correct_hitids_in_layer(next_layer)
    correct_layer = self.p.unique_layer_id.unique()

    # If no comp hits found, and current layer is not correct, go to the next layer
    if len(comp_hits) == 0 and self.current_layer not in correct_layer:
      return self.propagate_one_layer(hit_holder)

    if len(comp_hits) > 0:
      # Shape the comp hits and find the best match
      comp_hits = comp_hits_shaper.reformat_comp_hits(comp_hits, self.correct_hit_ids_in_layer, hit_holder)
      best_hit = self.get_closest_hit_to_prop(comp_hits)

      if int(best_hit.hit_id) in self.correct_hit_ids:
        correct_is_best += 1

      correct_in_comp = sum(
        1 for hit_id in comp_hits.hit_id.values() if int(hit_id) in self.correct_hit_ids_in_layer
      )

      # Calculate rewards for each hit
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
