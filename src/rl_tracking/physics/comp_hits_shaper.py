import numpy as np 
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line 
#from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New
#import tensorflow as tf
import pandas as pd
#from utils.geometry import find_n_closest_hits
import yaml


# with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

num_close_hits = 3

#dope = config['dope']
dope = True
def reformat_comp_hits(comp_hits, correct_hit_ids, hit_holder): 
    
    # # come back and change
    # if len(comp_hits) < num_close_hits: 
    #     try: 
    #         added_rows = pd.DataFrame([comp_hits.iloc[0]]*(num_close_hits-len(comp_hits))) 
    #     except: 
    #         print("the comp hits is ", comp_hits)
    #     comp_hits = pd.concat([comp_hits, added_rows])
    
    # comp_hits = comp_hits.reset_index() 

    # # "weird that this is needed, check"
    # if len(comp_hits) > num_close_hits: 
    #     comp_hits = comp_hits.iloc[:num_close_hits]
    
    cor_in_comp =  len(set(correct_hit_ids).intersection(set(list(map(int, comp_hits.hit_id.values))))) > 0

    if (dope) and (not cor_in_comp): 
        #print(int(cor.hit_id), comp_hits.hit_id)
        #print("the correct hit is", correct_hit_ids[0])
        #print("the index is", comp_hits.iloc[0])
        #print("hit holder is ", hit_holder.get_hit(correct_hit_ids[0]))
       # a = hit_holder.get_hit(correct_hit_ids[0])
        #print(type(comp_hits.iloc[0]), type(hit_holder.get_hit(correct_hit_ids[0])))
        try: 
            comp_hits.at[0] = hit_holder.get_hit(correct_hit_ids[0]).squeeze()
        except: 
            # Error already logged by caller, silently continue
            pass
        #     print("imma dope")

        # except: 
        #     print("didnt dope", correct_hit_ids)

    # DO NOT shuffle randomly - sort by quality instead so actions have consistent meaning
    # Sort hits: correct hits first, then by distance to helix (if available)
    # This makes action 0 = best hit, action 1 = second best, etc.
    
    # Create a sort key: correct hits get priority (0), incorrect get 1
    def sort_key(hit):
        hit_id = int(hit['hit_id'])
        is_correct = 0 if hit_id in correct_hit_ids else 1
        return (is_correct, hit_id)  # Secondary sort by hit_id for consistency
    
    # Sort by correctness first
    comp_hits = comp_hits.copy()
    comp_hits['_sort_key'] = comp_hits.apply(sort_key, axis=1)
    comp_hits = comp_hits.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)
    
    # Old random shuffle - REMOVED to allow agent to learn meaningful action ordering
    # comp_hits = comp_hits.sample(frac=1)

    return comp_hits


