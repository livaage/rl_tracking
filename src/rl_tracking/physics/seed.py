import math 
import numpy as np

# this uses truth level seeding! 
from numba import jit


def track_from_seed_hits(seed_hits): 

    """
    Inputs: Two arrays containing x and y values of three hits 
    Returns: Estimated pT using conformal hit 
    Taken from : https://github.com/trackreco/mkFit/blob/a081c4dfa629fc958ef974115618b5818d727c55/mkFit/ConformalUtilsMPlex.cc
    
    """
    x = seed_hits['x'].values
    y = seed_hits['y'].values
    z = seed_hits['z'].values 

    pT, px, py, pz, Q, theta = accelerated_seed(x, y, z)

    return pT, px, py, pz, Q, theta


#@jit
def accelerated_seed(x, y, z): 
    initphi = np.abs(np.arctan2(y[0], x[0]))
    xtou = initphi < np.pi/4 or initphi > 3*np.pi/4 

    if xtou: 
        u = x/(x**2+y**2) 
        v = y/(x**2+y**2)
    else: 
        v = x/(x**2+y**2) 
        u = y/(x**2+y**2)


    B = v

    A = np.array([np.ones(len(x)), -np.array(u), -np.array(u**2)]).T
    #A = np.array(np.array([]))
    A = np.linalg.inv(A)
    C = np.matmul(A, B)#
    b = 1/(2*C[0])
    a = b * C[1]

    if xtou: 
        vrx = x[0] - a 
        vry = y[0] - b 
    else: 
        vrx = x[0] - b 
        vry = y[0] - a 

    R = np.sqrt(vrx**2 + vry**2)
    sol = 0.299792458  # speed of light in GeV·cm/Tesla (approximately)
    Bfield = 2  # Tesla
    # k factor: R is in cm, we need pT in GeV/c
    # The /100 factor converts from cm-based units to GeV/c
    k = sol*Bfield/100
    pT = R*k


    px = math.copysign(k*vry, (x[2] - x[0]))  
    py = math.copysign(k*vrx, (y[2] - y[0]))
    pz = (pT * (z[2] - z[0])) / np.sqrt((x[2]-x[0])**2 + (y[2] - y[0])**2)
    
    pi = np.sqrt(px**2+py**2)
    r = np.sqrt(x[0]**2+y[0]**2)
    #theta = np.arctan2(r, z[0])
    theta = np.arctan2(pT, pz)
    #theta 
    Q = -pi/(0.2998 * R * 2)


    if (y[2]-y[0])*(x[2]-x[1])>(y[2]-y[1])*(x[2]-x[0]): 
        Q = 1 
    else: 
        Q = -1 

    
    return pT, px, py, pz, Q, theta


def select_seed_hits(p): 
    """Because there could be double hits, we need to select the first seed hits carefully to not mess up the momentum estimation. The momentum is quite wrong if estimated from a double hit in a layer
    Quick fix - do the average of the position if there are several hits in a layer"""

    unique_layers = p.unique_layer_id.unique()
    
    # Ensure we have at least 3 unique layers for the seed
    # This should be filtered at the particle selection level, but check here as a safety
    if len(unique_layers) < 3:
        # Return None or empty result instead of raising error, so caller can handle gracefully
        # This should rarely happen if filtering is working correctly
        raise ValueError(f"Track has only {len(unique_layers)} unique layer(s), but at least 3 are required for seeding. This particle should have been filtered out during selection.")
    
    three_first_layers = unique_layers[:3]
    potential_seed_hits = p[p['unique_layer_id'].isin(three_first_layers)]
    #correct index here shouldn't matter, it just doens't need to be one of the wrong ones 
    averaged_seeds = potential_seed_hits[['x', 'y', 'z', 'r', 'phi', 'unique_layer_id', 'particle_id']].groupby('unique_layer_id').mean()
    
    # Ensure we have exactly 3 layers after averaging (should be guaranteed, but double-check)
    if len(averaged_seeds) < 3:
        raise ValueError(f"Only {len(averaged_seeds)} unique layers after averaging, but 3 are required. This should not happen if there are >= 3 unique layers initially.")
    
    # Take exactly the first 3 layers (in case there are more)
    averaged_seeds = averaged_seeds.head(3)
    
    # Assign one hit_id per unique layer (take first hit from each layer)
    # This ensures the hit_id array matches the number of unique layers in averaged_seeds
    averaged_seeds['hit_id'] = [potential_seed_hits[potential_seed_hits['unique_layer_id'] == layer_id].index[0] 
                                 for layer_id in averaged_seeds.index]
    b = p.reset_index() 
    pos_final_hit = b[b['x']==potential_seed_hits.iloc[-1].x].index[0]

    return averaged_seeds, pos_final_hit