import numpy as np 
import pandas as pd 
import sys
sys.path.append('../')
#from material import apply_cms_multiple_scattering
#from Event import EventHolder
from numba import jit 
#from tracker.find_helix_hits import HitHolder
#from newProp import helixAtR, helixAtZ
from rl_tracking.utils.logger import get_logger

logger = get_logger()

allowed_layer_connections = pd.read_csv('/Users/liv/rl_tracking/src/rl_tracking/physics/allowed_layer_connections.csv')
layer_info = pd.read_csv('/Users/liv/rl_tracking/src/rl_tracking/physics/layer_info.csv', header=[0,1], index_col=0)
#path_plans = pd.read_pickle(BASE_DIR+'/utils/theta_path_plan.pkl')


def propagate_and_get_comp_hits(helix, layer, hit_holder): 

    comps = propagate_helix_to_layer(helix, layer, hit_holder)

    helix.current_layer = layer
    
    return comps




def propagate_helix_to_layer(helix, layer_id, hit_holder): 
    
    # layer_id should be the unique_layer_id from path_plan
    # Debug: show what we're receiving
    logger.debug(f"[DEBUG propagate_helix_to_layer] Received layer_id={layer_id} (type={type(layer_id)})")
    
    # layer_info.csv is indexed by unique_layer_id (1.0, 2.0, ..., 48.0)
    # So if layer_id is already a unique_layer_id, use it directly
    original_layer_id = int(layer_id)  # Don't subtract 1 - layer_info is 1-indexed
    
    # Store position before propagation
    pos_before = (helix.x, helix.y, helix.z)
    
    # Use .loc with unique_layer_id to index layer_info
    layer_row = layer_info.loc[original_layer_id]
    
    if original_layer_id < 10: 
        target_r = layer_row.r['median']
        helixAtR(helix, target_r)
        # Position after propagation
        pos_after = (helix.x, helix.y, helix.z)
        
        comps = hit_holder.find_close_hits(helix, 1, 0, target_layer=original_layer_id)
        
        # Debug output
        logger.debug(f"\n[PROPAGATION DEBUG] Layer {original_layer_id} (R propagation)")
        logger.debug(f"  Before: x={pos_before[0]:.2f}, y={pos_before[1]:.2f}, z={pos_before[2]:.2f}")
        logger.debug(f"  After:  x={pos_after[0]:.2f}, y={pos_after[1]:.2f}, z={pos_after[2]:.2f}")
        logger.debug(f"  Target r: {target_r:.2f}")
        
        # Find closest hit even if not within tolerance
        if hasattr(hit_holder, 'hits') and len(hit_holder.hits) > 0:
            layer_hits = hit_holder.hits[hit_holder.hits['unique_layer_id'] == original_layer_id]
            if len(layer_hits) > 0:
                # Show layer info vs actual hits
                z_min_info = layer_row.z['min']
                z_max_info = layer_row.z['max']
                z_median_info = layer_row.z['median']
                actual_z_min = layer_hits['z'].min()
                actual_z_max = layer_hits['z'].max()
                actual_z_mean = layer_hits['z'].mean()
                
                # Show volume_id and layer_id of actual hits
                if 'volume_id' in layer_hits.columns and 'layer_id' in layer_hits.columns:
                    unique_vol_layers = layer_hits[['volume_id', 'layer_id']].drop_duplicates()
                    vol_layer_str = ', '.join([f"vol={row['volume_id']}, lay={row['layer_id']}" 
                                              for _, row in unique_vol_layers.iterrows()])
                    logger.debug(f"  Actual hits volume_id/layer_id: {vol_layer_str}")
                
                logger.debug(f"  Layer info z: [{z_min_info:.2f}, {z_max_info:.2f}], median={z_median_info:.2f}")
                logger.debug(f"  Actual hits z: [{actual_z_min:.2f}, {actual_z_max:.2f}], mean={actual_z_mean:.2f}")
                if abs(actual_z_mean - z_median_info) > 100:
                    logger.warning(f"  *** MISMATCH: Layer info z_median={z_median_info:.2f} but actual hits z_mean={actual_z_mean:.2f} ***")
                    logger.warning(f"  *** Difference: {abs(actual_z_mean - z_median_info):.2f} cm - layer_info.csv may be from different data! ***")
                
                distances = np.sqrt(
                    (layer_hits['x'] - pos_after[0])**2 +
                    (layer_hits['y'] - pos_after[1])**2 +
                    (layer_hits['z'] - pos_after[2])**2
                )
                closest_idx = distances.idxmin()
                closest_hit = layer_hits.loc[closest_idx]
                closest_dist = distances.loc[closest_idx]
                logger.debug(f"  Closest hit: hit_id={closest_hit['hit_id']:.0f}, "
                      f"x={closest_hit['x']:.2f}, y={closest_hit['y']:.2f}, z={closest_hit['z']:.2f}, "
                      f"distance={closest_dist:.2f} cm")
                logger.debug(f"  Found {len(comps)} compatible hits within tolerance")
            else:
                logger.warning(f"  WARNING: No hits found with unique_layer_id={original_layer_id}")
                # Check what unique_layer_ids actually exist
                unique_layers = hit_holder.hits['unique_layer_id'].unique()
                logger.debug(f"  Available unique_layer_ids in data: {sorted(unique_layers)}")
                logger.debug(f"  Using unique_layer_id {original_layer_id} to index layer_info")

        #comps = propagate_helix_to_r(helix, layer_id, hits)
        #print("propagating to r=", layer_row.r['median'])
    else: 
        z_min = layer_row.z['min']
        z_max = layer_row.z['max']
        
        # Propagate to min z
        helixAtZ(helix, z_min)
        pos_after_min = (helix.x, helix.y, helix.z)
        comps1 = hit_holder.find_close_hits(helix, 1, 0, target_layer=original_layer_id)
        
        # Propagate to max z
        helixAtZ(helix, z_max)
        pos_after_max = (helix.x, helix.y, helix.z)
        comps2 = hit_holder.find_close_hits(helix, 1, 0, target_layer=original_layer_id)
        
        # Debug output for Z propagation
        logger.debug(f"\n[PROPAGATION DEBUG] Layer {original_layer_id} (Z propagation)")
        logger.debug(f"  Before: x={pos_before[0]:.2f}, y={pos_before[1]:.2f}, z={pos_before[2]:.2f}")
        logger.debug(f"  After (z_min={z_min:.2f}): x={pos_after_min[0]:.2f}, y={pos_after_min[1]:.2f}, z={pos_after_min[2]:.2f}")
        logger.debug(f"  After (z_max={z_max:.2f}): x={pos_after_max[0]:.2f}, y={pos_after_max[1]:.2f}, z={pos_after_max[2]:.2f}")
        
        # Find closest hits for both propagations
        if hasattr(hit_holder, 'hits') and len(hit_holder.hits) > 0:
            layer_hits = hit_holder.hits[hit_holder.hits['unique_layer_id'] == original_layer_id]
            if len(layer_hits) > 0:
                # Show layer info vs actual hits
                z_min_info = layer_row.z['min']
                z_max_info = layer_row.z['max']
                z_median_info = layer_row.z['median']
                actual_z_min = layer_hits['z'].min()
                actual_z_max = layer_hits['z'].max()
                actual_z_mean = layer_hits['z'].mean()
                
                # Show volume_id and layer_id of actual hits
                if 'volume_id' in layer_hits.columns and 'layer_id' in layer_hits.columns:
                    unique_vol_layers = layer_hits[['volume_id', 'layer_id']].drop_duplicates()
                    vol_layer_str = ', '.join([f"vol={row['volume_id']}, lay={row['layer_id']}" 
                                              for _, row in unique_vol_layers.iterrows()])
                    logger.debug(f"  Actual hits volume_id/layer_id: {vol_layer_str}")
                
                logger.debug(f"  Layer info z: [{z_min_info:.2f}, {z_max_info:.2f}], median={z_median_info:.2f}")
                logger.debug(f"  Actual hits z: [{actual_z_min:.2f}, {actual_z_max:.2f}], mean={actual_z_mean:.2f}")
                if abs(actual_z_mean - z_median_info) > 100:
                    logger.warning(f"  *** MISMATCH: Layer info z_median={z_median_info:.2f} but actual hits z_mean={actual_z_mean:.2f} ***")
                    logger.warning(f"  *** Difference: {abs(actual_z_mean - z_median_info):.2f} cm - layer_info.csv may be from different data! ***")
                
                # For min z
                dists_min = np.sqrt(
                    (layer_hits['x'] - pos_after_min[0])**2 +
                    (layer_hits['y'] - pos_after_min[1])**2 +
                    (layer_hits['z'] - pos_after_min[2])**2
                )
                closest_min_idx = dists_min.idxmin()
                closest_min_hit = layer_hits.loc[closest_min_idx]
                closest_min_dist = dists_min.loc[closest_min_idx]
                logger.debug(f"  Closest hit (z_min): hit_id={closest_min_hit['hit_id']:.0f}, "
                      f"x={closest_min_hit['x']:.2f}, y={closest_min_hit['y']:.2f}, z={closest_min_hit['z']:.2f}, "
                      f"distance={closest_min_dist:.2f} cm")
                
                # For max z
                dists_max = np.sqrt(
                    (layer_hits['x'] - pos_after_max[0])**2 +
                    (layer_hits['y'] - pos_after_max[1])**2 +
                    (layer_hits['z'] - pos_after_max[2])**2
                )
                closest_max_idx = dists_max.idxmin()
                closest_max_hit = layer_hits.loc[closest_max_idx]
                closest_max_dist = dists_max.loc[closest_max_idx]
                logger.debug(f"  Closest hit (z_max): hit_id={closest_max_hit['hit_id']:.0f}, "
                      f"x={closest_max_hit['x']:.2f}, y={closest_max_hit['y']:.2f}, z={closest_max_hit['z']:.2f}, "
                      f"distance={closest_max_dist:.2f} cm")
                
                logger.debug(f"  Found {len(comps1)} compatible hits (z_min), {len(comps2)} compatible hits (z_max)")
            else:
                logger.warning(f"  WARNING: No hits found with unique_layer_id={original_layer_id}")
                # Check what unique_layer_ids actually exist
                unique_layers = hit_holder.hits['unique_layer_id'].unique()
                logger.debug(f"  Available unique_layer_ids in data: {sorted(unique_layers)}")
                logger.debug(f"  Layer_info index {layer_id} corresponds to unique_layer_id {original_layer_id}") 

        if (len(comps1)> 0)  and (len(comps2)> 0): 
            comps = pd.concat([comps1, comps2])
            # could have found similar compatible hits
            comps = comps.drop_duplicates() 
        elif len(comps1) > 0: 
            comps = comps1
        elif len(comps2) > 0:
            comps = comps2  
        else: 
            logger.warning(f"No compatible hits found for layer {original_layer_id}: comps1={len(comps1)}, comps2={len(comps2)}")
            comps = []
        #print("propagating to z=",  layer_row.z['median'])
        # z is double layered, so propagate to both? 
        #comps = propagate_helix_to_z(helix, layer_id, hits)
        # was 0.5 sensititivity (way too many i s)
    return comps



layer_info = pd.read_csv('/Users/liv/rl_tracking/src/rl_tracking/physics/layer_info.csv', header=[0,1], index_col=0)

#@jit 
def helixAtZ(helix, zout): 
    """ 
    Propagate helix to z position.
    
    Note: helix positions are in cm, momentum in GeV/c
    The factor of 100 converts units: GeV/c -> appropriate units for cm coordinates
    https://github.com/trackreco/mkFit/blob/15d90864b677a08cbebc514141a7de2de19394d4/mkFit/PropagationMPlex.cc#L604-L713
    """
    sol = 0.299792458  # speed of light in GeV·cm/Tesla (approximately)
    Bfield = 2  # Tesla
    # k factor: converts from GeV/c momentum to curvature radius in cm
    # The *100 factor accounts for unit conversion (GeV -> appropriate units for cm)
    k = -helix.Q*100/(sol*Bfield)

    pt = helix.pt

    cosP = np.cos(helix.phi)
    sinP = np.sin(helix.phi)
    cosT = np.cos(helix.theta) 
    sinT = np.sin(helix.theta)

    pxin = cosP*pt 
    pyin  = sinP*pt 

    deltaZ = zout - helix.z
    alpha = deltaZ * sinT * helix.ipt/(cosT*k)
    cosa = np.cos(alpha)
    sina = np.sin(alpha)

    helix.x = helix.x + k* (pxin * sina - pyin * (1-cosa))
    helix.y = helix.y + k* (pyin * sina + pxin*(1-cosa))
    helix.z = zout 
    helix.phi = helix.phi + alpha  

    #helix.ipt = material.apply_cms_multiple_scattering(helix.ipt, helix.theta)

#@jit
def helixAtR(helix, r): 
    """Propagate helix to radius r.
    
    Note: helix positions are in cm, momentum in GeV/c
    The factor of 100 converts units: GeV/c -> appropriate units for cm coordinates
    https://github.com/trackreco/mkFit/blob/15d90864b677a08cbebc514141a7de2de19394d4/mkFit/PropagationMPlex.cc#L259-L405
    """
    pt = helix.pt
    #pt = np.sqrt(helix.px**2 + helix.py**2 + helix.pz**2)
    #helix.ipt = 1/pt 
    cosPorT = np.cos(helix.phi)
    sinPorT = np.sin(helix.phi) 
    sol = 0.299792458  # speed of light in GeV·cm/Tesla (approximately)
    Bfield = 2  # Tesla
    # k factor: converts from GeV/c momentum to curvature radius in cm
    # The *100 factor accounts for unit conversion (GeV -> appropriate units for cm)
    k = -helix.Q*100/(sol*Bfield)
    kinv = 1/k 
    
    pxin = cosPorT*pt 
    pyin  = sinPorT*pt

    for i in range(5):
        r0 = np.sqrt(helix.x**2 + helix.y**2)
        ialpha = (r-r0)*helix.ipt/k 
        cosah = np.cos(ialpha/2)
        sinah = np.sin(ialpha/2)
        helix.x = helix.x + 2*k*sinah*(pxin*cosah - pyin*sinah)
        helix.y = helix.y + 2*k*sinah*(pyin*cosah  + pxin*sinah) 
        cosT = np.cos(helix.theta)
        sinT = np.sin(helix.theta)
        helix.z = helix.z + k*ialpha*cosT/(helix.ipt*sinT)
        #oodotp = r0 * pt/(pxin * helix.x + pyin*helix.y)
        #id = (r-r0) * oodotp 
        #id = r- r0
        #D += id 
        #cosah = np.cos(id * helix.ipt * kinv * 0.5)
        #sinah = np.sin(id * helix.ipt * kinv * 0.5)
        cosa = 1 - 2*sinah*sinah
        sina = 2*sinah*cosah
        #cosa = np.cos(id * helix.ipt * kinv)
        #sina = np.sin(id * helix.ipt * kinv)
        #alpha = D *helix.ipt * kinv 

        #helix.x = helix.x + k*(pxin*sina - pyin*(1-cosa))
        #helix.y = helix.y + k*(pyin*sina + pxin*(1-cosa))
        #helix.x = helix.x + 2*k*sinah*(pxin*cosah - pyin*sinah)
        #helix.y = helix.y + 2*k*sinah*(pyin*cosah + pxin*sinah)
        #helix.z = helix.z + k*alpha*cosPorT*pt*sinPorT 
        #a = np.cos(helix.theta)
        #b = np.sin(helix.theta)
        #helix.z = helix.z +k*alpha*a*b
        #helix.phi = helix.phi + alpha 

        pxinold = pxin

        pxin = pxin*cosa - pyin*sina
        pyin = pyin*cosa + pxinold*sina
        helix.phi = helix.phi + ialpha 

    #helix.z =  helix.z + k*alpha*cosPorT*pt/sinPorT 

    #print("developed in r to ", helix.x, helix.y, helix.z)

