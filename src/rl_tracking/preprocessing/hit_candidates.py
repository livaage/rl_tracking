
import numpy as np
import pandas as pd
from trackml.dataset import load_event
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, field
from pathlib import Path
import re
import os
import torch
from rl_tracking.utils.logger import get_logger

logger = get_logger()
# TODO read features from a config file

DEFAULT_FEATURES = ["r", "z", "prev_z", "prev_r"]
DEFAULT_NUM_COMPATIBLE_HITS = 3

GEOMETRY_DIR = Path(__file__).resolve().parent.parent / 'geometry'


def _resolve_geometry_path(filename: str) -> Path:
    """
    Resolve geometry assets with graceful fallback to the package root.

    Args:
        filename: Name of the geometry file.

    Returns:
        Path to the requested file.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    candidates = [
        GEOMETRY_DIR / filename,
        Path(__file__).resolve().parent.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate '{filename}'. Locations checked: {searched}")


# Load layer remapping file using shared geometry directory
LAYER_REMAPPING_PATH = _resolve_geometry_path('tml_layer_remap.csv')
LAYER_REMAPPING = pd.read_csv(LAYER_REMAPPING_PATH)

@dataclass
class EventProcessor:
    data_dir: Path
    num_comps: int = DEFAULT_NUM_COMPATIBLE_HITS
    features: list[str] = field(default_factory=lambda: DEFAULT_FEATURES)

    def __post_init__(self):

        if not self.data_dir.exists():
            raise ValueError(f"The path {self.data_dir} does not exist.")
        # Normalize the path (e.g., remove redundant separators)
        self.data_dir = self.data_dir.resolve()

    def _get_prev_hit(self):
        self.hits[['prev_r', 'prev_z']] = self.hits.groupby('particle_id').shift(1)[['r', 'z']]

    def _get_prev_prev_hit(self):
        self.hits[['prev_prev_r', 'prev_prev_z']] = self.hits.groupby('particle_id').shift(2)[['r', 'z']]

    def _get_next_hit(self):
        self.hits[['next_r', 'next_z', 'next_x', 'next_y', 'next_hit_id']] = self.hits.groupby('particle_id').shift(-1)[['r', 'z', 'x', 'y', 'hit_id']]

    def _remove_seed_hits(self):
        self.hits = self.hits.sort_values(['r', 'r'])
        seeds = self.hits.groupby('particle_id').head(3)
        self.hits = self.hits[~self.hits['hit_id'].isin(seeds.hit_id)]

    def _get_comp_hits(self):
        hits_wo_last = self.hits.groupby('particle_id').head(-1)

        nbrs = NearestNeighbors(n_neighbors=self.num_comps + 1).fit(
            hits_wo_last[["x", "y", "z"]])  # Add 1 to exclude self
        distances, indices = nbrs.kneighbors(hits_wo_last[["next_x", "next_y", "next_z"]].rename(columns={'next_x': 'x', 'next_y': 'y', 'next_z': 'z'}))
        correct_hit = indices[:, 0]
        # Remove the first column (self) from the results
        neighbor_r = hits_wo_last.iloc[indices[:,1:].flatten()]['r'].values.reshape((indices.shape[0], indices.shape[1]-1))
        neighbor_z = hits_wo_last.iloc[indices[:,1:].flatten()]['z'].values.reshape((indices.shape[0], indices.shape[1]-1))
        # need to add the correct next hit position


        sorted_order = np.lexsort((neighbor_z, neighbor_r))
        sorted_indices = np.take_along_axis(indices, sorted_order, axis=1)

        positions_in_sorted = np.argwhere(sorted_indices == correct_hit[:, None])

        # Now, assign the sorted neighbors' r and z values back to the dataframe
        for i in range(self.num_comps):
            hits_wo_last[f'comp_{i + 1}_r'] = hits_wo_last.iloc[sorted_indices[:, i]]['r'].values
            hits_wo_last[f'comp_{i + 1}_z'] = hits_wo_last.iloc[sorted_indices[:, i]]['z'].values

        self.hits = hits_wo_last
        self.hits['label'] = positions_in_sorted[:,1]

    def _get_pixel_tracks(self):
        pass

    def _add_cols(self):
        # Drop the Unnamed: 0 column if it exists
        remapping_cols = LAYER_REMAPPING.drop(['Unnamed: 0'], axis=1, errors='ignore') if 'Unnamed: 0' in LAYER_REMAPPING.columns else LAYER_REMAPPING
        
        # Check for missing volume_id/layer_id combinations
        unique_vol_layers = self.hits[['volume_id', 'layer_id']].drop_duplicates()
        remapped_vol_layers = remapping_cols[['volume_id', 'layer_id']].drop_duplicates()
        
        missing = unique_vol_layers.merge(remapped_vol_layers, on=['volume_id', 'layer_id'], how='left', indicator=True)
        missing = missing[missing['_merge'] == 'left_only'][['volume_id', 'layer_id']]
        
        if len(missing) > 0:
            logger.warning(f"{len(missing)} volume_id/layer_id combinations not found in remapping:")
            logger.warning(str(missing.head(10)))
        
        self.hits = self.hits.merge(remapping_cols, on=['volume_id', 'layer_id'], how='left')
        
        # Check for NaN unique_layer_ids
        nan_count = self.hits['unique_layer_id'].isna().sum()
        if nan_count > 0:
            logger.warning(f"{nan_count} hits have NaN unique_layer_id after remapping")
            nan_vol_layers = self.hits[self.hits['unique_layer_id'].isna()][['volume_id', 'layer_id']].drop_duplicates()
            logger.warning(f"Missing combinations: {nan_vol_layers}")
        
        logger.debug(f"Remapping applied: {len(self.hits)} hits, unique_layer_id range: [{self.hits['unique_layer_id'].min():.0f}, {self.hits['unique_layer_id'].max():.0f}]")
        self.hits['theta'] = np.arctan2(self.hits.r, self.hits.z)
        self.hits['phi'] = np.arctan2(self.hits.y, self.hits.x)

    def process(self, event_dir:str|Path) -> [np.array, np.array]:
        filename = str(event_dir)
        try:
            hits, cells, particles, truth = load_event(filename)
        except FileNotFoundError:
            logger.error(f"Error: The file '{filename}' does not exist.")
            raise
        except IOError as e:
            logger.error(f"Error: An I/O error occurred while reading the file '{filename}'.")
            logger.error(f"Details: {e}")
            raise

        hits = hits.merge(truth, on='hit_id')
        self.hits = hits.merge(particles, on='particle_id')
        
        # TrackML uses mm, but layer_info.csv and propagation expect cm
        # Convert positions from mm to cm
        self.hits['x'] = self.hits['x'] * 0.1  # mm to cm
        self.hits['y'] = self.hits['y'] * 0.1  # mm to cm
        self.hits['z'] = self.hits['z'] * 0.1  # mm to cm
        
        self.hits['r'] = np.sqrt(self.hits['x'] ** 2 + self.hits['y'] ** 2)
        self.hits["pt"] = np.sqrt(self.hits.px ** 2 + self.hits.py ** 2)
        logger.debug(f"Hits statistics:\n{self.hits.describe()}")
        self.hits = self.hits.sort_values(['r', 'z'])
        self._add_cols()

        # self._get_prev_hit()
        # self._get_prev_prev_hit()
        # self._get_next_hit()
        # self._get_comp_hits()
        # self.hits = self.hits[self.hits['pt'] > 2]
        # self.hits = self.hits[~self.hits['volume_id'].isin([7,8,9])]
        # self.hits = self.hits.dropna()
        # self.hits.to_csv('test.csv')
        # selected_cols = [x for x in self.hits.columns if x in self.features or x.startswith('comp_')
        #                  or x in ['pt', 'px', 'py', 'pz']]
        # return self.hits[selected_cols].to_numpy(), self.hits['label'].values
        return self.hits.to_numpy()



