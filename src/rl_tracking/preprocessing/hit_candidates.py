
import numpy as np
import pandas as pd
from trackml.dataset import load_event
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, field
from pathlib import Path
import re
import os
import torch
# TODO read features from a config file

DEFAULT_FEATURES = ["r", "z", "prev_z", "prev_r"]
DEFAULT_NUM_COMPATIBLE_HITS = 3

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

    def process(self, event_dir:str|Path) -> [np.array, np.array]:
        filename = str(event_dir)
        try:
            hits, cells, particles, truth = load_event(filename)
        except FileNotFoundError:
            print(f"Error: The file '{filename}' does not exist.")
            raise
        except IOError as e:
            print(f"Error: An I/O error occurred while reading the file '{filename}'.")
            print(f"Details: {e}")
            raise

        hits = hits.merge(truth, on='hit_id')
        self.hits = hits.merge(particles, on='particle_id')
        self.hits['r'] = np.sqrt(self.hits['x'] ** 2 + self.hits['y'] ** 2)
        self.hits["pt"] = np.sqrt(self.hits.px ** 2 + self.hits.py ** 2)
        self.hits = self.hits[self.hits['pt'] > 2]
        self.hits = self.hits.sort_values(['r', 'z'])

        self._get_prev_hit()
        self._get_prev_prev_hit()
        self._get_next_hit()
        self._get_comp_hits()
        self.hits = self.hits.dropna()
        self.hits.to_csv('test.csv')
        selected_cols = [x for x in self.hits.columns if x in self.features or x.startswith('comp_')]

        return self.hits[selected_cols].to_numpy(), self.hits['label'].values

