import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete
from rl_tracking.physics.track import Helix

COLS = ['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id',
       'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight', 'vx',
       'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits', 'r', 'pt']


class TrackingEnv:
    def __init__(self, data_loader):
        """Initialize the environment with a data loader and a column for step count."""
        self.data_loader = iter(data_loader)  # Streaming file list
        self.current_data = None  # Stores the data from the current file
        self.unique_values = []  # Tracks remaining unique values for steps
        self.particle_index = 0  # Tracks current step within the file
        self.load_next_file()  # Load the first file
        self.observation_space = Box(low=0, high=120, shape=(3, 4), dtype=np.float32)
        self.action_space = Discrete(3)  # Example: 2 discrete actions
        self.hit_number = 0

    def _perform_particle_selections(self):
        pt_cut = 2
        self.current_data = self.current_data[self.current_data['pt'] > pt_cut]
        self.current_data = self.current_data[self.current_data['nhits'] > 3]
        self.current_data = self.current_data[self.current_data['nhits'] < 20]


    def load_next_file(self):
        """Load the next file and compute the number of steps from the column."""
        try:
            self.current_data = next(self.data_loader)  # Get new file's data
            self.current_data = pd.DataFrame(
                self.current_data.view(self.current_data.shape[1], self.current_data.shape[2])
                , columns = COLS)
            self._perform_particle_selections()
            self.accepted_particle_ids = np.random.shuffle(self.current_data['particle_id'].unique())
            self.particle_index = 0  # Reset index
        except StopIteration:
            self.current_data, self.unique_values = None, []  # End of stream

    def reset(self):
        """Reset environment to the start of the current file."""
        if not self.unique_values:  # If the file is exhausted, load a new one
            self.load_next_file()
            self.particle_index = 0
        if self.current_data is None:
            return None  # No more data

        if self.hit_number > len(self.current_track):
            self.get_new_track()
            self.hit_number = 0
        return self.get_current_state()

    def get_new_track(self):
        self.current_track = self.current_data[
            self.current_data['particle_id'] == self.accepted_particle_ids[self.particle_index]]
        self.particle_index += 1
        self.helix = Helix(self.current_track)

    def get_current_state(self):
        """Return the current step’s data."""
        if self.particle_index < len(self.unique_values):
            self.current_track = self.current_data[self.current_data['particle_id'] == self.accepted_particle_ids[self.particle_index]]

        return None

    def step(self, action):
        """Take an action and move to the next unique value in the column."""

        comp_hits, rewards, done, correct_in_comp, correct_is_best = (
            self.helix.propagate_one_layer(self.current_data))
        self.hit_number +=1
        self.particle_index += 1
        if self.particle_index >= len(self.unique_values):  # If all steps are done, load a new file
            self.load_next_file()

        next_state = self.get_current_state()
        reward = self.calculate_reward(next_state)  # Define reward logic
        done = next_state is None  # If no more data, episode ends

        return next_state, reward, done

    def calculate_reward(self, state):
        """Placeholder for reward logic (adjust based on your task)."""
        return 1 if state is not None else 0  # Example: reward for valid step