import pandas as pd
import torch
from torch.utils.data import IterableDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

class RLKernelIterableDataset(IterableDataset):
    def __init__(self, event_numbers: list[str], event_processor=None):
        super().__init__()
        self.event_numbers = event_numbers
        self.event_processor = event_processor
        self.sample_count = 0
        self.epoch_size = 10024

    def _read_file(self, event_number):
        # Efficiently read a file in chunks if needed
        sample, label = self.event_processor.process(event_number)
        for i in range(len(sample)):
            yield sample[i], label[i]
            self.sample_count += 1

    def __iter__(self):
        self.sample_count = 0
        for file in self.event_numbers:
            yield from self._read_file(file)
            if self.epoch_size is not None and self.sample_count >= self.epoch_size:
                return

class TrackingDataModule(LightningDataModule):
    def __init__(self, file_paths: list[str] |Path, batch_size: int, event_processor=None, num_workers: int = 1, val_split:float = 0.2):
        super().__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.event_processor = event_processor
        self.num_workers = num_workers
        file_names = Path(file_paths).iterdir()
        self.event_numbers = list(set([str(x).split("-")[0] for x in file_names]))
        self.val_split = val_split

    def setup(self, stage=None):
        # Split event numbers into train and validation sets
        train_events, val_events = train_test_split(self.event_numbers, test_size=self.val_split)

        if stage == 'fit' or stage is None:
            self.train_dataset = RLKernelIterableDataset(train_events, event_processor=self.event_processor)
            self.val_dataset = RLKernelIterableDataset(val_events, event_processor=self.event_processor)

        if stage == 'test' or stage is None:
            # Use validation set as test set (or split further)
            self.test_dataset = RLKernelIterableDataset(val_events, event_processor=self.event_processor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)  # Optionally drop the last incomplete batch

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)  # Optionally drop the last incomplete batch

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)  # Optionally drop the last incomplete batch