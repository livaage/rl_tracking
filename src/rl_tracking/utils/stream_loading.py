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
        # Remove epoch_size limit to use all events
        self.epoch_size = None  # Use all events instead of limiting

    def _read_file(self, event_number):
        # Efficiently read a file in chunks if needed
        sample = self.event_processor.process(event_number)
        yield sample
        self.sample_count += 1

    def __iter__(self):
        self.sample_count = 0
        # Iterate through all events in the dataset
        for file in self.event_numbers:
            yield from self._read_file(file)
            # Removed epoch_size limit - use all events

class TrackingDataModule(LightningDataModule):
    def __init__(self, file_paths: list[str] |Path, batch_size: int, event_processor=None, num_workers: int = 0, val_split:float = 0.2, test_split:float = 0.1):
        super().__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.event_processor = event_processor
        self.num_workers = num_workers
        file_names = Path(file_paths).iterdir()
        self.event_numbers = list(set([str(x).split("-")[0] for x in file_names]))
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        # CRITICAL: Use consistent random_state=42 for all splits to ensure reproducibility
        # and that train/val/test are always separate
        RANDOM_STATE = 42
        
        if self.test_split <= 0.0 or self.test_split >= 1.0:
            raise ValueError(f"test_split must be between 0.0 and 1.0, got {self.test_split}")
        if self.val_split <= 0.0 or self.val_split >= 1.0:
            raise ValueError(f"val_split must be between 0.0 and 1.0, got {self.val_split}")
        
        train_ratio = 1.0 - self.val_split - self.test_split
        if train_ratio <= 0:
            raise ValueError(f"val_split + test_split must be < 1.0, got {self.val_split + self.test_split}")
        
        # CRITICAL: Always perform the same split regardless of stage
        # This ensures train/val/test are consistently separated
        # Sort event_numbers for reproducibility
        sorted_event_numbers = sorted(self.event_numbers)
        
        # First split: train vs (val+test)
        train_events, temp_events = train_test_split(
            sorted_event_numbers,
            test_size=(self.val_split + self.test_split),
            random_state=RANDOM_STATE
        )
        
        # Second split: val vs test
        val_test_ratio = self.val_split / (self.val_split + self.test_split)
        val_events, test_events = train_test_split(
            temp_events,
            test_size=(1 - val_test_ratio),
            random_state=RANDOM_STATE
        )
        
        # Verify no overlap between splits (safety check)
        train_set = set(train_events)
        val_set = set(val_events)
        test_set = set(test_events)
        
        if train_set & val_set:
            raise ValueError(f"CRITICAL: Train and Val sets overlap! Overlapping events: {sorted(train_set & val_set)[:10]}")
        if train_set & test_set:
            raise ValueError(f"CRITICAL: Train and Test sets overlap! Overlapping events: {sorted(train_set & test_set)[:10]}")
        if val_set & test_set:
            raise ValueError(f"CRITICAL: Val and Test sets overlap! Overlapping events: {sorted(val_set & test_set)[:10]}")
        
        # Log split information for verification
        print(f"\n{'='*60}")
        print(f"DATA SPLIT VERIFICATION (stage={stage})")
        print(f"{'='*60}")
        print(f"Train events: {len(train_events)} ({len(train_events)/len(sorted_event_numbers):.2%})")
        print(f"Val events: {len(val_events)} ({len(val_events)/len(sorted_event_numbers):.2%})")
        print(f"Test events: {len(test_events)} ({len(test_events)/len(sorted_event_numbers):.2%})")
        print(f"Total events: {len(sorted_event_numbers)}")
        print(f"Overlap checks: Train-Val={len(train_set & val_set)}, Train-Test={len(train_set & test_set)}, Val-Test={len(val_set & test_set)}")
        print(f"✓ All splits are separate (no overlap)")
        print(f"{'='*60}\n")

        if stage == 'test':
            # For test stage, only create test dataset
            self.test_dataset = RLKernelIterableDataset(test_events, event_processor=self.event_processor)
            print(f"Test dataset: {len(test_events)} events (separate test set)")
        elif stage == 'fit' or stage is None:
            # For fit stage, create train, val, and test datasets
            # CRITICAL: Test dataset is created but should NEVER be used during training
            self.train_dataset = RLKernelIterableDataset(train_events, event_processor=self.event_processor)
            self.val_dataset = RLKernelIterableDataset(val_events, event_processor=self.event_processor)
            self.test_dataset = RLKernelIterableDataset(test_events, event_processor=self.event_processor)
            print(f"Created datasets: Train={len(train_events)}, Val={len(val_events)}, Test={len(test_events)}")
            print(f"⚠️  WARNING: Test dataset is created but should NEVER be used during training or validation!")

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