import pandas as pd
import torch
from torch.utils.data import IterableDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Iterable, Sequence, List, Set, Dict
from collections import Counter
import numpy as np
from rl_tracking.utils.logger import get_logger

logger = get_logger()
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
        logger.info("Processing event %s", event_number)
        try:
            sample = self.event_processor.process(event_number)
        except Exception as exc:
            logger.error("Failed to process event %s: %s", event_number, exc)
            raise
        yield sample
        self.sample_count += 1

    def __iter__(self):
        self.sample_count = 0
        # Iterate through all events in the dataset
        for file in self.event_numbers:
            yield from self._read_file(file)
            # Removed epoch_size limit - use all events

class TrackingDataModule(LightningDataModule):
    def __init__(
        self,
        file_paths: str | Path | Sequence[str | Path],
        batch_size: int,
        event_processor=None,
        num_workers: int = 0,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.event_processor = event_processor
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.data_directories = self._normalize_directories(file_paths)
        self.event_numbers = self._discover_event_numbers(self.data_directories)

        if not self.event_numbers:
            searched = ", ".join(str(path) for path in self.data_directories)
            raise FileNotFoundError(
                f"No TrackML events found in provided directories: {searched}"
            )

        print(f"Discovered {len(self.event_numbers)} TrackML events across {len(self.data_directories)} directories.")

        # Split tracking
        self.train_events: List[str] = []
        self.val_events: List[str] = []
        self.test_events: List[str] = []
        self.split_summary: Dict[str, Dict[str, int]] = {
            'train': {},
            'val': {},
            'test': {},
        }

    def setup(self, stage=None):
        # CRITICAL: Use consistent random_state=42 for all splits to ensure reproducibility
        # and that train/val/test are always separate
        split_random_state = self.random_seed

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
        shuffled_events = self.event_numbers[:]
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(shuffled_events)

        val_test_size = int(len(shuffled_events) * (self.val_split + self.test_split))
        train_events = shuffled_events[val_test_size:]
        temp_events = shuffled_events[:val_test_size]

        val_size = int(len(temp_events) * (self.val_split / (self.val_split + self.test_split)))
        val_events = temp_events[:val_size]
        test_events = temp_events[val_size:]
        
        # Ensure list types
        train_events = list(train_events)
        temp_events = list(temp_events)
        val_events = list(val_events)
        test_events = list(test_events)

        # Store split event identifiers
        self.train_events = train_events
        self.val_events = val_events
        self.test_events = test_events

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
        total_events = len(shuffled_events)
        print(f"Train events: {len(train_events)} ({len(train_events)/total_events:.2%})")
        print(f"Val events: {len(val_events)} ({len(val_events)/total_events:.2%})")
        print(f"Test events: {len(test_events)} ({len(test_events)/total_events:.2%})")
        print(f"Total events: {total_events}")
        print(f"Overlap checks: Train-Val={len(train_set & val_set)}, Train-Test={len(train_set & test_set)}, Val-Test={len(val_set & test_set)}")
        print(f"✓ All splits are separate (no overlap)")
        print(f"{'='*60}\n")

        self._update_split_summary('train', train_events)
        self._update_split_summary('val', val_events)
        self._update_split_summary('test', test_events)

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

    def get_split_summary(self) -> Dict[str, Dict[str, int]]:
        """Return shallow copy of event counts per part for each split."""
        return {name: counts.copy() for name, counts in self.split_summary.items()}

    def _update_split_summary(self, split_name: str, events: List[str]) -> None:
        part_counts = Counter()
        for event in events:
            part_dir = Path(event).parent.name
            part_counts[part_dir] += 1
        self.split_summary[split_name] = dict(sorted(part_counts.items()))

    @staticmethod
    def _normalize_directories(
        file_paths: str | Path | Sequence[str | Path],
    ) -> List[Path]:
        if isinstance(file_paths, (str, Path)):
            candidates = [file_paths]
        else:
            candidates = list(file_paths)

        resolved_paths: List[Path] = []
        for candidate in candidates:
            path_obj = Path(candidate).expanduser().resolve()
            if not path_obj.exists():
                raise FileNotFoundError(f"Data directory does not exist: {path_obj}")
            resolved_paths.append(path_obj)

        return sorted(resolved_paths)

    @staticmethod
    def _discover_event_numbers(data_directories: Iterable[Path]) -> List[str]:
        prefixes: Set[str] = set()

        hit_file_patterns = [
            "event*-hits.csv",
            "event*-hits.csv.gz",
            "event*-hits.csv.zip",
        ]

        for directory in data_directories:
            if directory.is_file():
                candidate_files = [directory]
            else:
                candidate_files = []
                # Prefer shallow search first to avoid descending into nested archives unnecessarily.
                for pattern in hit_file_patterns:
                    candidate_files.extend(directory.glob(pattern))
                if not candidate_files:
                    for pattern in hit_file_patterns:
                        candidate_files.extend(directory.rglob(pattern))

            for hits_file in candidate_files:
                name = hits_file.name
                if name.endswith(".gz"):
                    name = name[:-3]
                if name.endswith(".zip"):
                    name = name[:-4]
                if not name.endswith("-hits.csv"):
                    continue
                prefix_name = name[: -len("-hits.csv")]
                prefix_path = hits_file.parent / prefix_name
                prefixes.add(str(prefix_path))

        return sorted(prefixes)