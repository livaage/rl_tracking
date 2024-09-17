import torch
from torch.utils.data import Dataset, DataLoader
from rl_tracking.preprocessing.hit_candidates import EventProcessor
import os
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler
from typing import Any
from torch.utils.data import IterableDataset

class RLKernelIterableDataset(IterableDataset):
    def __init__(self, in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop=None,
        event_processor: EventProcessor = None):
        """
        Args:
            data (list or array): List or array of data samples.
            labels (list or array): List or array of labels corresponding to the data samples.
        """
        super().__init__()
        self.in_dir = Path(in_dir)
        self._processed_paths = self._get_paths(
            in_dir, start=start, stop=stop
        )
        self.event_processor = event_processor

    @staticmethod
    def _get_paths(
            in_dir: os.PathLike | list[str] | list[os.PathLike],
            start:int=0,
            stop:int=None):

        file_names = Path(in_dir).iterdir()
        event_numbers = list(set([str(x).split("-")[0] for x in file_names]))
        considered_files = event_numbers[start:stop]
        return considered_files

    def __len__(self):
        return len(self._processed_paths)

    def __getitem__(self, idx):
        sample, label = self.event_processor.process(self.in_dir/self._processed_paths[idx])
        return sample, label


class TrackingDataModule(LightningDataModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        identifier: str,  # noqa: ARG002
        train: dict | None = None,
        val: dict | None = None,
        test: dict | None = None,
        cpus: int = 1,
        event_processor: EventProcessor = None,
        ):
        """This subclass of `LightningDataModule` configures all data for the
        ML pipeline.

        Args:
            identifier: Identifier of the dataset (e.g., `graph_v5`)
            train: Config dictionary for training data (see below)
            val: Config dictionary for validation data (see below)
            test: Config dictionary for test data (see below)
            cpus: Number of CPUs to use for loading data.


        The following keys are available for each config dictionary:

        - `dirs`: List of dirs to load from (required)
        - `start=0`: Index of first file to load
        - `stop=None`: Index of last file to load
        - `sector=None`: Sector to load from (if None, load all sectors)
        - `batch_size=1`: Batch size

        Training has the following additional keys:

        - `sample_size=None`: Number of samples to load for each epoch
            (if None, load all samples)
        """
        self.save_hyperparameters()
        super().__init__()
        self._configs = {
            "train": self._fix_datatypes(train),
            "val": self._fix_datatypes(val),
            "test": self._fix_datatypes(test),
        }
        self._datasets = {}
        self._cpus = cpus
        self._event_processor = event_processor

    @property
    def datasets(self) -> dict[str, RLKernelDataset]:
        return self._datasets

    @staticmethod
    def _fix_datatypes(dct: dict[str, Any] | None) -> dict[str, Any] | None:
        """Fix datatypes of config dictionary.
        This is necessary because when configuring values from the command line,
        all values might be strings.
        """
        if dct is None:
            return {}
        for key in ["start", "stop", "batch_size", "sample_size"]:
            if key in dct:
                dct[key] = int(dct[key])
        return dct

    def _get_dataset(self, key: str) -> RLKernelDataset:
        config = self._configs[key]

        in_dir = config["dirs"]
        # config = self._configs[key]
        if not config:
            msg = f"DataLoaderConfig for key {key} is None."
            raise ValueError(msg)

        return RLKernelDataset(
            in_dir=in_dir,
            start=config.get("start", 0),
            stop=config.get("stop", None),
            event_processor = self._event_processor
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._datasets["train"] = self._get_dataset("train")
            self.setup("validate")
        elif stage == "validate":
            self._datasets["val"] = self._get_dataset("val")
        elif stage == "test":
            self._datasets["test"] = self._get_dataset("test")
        else:
            _ = f"Unknown stage '{stage}'"
            raise ValueError(_)

    def _get_dataloader(self, key: str) -> DataLoader:
        sampler = None
        dataset = self._datasets[key]
        n_samples = len(dataset)
        if key == "train" and len(self._datasets[key]):
            if "max_sample_size" in self._configs[key]:
                msg = "max_sample_size has been replaced by sample_size"
                raise ValueError(msg)
            n_samples = self._configs[key].get("sample_size", len(dataset))
            sampler = RandomSampler(
                self._datasets[key],
                replacement=n_samples > len(dataset),
                num_samples=n_samples,
            )
        return DataLoader(
            dataset,
            batch_size=self._configs[key].get("batch_size", 1),
            num_workers=max(1, min(n_samples, self._cpus)),
            sampler=sampler,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")
