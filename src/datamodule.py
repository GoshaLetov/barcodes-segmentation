import pandas as pd

from typing import Optional
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.augmentations import get_transforms
from src.config import DataConfig
from src.constants import ANNOTATIONS_PATH
from src.dataset import BarCodeDataset


def read_annotations(path: str) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=path, sep='\t')


def train_valid_test_split(annotations: pd.DataFrame, train_fraction: float) -> pd.DataFrame:
    train_idx = annotations.sample(frac=train_fraction, random_state=42, replace=False).index
    valid_idx = annotations.drop(index=train_idx).sample(frac=0.5, random_state=42, replace=False).index
    test_idx = annotations.drop(index=train_idx).drop(index=valid_idx).index

    annotations.loc[train_idx, 'part'] = 'train'
    annotations.loc[valid_idx, 'part'] = 'valid'
    annotations.loc[test_idx, 'part'] = 'test'

    print(annotations.part.value_counts(normalize=True))

    return annotations


class BarCodeDataModule(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config

        self.train_dataset: Optional[BarCodeDataset] = None
        self.valid_dataset: Optional[BarCodeDataset] = None
        self.test_dataset: Optional[BarCodeDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        annotations = train_valid_test_split(
            annotations=read_annotations(path=ANNOTATIONS_PATH),
            train_fraction=self._config.train_fraction,
        )

        self.train_dataset = BarCodeDataset(
            annotations=annotations[annotations.part == 'train'],
            transforms=get_transforms(config=self._config),
        )
        self.valid_dataset = BarCodeDataset(
            annotations=annotations[annotations.part == 'valid'],
            transforms=get_transforms(config=self._config, augmentations=False),
        )
        self.test_dataset = BarCodeDataset(
            annotations=annotations[annotations.part == 'test'],
            transforms=get_transforms(config=self._config, augmentations=False),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.train_batch_size,
            num_workers=self._config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.valid_batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.valid_batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
