from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from torch.utils.data import Dataset

from src.augmentations import TRANSFORM_TYPE
from src.constants import DATA_PATH
from src.io import read_image_rgb


class BarCodelDatasetRow(BaseModel):
    filename: str
    code: int
    x_from: int
    y_from: int
    width: int
    height: int

    @property
    def image(self) -> np.ndarray:
        return read_image_rgb(image_path=str(DATA_PATH / self.filename))

    @property
    def mask(self) -> np.ndarray:
        mask = np.zeros(shape=self.image.shape[:2], dtype=np.float32)
        mask[self.y_from:self.y_from + self.height, self.x_from:self.x_from + self.width] = 1
        mask = mask[:, :, np.newaxis]
        return mask


class BarCodeDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, transforms: Optional[TRANSFORM_TYPE] = None):
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        row = BarCodelDatasetRow(**self.annotations.iloc[idx])
        image, mask = row.image, row.mask
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        return image, np.transpose(mask, axes=[2, 0, 1])
