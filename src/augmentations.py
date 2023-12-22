import albumentations

from albumentations.pytorch import ToTensorV2
from src.config import AugmentationConfig
from src.config import DataConfig
from src.utilities import load_object
from typing import List
from typing import Union


TRANSFORM_TYPE = Union[albumentations.BasicTransform, albumentations.BaseCompose]


def get_augmentations(augmentations: List[AugmentationConfig]) -> List:
    return [
        load_object(object_path=f'albumentations.{augmentation.name}')(**augmentation.kwargs)
        for augmentation in augmentations
    ]


def get_transforms(
    config: DataConfig,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:

    transforms = []

    if preprocessing:
        transforms.append(albumentations.Resize(height=config.height, width=config.width))

    if augmentations:
        transforms.extend(get_augmentations(augmentations=config.augmentations))

    if postprocessing:
        transforms.extend([albumentations.Normalize(), ToTensorV2()])

    return albumentations.Compose(transforms)
