from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import List


class ModelConfig(BaseModel):
    monitor_metric: str
    monitor_mode: str
    n_epochs: int
    accelerator: str
    device: int


class AugmentationConfig(BaseModel):
    name: str
    kwargs: dict


class DataConfig(BaseModel):
    num_workers: int
    train_fraction: float
    train_batch_size: int
    valid_batch_size: int
    width: int
    height: int
    augmentations: List[AugmentationConfig]


class LossConfig(BaseModel):
    name: str
    weight: float
    kwargs: dict


class OptimizerConfig(BaseModel):
    name: str
    lr: float
    kwargs: dict


class SchedulerConfig(BaseModel):
    name: str
    kwargs: dict


class Config(BaseModel):
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
