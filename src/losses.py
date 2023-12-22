from dataclasses import dataclass

from torch import nn

from src.config import LossConfig
from src.utilities import load_object


@dataclass
class Loss:
    name: str
    weight: float
    loss: nn.Module


def get_loss(loss_cfg: LossConfig) -> Loss:
    loss = load_object(loss_cfg.name)(**loss_cfg.kwargs)
    name = loss_cfg.name.split('.')[-1]
    return Loss(name=name, weight=loss_cfg.weight, loss=loss)
