import torch

from typing import List, Dict, Optional

from torchmetrics import Metric, MetricCollection
from segmentation_models_pytorch.metrics import get_stats


class IoUMultiLabel(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, threshold: Optional[float] = 0.5):
        super().__init__()
        self._threshold = threshold
        self._init_logical_values()

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        tp, fp, fn, _ = get_stats(
            output=output.long(),
            target=target.long(),
            mode='binary',
            threshold=self._threshold,
        )
        self.tp += tp.sum()
        self.fp += fp.sum()
        self.fn += fn.sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        result = {'IoU': self.tp / (self.tp + self.fp + self.fn)}
        return result

    def _init_logical_values(self):
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')


def get_metrics(threshold: Optional[float] = 0.5) -> MetricCollection:
    return MetricCollection({'IoU': IoUMultiLabel(threshold=threshold)})
