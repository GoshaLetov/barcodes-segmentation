from typing import Tuple

import torch
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp

from src.config import Config
from src.losses import get_loss
from src.metrics import get_metrics
from src.utilities import load_object


class BarCodeLightningModule(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self.model = smp.Unet(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            classes=1,
            aux_params={'pooling': 'avg', 'dropout': 0.2, 'classes': 1},
        )

        self._loss = get_loss(loss_cfg=self._config.loss)

        seg_metrics = get_metrics()
        self._val_seg_metrics = seg_metrics.clone(prefix='valid.')
        self._test_seg_metrics = seg_metrics.clone(prefix='test.')

        self.save_hyperparameters(self._config.model_dump())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer.name)(
            self.model.parameters(), lr=self._config.optimizer.lr, **self._config.optimizer.kwargs,
        )
        scheduler = load_object(self._config.scheduler.name)(optimizer, **self._config.scheduler.kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid.IoU',
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, masks = batch
        pred_masks_logits, _ = self.forward(images)
        return self._calculate_loss(masks=masks, pred_masks_logits=pred_masks_logits, prefix='train.')

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        pred_masks_logits, _ = self.forward(images)
        self._calculate_loss(masks=masks, pred_masks_logits=pred_masks_logits, prefix='valid.')
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._val_seg_metrics(pred_masks, masks)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        pred_masks_logits, _ = self.forward(images)
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._test_seg_metrics(pred_masks, masks)

    def on_validation_epoch_start(self) -> None:
        self._val_seg_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._val_seg_metrics.compute(), on_epoch=True, on_step=False)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_seg_metrics.compute(), on_epoch=True, on_step=False)

    def _calculate_loss(
        self,
        masks: torch.Tensor,
        pred_masks_logits: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        loss = self._loss.loss(pred_masks_logits, masks) * self._loss.weight
        self.log(name=f'{prefix}.{self._loss.name}', value=loss.item())
        return loss
