from torch import set_float32_matmul_precision
from clearml import Task
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.fabric import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from src.config import Config
from src.constants import CONFIGS_PATH, EXPERIMENTS_PATH, PL_LOGS_PATH
from src.datamodule import BarCodeDataModule
from src.lightningmodule import BarCodeLightningModule


def train(config: Config):
    set_float32_matmul_precision('high')

    task = Task.init(
        project_name='BarCodesSegmentation',
        task_name=config.experiment_name,
        auto_connect_frameworks=True,
    )
    task.connect(config.model_dump())

    datamodule = BarCodeDataModule(config=config.data)
    model = BarCodeLightningModule(config=config)
    experiment_save_path = EXPERIMENTS_PATH / config.experiment_name
    experiment_save_path.mkdir(exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.model.monitor_metric,
        mode=config.model.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{config.model.monitor_metric}_{{{config.model.monitor_metric}:.3f}}',
    )
    trainer = Trainer(
        max_epochs=config.model.n_epochs,
        accelerator=config.model.accelerator,
        devices=[config.model.device],
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir=PL_LOGS_PATH, name=config.experiment_name),
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.model.monitor_metric, patience=10, mode=config.model.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        deterministic=True,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == '__main__':
    seed_everything(seed=42, workers=True)
    config = Config.from_yaml(CONFIGS_PATH / 'config.yaml')
    train(config)
