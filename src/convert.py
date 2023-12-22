import argparse
import torch

from src.lightningmodule import BarCodeLightningModule
from src.config import Config
from src.constants import CONFIGS_PATH


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='model checkpoint path')
parser.add_argument('onnx', type=str, help='onnx model path')


def torch_to_onnx(config: Config, checkpoint: str, onnx: str) -> None:
    model = BarCodeLightningModule.load_from_checkpoint(checkpoint, config=config)
    model.to_onnx(
        file_path=onnx,
        input_sample=torch.randn(1, 3, config.data.height, config.data.width),
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    args = parser.parse_args()
    config = Config.from_yaml(CONFIGS_PATH / 'config.yaml')
    torch_to_onnx(config=config, checkpoint=args.checkpoint, onnx=args.onnx)
