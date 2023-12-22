import argparse
import numpy as np

from cv2 import resize
from onnxruntime import InferenceSession
from src.config import Config
from src.constants import CONFIGS_PATH
from src.augmentations import get_transforms, TRANSFORM_TYPE
from src.io import read_image_rgb

from skimage.measure import label, regionprops


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class BBoxExtractor:
    def __init__(self, model: InferenceSession, transforms: TRANSFORM_TYPE):
        self.model = model
        self.transforms = transforms

    def extract(self, image: np.ndarray) -> list[dict[str, int]]:
        tensor = self.transforms(image=image).get('image')

        mask = self.model.run(output_names=None, input_feed={'input': [tensor]})
        mask = resize(src=mask[0][0, 0], dsize=[image.shape[1], image.shape[0]])
        mask = sigmoid(mask) > 0.5
        mask = mask.astype(int)

        return [{
            'x_min': prop.bbox[0],
            'x_max': prop.bbox[2],
            'y_min': prop.bbox[1],
            'y_max': prop.bbox[3],
        } for prop in regionprops(label(mask))]


parser = argparse.ArgumentParser()
parser.add_argument('onnx', type=str, help='onnx model path')
parser.add_argument('image', type=str, help='image path')


if __name__ == '__main__':
    args = parser.parse_args()
    config = Config.from_yaml(CONFIGS_PATH / 'config.yaml')
    model = InferenceSession(path_or_bytes=args.onnx, providers=['CPUExecutionProvider'])
    transforms = get_transforms(config=config.data, augmentations=False, postprocessing=True)
    image = read_image_rgb(image_path=args.image)
    extractor = BBoxExtractor(model=model, transforms=transforms)
    print(extractor.extract(image=image))
