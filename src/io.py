import cv2
import numpy as np


def read_image_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Image does not exist: {image_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
