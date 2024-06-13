import cv2
from time import perf_counter
from typing import Tuple, List
from dataclasses import dataclass, field
import numpy as np


@dataclass(init=True)
class ImageTransformationsParams:
    rotation_angles: Tuple[int] | List[int] = field(default=(0, 30, 45, 60, 90, 135, 180, 270), init=True)
    scaling: Tuple[float] | List[float] = field(default=(1, 0.75, 0.5), init=True)
    shifting: Tuple[bool] | List[bool] = field(default=(0, 100, -100), init=True)

    def get_values(self):
        return [self.rotation_angles, self.scaling, self.shifting]


DEFAULT_TRANSFORMATION_PARAMS = ImageTransformationsParams()


def make_test_images(training_images,
                     transformation_params: ImageTransformationsParams = DEFAULT_TRANSFORMATION_PARAMS,
                     with_labels: bool = False,
                     plus_one: bool = False):
    from itertools import product
    test_images = []
    true_labels = []

    for angle, scale, shift in product(*transformation_params.get_values()):
        for i, image in enumerate(training_images):
            height, width = image.shape[:2]
            center_point = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center_point, angle, scale)
            img_translation = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            shift_matrix = np.array([[1, 0, shift], [0, 1, shift]], dtype=np.float32)
            img_translation = cv2.warpAffine(img_translation, shift_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            test_images.append(img_translation)

            true_labels.append(i + 1 if plus_one else i)

    if with_labels:
        return test_images, true_labels
    return test_images


def timer_function(function):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        end = perf_counter()
        print(f"Time: {end - start:.2f}s")
        return result

    return wrapper


def get_descriptors_and_keypoints(images, n_features=500, detector=None):
    ORB = cv2.ORB_create(nfeatures=n_features) if detector is None else detector
    descriptors_per_image = []
    keypoints_per_image = []

    for i, img in enumerate(images):
        keypoints, descriptors = ORB.detectAndCompute(img, None)
        descriptors_per_image.append(descriptors)
        keypoints_per_image.append(keypoints)

    return descriptors_per_image, keypoints_per_image
