from scipy.spatial.distance import cdist
from utils.utils import get_descriptors_and_keypoints
import numpy as np
import cv2
import matplotlib.pyplot as plt

def skip_diag_strided(matrix: np.ndarray) -> np.ndarray:
    """ Function to remove diagonal elements"""
    rows = matrix.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = matrix.strides
    return strided(matrix.ravel()[1:], shape=(rows - 1, rows), strides=(s0 + s1, s1)).reshape(rows, -1)


# def get_descriptors_and_keypoints(images, n_features=500, detector=None):
#     ORB = cv2.ORB_create(nfeatures=n_features) if detector is None else detector
#     descriptors_per_image = []
#     keypoints_per_image = []
#
#     for i, img in enumerate(images):
#         keypoints, descriptors = ORB.detectAndCompute(img, None)
#         descriptors_per_image.append(descriptors)
#         keypoints_per_image.append(keypoints)
#     return descriptors_per_image, keypoints_per_image


class DescriptorsReduction:
    def __init__(self, filter_count: int, distance="hamming"):
        self.distance = distance
        self.filter_count = filter_count
        self._first_rating = None
        self._second_rating = None
        pass

    def _get_reduction_difference(self, descriptors1: np.ndarray,
                                  descriptors2: np.ndarray) -> np.ndarray:
        closest_to_others = np.amin(cdist(descriptors1, descriptors2, metric=self.distance),
                                    axis=1)

        distance_w_diag = cdist(descriptors1, descriptors1, metric=self.distance)
        distance_no_diag = skip_diag_strided(distance_w_diag)
        closest_to_ours = np.amin(distance_no_diag, axis=1)
        return closest_to_others - closest_to_ours

    def fit_pair(self, descriptors1, descriptors2):
        self._first_rating = self._get_reduction_difference(descriptors1, descriptors2)
        self._second_rating = self._get_reduction_difference(descriptors2, descriptors1)

    def get_remaining_indices(self, reduction_rating: np.ndarray) -> np.ndarray:
        return np.argsort(-reduction_rating)[:self.filter_count]

    def transform_pair(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> tuple:
        first_remaining_indices = self.get_remaining_indices(self._first_rating)
        second_remaining_indices = self.get_remaining_indices(self._second_rating)
        return descriptors1[first_remaining_indices], descriptors2[second_remaining_indices]

    def fit(self, descriptors1: np.ndarray, descriptors_all: np.ndarray):
        self._first_rating = self._get_reduction_difference(descriptors1, descriptors_all)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        remaining_indices = self.get_remaining_indices(self._first_rating)
        return descriptors[remaining_indices]


def database_descriptors_reduction(database_samples_bits: np.ndarray, reduction_number: int) -> np.ndarray:
    selected_descriptors_per_sample = []
    all_indices = np.arange(len(database_samples_bits))

    for sample_index in all_indices:
        other_indices = all_indices[all_indices != sample_index]
        descs1 = database_samples_bits[sample_index]
        descs_all = np.vstack(database_samples_bits[other_indices])

        dr = DescriptorsReduction(reduction_number)
        dr.fit(descs1, descs_all)
        descs2_1 = dr.transform(descs1)
        selected_descriptors_per_sample.append(descs2_1)

    return np.asarray(selected_descriptors_per_sample)


"""### Descriptors reduction impact demonstration"""


def extract_by_index(sequence, index: int, with_stacking=False):
    extracted = None
    if isinstance(sequence, list):
        extracted = sequence.copy().pop(index)
    if isinstance(sequence, np.ndarray):
        extracted = np.delete(sequence, index, 0)
    if with_stacking:
        return np.vstack(extracted)
    return extracted


def show_reduction_impact(images, sample_index, n_filtered, n_features_normal, n_features_big):
    many_descriptors, many_kp = get_descriptors_and_keypoints(images, n_features_big)
    many_bits = np.asarray([np.unpackbits(descriptors, axis=1) for descriptors in many_descriptors])

    normal_descriptors, normal_kp = get_descriptors_and_keypoints(images, n_features_normal)
    # normal_bits = np.asarray([np.unpackbits(descriptors, axis=1) for descriptors in normal_descriptors])

    dr = DescriptorsReduction(n_filtered)
    other_samples_bits = extract_by_index(many_bits, sample_index, True)
    dr.fit(many_bits[sample_index], other_samples_bits)

    current_kp = many_kp[sample_index]
    selected_indices = dr.get_remaining_indices(dr._first_rating)
    selected_kp = np.asarray(current_kp)[selected_indices]
    normal_kp = normal_kp[sample_index]

    img = images[sample_index]
    img_with_reduced_kp = cv2.drawKeypoints(img, selected_kp, None, color=(0, 255, 0), flags=0)
    img_many_kp = cv2.drawKeypoints(img, current_kp, None, color=(0, 255, 0), flags=0)
    img_normal_kp = cv2.drawKeypoints(img, normal_kp, None, color=(0, 255, 0), flags=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(img_normal_kp)
    axes[1].imshow(img_many_kp)
    axes[2].imshow(img_with_reduced_kp)

    titles = [f'Звичайні {n_features_normal} дескрипторів',
              f'Звичайні {n_features_big} дескрипторів',
              f'Відфільтровані {n_filtered} дескриптоірв']
    for title, ax in zip(titles, axes.flatten()):
        ax.set_axis_off()
        ax.set_title(title)
    plt.plot()


# for i in range(len(train_images)):
#     show_reduction_impact(train_images, i, 500, 500, 2500)
