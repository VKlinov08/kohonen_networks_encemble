import numpy as np
import cv2
from .ensemble import *
from .utils.utils import *
from .descriptors_filtration import database_descriptors_reduction


def get_labels_per_image(n_classes, n_descriptors):
    labels_per_sample = np.empty((n_classes, n_classes * n_descriptors),
                                 dtype=np.int32)
    n_classes_range = np.arange(n_classes)
    for sample_id in n_classes_range:
        current_labels = n_classes_range == sample_id
        labels_per_sample[sample_id, :] = np.repeat(current_labels, n_descriptors)
    return labels_per_sample


def check_descriptors(descriptors):
    lengths = list(map(len, descriptors))
    min_len = min(lengths)
    if not all(list(map(lambda length: length == min_len, lengths))):
        descriptors = [sample[:min_len] for sample in descriptors]
    return descriptors


class ClassifierInitParams:
    def __init__(self, n_neurons: int,
                 epochs: int,
                 alpha0: float,
                 n_threads: int = 5,
                 distance: str = 'cityblock',
                 descriptors_func=cv2.ORB_create,
                 normalized: bool = False):
        self.n_neurons = n_neurons
        self.epochs = epochs
        self.alpha0 = alpha0
        self.n_threads = n_threads
        self.distance = distance
        self.descriptors_func = descriptors_func
        self.normalized = normalized
        pass


class ClassifierTrainParams:
    def __init__(self, n_neurons: int = None,
                 epochs: int = None,
                 alpha0: float = None,
                 distance: str = None,
                 descriptors_func=None,
                 n_threads: int = None,
                 class_names=None,
                 reduction_number: int = None,
                 detector_params=1000,
                 eps: float = None,
                 parallel: bool = True,
                 decision_bound: float = 0.5):
        self.n_neurons: int = n_neurons
        self.epochs: int = epochs
        self.alpha0: float = alpha0
        self.distance: str = distance
        self.descriptors_func = descriptors_func
        self.n_threads: int = n_threads
        self.class_names = class_names
        self.reduction_number: int = reduction_number
        self.detector_params = detector_params
        self.eps: float = eps
        self.parallel: bool = parallel
        self.decision_bound: float = decision_bound
        pass


class EnsembleImageClassifier:
    def __init__(self, init_params: ClassifierInitParams):
        self.class_names = None
        self.n_neurons = init_params.n_neurons
        self.epochs = init_params.epochs
        self.alpha0 = init_params.alpha0
        self.distance = init_params.distance
        self.descriptors_func = init_params.descriptors_func
        self.n_threads = init_params.n_threads
        self.normalized = init_params.normalized
        self.ensemble = None

    @staticmethod
    def get_descriptors_(image, detector):
        return detector.detectAndCompute(image, None)

    @staticmethod
    def get_descriptors_per_image_(images, detector_params, descriptors_func):
        descriptors_creator = descriptors_func(detector_params)

        image_descriptors = []
        for i, img in enumerate(images):
            _, descriptors = descriptors_creator.detectAndCompute(img, None)
            image_descriptors.append(descriptors)
        return image_descriptors

    @timer_function
    def train(self, train_images, train_params: ClassifierTrainParams):
        n_neurons = self.n_neurons if train_params.n_neurons is None else train_params.n_neurons
        epochs = self.epochs if train_params.epochs is None else train_params.epochs
        alpha0 = self.alpha0 if train_params.alpha0 is None else train_params.alpha0
        distance = self.distance if train_params.distance is None else train_params.distance
        descriptors_func = self.descriptors_func if train_params.descriptors_func is None else train_params.descriptors_func
        n_threads = self.n_threads if train_params.n_threads is None else train_params.n_threads
        self.class_names = train_params.class_names

        if len(train_images) < 1: return

        train_descriptors = EnsembleImageClassifier.get_descriptors_per_image_(train_images,
                                                                               train_params.detector_params,
                                                                               descriptors_func)
        train_descriptors = check_descriptors(train_descriptors)
        train_bits = np.asarray([np.unpackbits(sample, axis=1) for sample in train_descriptors])

        if train_params.reduction_number is not None:
            train_bits = database_descriptors_reduction(train_bits, train_params.reduction_number)

        n_classes = len(train_descriptors)
        ndim = train_bits.shape[2]
        if train_params.parallel:
            self.ensemble = ParallelSelfLearningEnsemble(n_classes, n_neurons, ndim, distance, n_threads,
                                                         self.normalized)
        else:
            self.ensemble = SelfLearningEnsemble(n_classes, n_neurons, ndim, distance, self.normalized)
        print("Training started!")
        self.ensemble.train(train_bits, epochs, alpha0, train_params.eps)
        print("Training finished!")
        pass

    def classify_descriptors(self, descriptors: np.ndarray,
                             decision_bound: float,
                             with_stats: bool = False):
        bits = np.unpackbits(descriptors, axis=1)
        return self.ensemble.classify(bits, decision_bound, with_stats)

    def classify_image(self, image: np.ndarray, decision_bound: float, detector_params=1000, by_name=False,
                       with_stats=False, with_error: bool = False):
        _, descriptors = EnsembleImageClassifier.get_descriptors_(image, self.descriptors_func(detector_params))
        result = self.classify_descriptors(descriptors, decision_bound, with_stats)

        to_return = []
        if with_error:
            bits = np.unpackbits(descriptors, axis=1)
            errors = [network.error(bits, normalize_input=self.normalized) for network in self.ensemble.ensemble]
            errors_round = np.round(errors, 3)
            to_return.insert(0, errors_round)
        if with_stats:
            to_return.insert(0, result[1])
        if not by_name or self.class_names is None:
            to_return.insert(0, result[0])
        if by_name and self.class_names is not None:
            name = self.class_names[result[0]] if result[0] != -1 else 'Mismatch'
            to_return.insert(0, name)

        return to_return

    def set_class_names(self, class_names):
        if self.ensemble is not None and len(self.ensemble.ensemble) != len(class_names):
            raise ValueError("Length of a list with class names doesn't match with a number of learned classes!")
        self.class_names = class_names
        pass
