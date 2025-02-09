from sklearn.metrics import accuracy_score
from time import perf_counter
from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
import sys
sys.path.append('..')
from models.learning_interface import *


def ensemble_classification_test(train_images, test_images, true_labels,
                                 init_params: ClassifierInitParams,
                                 train_params: ClassifierTrainParams, plus_one=False):
    """

    :param train_images: List[np.ndarray] - images for classifier learning
    :param test_images: List[np.ndarray] - images to classify
    :param true_labels: labels of test images
    :param init_params: parameters for classifier creation
    :param train_params: parameters for classifier learning
    :param plus_one: generated_test_images was made with `plus_one` option which means all labels
    starts from 1 instead of 0.
    :return: classifier - inctance of EnsembleImageClassifier,
             acc - AccuracyScore,
             training_time, testing_time - float number 
    """
    classifier = EnsembleImageClassifier(init_params)
    start_training = perf_counter()
    classifier.train(train_images, train_params)
    training_time = perf_counter() - start_training
    print(f"Learning Time: {training_time:.4f} s")

    # Класифікація еталонів
    classification_time_avg = 0.0
    class_names = train_params.class_names if train_params.class_names is not None else \
        [f"Image {i}" for i in range(1, len(train_images)+1, 1)]
    for sample, class_name in zip(train_images, class_names):
        classification_start = perf_counter()
        print(class_name, classifier.classify_image(sample,
                                                    train_params.decision_bound,
                                                    by_name=True, with_stats=True, with_error=True))
        sample_classification_time = perf_counter() - classification_start
        print(f"\t{sample_classification_time:.4f} s for reference image classification")
        classification_time_avg += sample_classification_time
    print("\n")
    print(f"\t{classification_time_avg / len(train_images):.4f} s - average time for reference image classification")

    # Класифікація тестових зображень
    pred_labels = []
    testing_start = perf_counter()
    for test, true_label in zip(test_images, true_labels):
        label, density = classifier.classify_image(test, train_params.decision_bound, with_stats=True)
        if plus_one:
            label = -1 * true_label if label == -1 or label + 1 != true_label else label + 1
        else:
            label = label if label == true_label else -1 * true_label
        pred_labels.append(label)

    testing_time = perf_counter() - testing_start
    print(f"Test Time: {testing_time} s. ~{testing_time / 360:.5f} s per image.")

    print("Accuracy")
    acc = accuracy_score(true_labels, pred_labels)
    print(f"{acc:.3f} / 1.")
    print(np.unique(pred_labels, return_counts=True))
    return classifier, acc, training_time, testing_time


@dataclass(init=True, frozen=True)
class TestParams:
    n_features: int = 500
    epochs_list: Tuple[int] | List[int] = field(default=(5, 10, 25, 50), init=True)
    alpha0_list: Tuple[float] | List[float] = field(default=(0.1, 0.25, 0.5, 0.75, 0.9), init=True)
    normalized_list: Tuple[bool] | List[bool] = field(default=tuple([True]), init=True)
    detector_params_list: Tuple[int] | List[int] = field(default=tuple([500]), init=True)
    decision_bound_list: Tuple[float] | List[float] = field(default=tuple([0.25]), init=True)
    reduction_number_list: Tuple[float] | List[float] | List[None] = field(default=tuple([None]), init=True)

    def get_values(self) -> tuple:
        return (self.epochs_list,
                self.alpha0_list,
                self.normalized_list,
                self.detector_params_list,
                self.decision_bound_list,
                self.reduction_number_list)


def get_test_params_generator(n_neurons_list: list, test_params: TestParams):
    from itertools import product

    generator = product(n_neurons_list,
                        *test_params.get_values())
    return generator
