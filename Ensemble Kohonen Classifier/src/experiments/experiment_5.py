from .common import ensemble_classification_test, TestParams, test_parameters_generator
import numpy as np
import sys
sys.path.append('..')
from models.learning_interface import *


DEFAULT_TEST_PARAMS = TestParams(500, [10], [0.25], [True], [500], [0.25], [None])
DEFAULT_N_NEURONS_LIST = (1, 3, 5, 10, 25, 50)


def get_time_and_accuracy_lists(training_images,
                                n_neurons_list=DEFAULT_N_NEURONS_LIST,
                                test_params: TestParams = DEFAULT_TEST_PARAMS,
                                transformation_params: ImageTransformationsParams = DEFAULT_TRANSFORMATION_PARAMS):
    generator = test_parameters_generator(n_neurons_list, test_params)
    generated_test_images, true_labels = make_test_images(training_images, transformation_params, with_labels=True,
                                                          plus_one=True)
    accuracy_list = []
    lt_list, tt_list = [], []

    for n_neurons, epochs, alpha0, normalized, detector_params, decision_bound, reduction_number in generator:
        if reduction_number is not None and reduction_number > detector_params:
            continue
        print("\n--------------------------------")
        print(n_neurons, epochs, alpha0, reduction_number, normalized, detector_params, decision_bound)
        init_params = ClassifierInitParams(n_neurons, epochs, alpha0, normalized=normalized)
        train_params = ClassifierTrainParams(decision_bound=decision_bound)
        classifier, acc, learning_time, test_time = ensemble_classification_test(training_images,
                                                                                 generated_test_images,
                                                                                 true_labels,
                                                                                 init_params,
                                                                                 train_params, plus_one=True)

        accuracy_list.append(acc)
        lt_list.append(learning_time)
        tt_list.append(test_time)
    return lt_list, tt_list, accuracy_list


def check_test_params(test_params: TestParams):
    if len(test_params.alpha0_list) + len(test_params.epochs_list) + len(test_params.normalized_list) + len(
            test_params.decision_bound_list) + \
            len(test_params.detector_params_list) + len(test_params.reduction_number_list) != 6:
        raise ValueError(
            "Lengths of all iterable parameters of a 'test_params' argument must be equal to 1 for current experiment!")


def run(training_images,
        test_params: TestParams=DEFAULT_TEST_PARAMS,
        n_neurons_list=DEFAULT_N_NEURONS_LIST,
        transformation_params: ImageTransformationsParams=DEFAULT_TRANSFORMATION_PARAMS):

    check_test_params(test_params)
    lt_list, tt_list, accuracy_list = get_time_and_accuracy_lists(training_images, n_neurons_list, test_params,
                                                                  transformation_params)
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(n_neurons_list, lt_list, marker='o', color='red')
    axes[0].set_xlim(-0.05, n_neurons_list[-1] + 1)
    axes[0].set_xlabel("Кількість нейронів на еталон")
    axes[0].set_ylabel('Час на навчання, c')
    axes[0].xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))

    axes[1].plot(n_neurons_list, tt_list, marker='o', color='blue')
    axes[1].set_xlim(-0.05, n_neurons_list[-1] + 1)
    axes[1].set_xlabel("Кількість нейронів на еталон")
    axes[1].set_ylabel('Час на класифікацію тестової вибірки, c')
    axes[1].xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))
    plt.plot()

    fig, ax = plt.subplots()
    ax.plot(n_neurons_list, accuracy_list, marker='o')
    ax.set_xlim(-0.05, n_neurons_list[-1] + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Кількість нейронів на еталон")
    ax.set_ylabel('Точність тестової класифікації')
    ax.set_title(
        f"{test_params.n_features} дескрипторів, {test_params.epochs_list[0]} епох навчання, коефіцієнт навчання {test_params.alpha0_list[0]} ")
    ax.xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))
    plt.plot()
