import numpy as np
import sys

sys.path.append('..')
from .common import ensemble_classification_test, TestParams, test_parameters_generator
from models.learning_interface import *
from matplotlib import pyplot as plt
from matplotlib import ticker

"""
    Research of accuracy of the classifier based on changing of a number of centroids per a class.
"""

DEFAULT_TEST_PARAMS = TestParams(500, [5],
                                 [0.1, 0.25, 0.5, 0.75, 0.9],
                                 [True], [500], [0.25], [None])
DEFAULT_N_NEURONS_LIST = (1, 3, 5, 7, 10, 15, 20, 25)


def get_plot_dict(n_neurons_list, training_images, class_names,
                  test_params: TestParams = DEFAULT_TEST_PARAMS,
                  transformation_params: ImageTransformationsParams = DEFAULT_TRANSFORMATION_PARAMS):
    """
    Function returns a dictionary with data about accuracy of a classification process
    with different start learning rate and fixed rest of test parameters.
    :param n_neurons_list: List[int] - a list of neurons (descriptors centroids) per sample of image class.
    :param training_images: List[np.ndarray] - images to process.
    :param class_names: List[str] - list of names of training images.
    :param test_params: TestParams - test parameters for current experiment.
    :param transformation_params: ImageTransformationsParam - affine transformations parameters for test images.
    :return: dictionary with accuracy list for fixed number of epochs and different start learning rates.
    """
    if len(test_params.epochs_list) != 1:
        raise ValueError("List of epochs for current experiment must be fixed and equals 1!")

    generator = test_parameters_generator(n_neurons_list, test_params)
    generated_test_images, true_labels = make_test_images(training_images,
                                                          transformation_params,
                                                          with_labels=True,
                                                          plus_one=True)

    plot_dict = {"epochs": test_params.epochs_list[0], 'accuracy': {}}
    for n_neurons, epochs, alpha0, normalized, detector_params, decision_bound, reduction_number in generator:
        if reduction_number is not None and reduction_number > detector_params:
            raise ValueError("Number of descriptors after initial detection process must be bigger than after "
                             "reduction process."
                             "Choose different reduction_number or detector_params!")
        print("\n" + "-" * 100)
        print(n_neurons, epochs, alpha0, reduction_number, normalized, detector_params, decision_bound)
        init_params = ClassifierInitParams(n_neurons, epochs, alpha0, normalized=normalized)
        train_params = ClassifierTrainParams(class_names=class_names, decision_bound=decision_bound)
        _, acc, _, _ = ensemble_classification_test(training_images,
                                                    generated_test_images,
                                                    true_labels,
                                                    init_params, train_params, plus_one=True)
        key = f"{alpha0}"
        if key not in plot_dict['accuracy'].keys():
            plot_dict['accuracy'][key] = [acc]
        else:
            plot_dict['accuracy'][key].append(acc)
    return plot_dict


def visualize_plot_dict(n_neurons_list, plot_dict, n_features):
    fig, ax = plt.subplots()
    line_legends = []
    acc_matrix = []
    keys = list(plot_dict['accuracy'].keys())
    for key in keys:
        acc_matrix.append(plot_dict['accuracy'][key])
    acc_matrix = np.asarray(acc_matrix)
    max_index = np.argmax(np.mean(acc_matrix, axis=1))
    best = keys[max_index]

    for key in keys:
        if key != best:
            line, = ax.plot(n_neurons_list, plot_dict['accuracy'][key], label=key, marker='o', alpha=0.4)
        else:
            line, = ax.plot(n_neurons_list, plot_dict['accuracy'][key], label=key, marker='o')
        line_legends.append(line)

    ax.legend(title='Learning rate', handles=line_legends)
    ax.set_xlim(-0.05, n_neurons_list[-1] + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Neurons per class")
    ax.set_ylabel('Test accuracy')
    epochs = plot_dict["epochs"]
    ax.set_title(f"{n_features} descriptors, {epochs} epochs")
    ax.xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))
    plt.show()


def check_test_params(test_params: TestParams):
    if len(test_params.epochs_list) + len(test_params.normalized_list) + len(test_params.decision_bound_list) + \
            len(test_params.detector_params_list) + len(test_params.reduction_number_list) != 5:
        raise ValueError(
            "Lengths of all iterable parameters of a 'test_params' argument (except of alpha0_list) must be equal to "
            "1 for current experiment!")


def run(training_images, class_names,
        test_params: TestParams = DEFAULT_TEST_PARAMS,
        n_neurons_list=DEFAULT_N_NEURONS_LIST,
        transformation_params: ImageTransformationsParams = DEFAULT_TRANSFORMATION_PARAMS):
    check_test_params(test_params)
    plot_dict = get_plot_dict(n_neurons_list, training_images, class_names, test_params, transformation_params)
    visualize_plot_dict(n_neurons_list, plot_dict, test_params.n_features)
