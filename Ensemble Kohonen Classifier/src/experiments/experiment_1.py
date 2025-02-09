import sys
sys.path.append('..')

from .common import *
from models.utils.utils import *
from models import kohonen_network, ensemble, learning_interface, descriptors_filtration

"""### A research study on the utilization of a single Kohonen network"""
DEFAULT_TEST_PARAMS = TestParams()


def test_single_kohonen_net(all_bits, init_centers,
                            training_images,
                            generated_test_images,
                            true_labels,
                            init_params, train_params,
                            plus_one=True):
    """
    Function for training and testing an image classifier initialized by centroids
    of a single Kohonen Network, which was trained on a common set of descriptors.
    :param all_bits: np.ndarray of matrices of binary descriptors for each training image.
    :param init_centers: set of descriptors for initial centers.
    :param training_images: list of np.ndarray matrices with RGB data
    :param generated_test_images: list of np.ndarray matrices with RGB data made with affine transformations
    :param true_labels: correct labels for each test image
    :param init_params: instance of ClassifierInitParams
    :param train_params: instance of ClassifierTrainParams
    :param plus_one: generated_test_images was made with `plus_one` option which means all labels
    starts from 1 instead of 0.
    :return: trained classifier EnsembleImageClassifier
    """
    kohnet = kohonen_network.KohonenNet(init_params.n_neurons, 256, 'cityblock', normalized=True)
    kohnet.train(all_bits, init_params.epochs, init_params.alpha0, init=init_centers)
    centroids = kohnet.centroids

    classifier = learning_interface.EnsembleImageClassifier(init_params)
    classifier.set_class_names(train_params.class_names)
    n_classes = len(init_centers)
    classifier.ensemble = ensemble.ParallelSelfLearningEnsemble(n_classes, 1, 256, normalized=init_params.normalized)
    classifier.ensemble.centroids = centroids

    sample_time_avg = 0.0
    for i, (img, class_name) in enumerate(zip(training_images, train_params.class_names)):

        sample_classification_start = perf_counter()
        print(class_name, classifier.classify_image(img, train_params.decision_bound, by_name=True, with_stats=True))
        classification_time = perf_counter() - sample_classification_start

        print(f"\t{classification_time:.4f} s for reference image classification")
        sample_time_avg += classification_time
    print(f"\t{sample_time_avg / len(training_images):.4f} s - average time for reference image classification")

    pred_labels = []
    start_testing = perf_counter()
    for test, true_label in zip(generated_test_images, true_labels):
        label, density = classifier.classify_image(test, train_params.decision_bound, with_stats=True)
        if plus_one:
            label = -1 * true_label if label == -1 or label + 1 != true_label else label + 1
        else:
            label = label if label == true_label else -1 * true_label
        pred_labels.append(label)

    testing_time = perf_counter() - start_testing
    print(f"Test Time: {testing_time:.4f} seconds. ~{testing_time / 360:.4f} s per image.")
    print("Accuracy:")
    acc = accuracy_score(true_labels, pred_labels)
    print(f"{acc:.3f} / 1.")
    print(np.unique(pred_labels, return_counts=True))
    return classifier


def run(training_images, class_names,
        test_params: TestParams = DEFAULT_TEST_PARAMS,
        transformation_params: ImageTransformationsParams = DEFAULT_TRANSFORMATION_PARAMS,
        plus_one=True):
    # One neuron per a reference image trained in a common set of descriptors
    # Get Top-1 descriptor for each class.
    N_FEATURES = test_params.n_features
    sample_descriptors, train_kp = get_descriptors_and_keypoints(training_images, N_FEATURES)
    sample_bits = np.asarray([np.unpackbits(sample, axis=1) for sample in sample_descriptors])

    init_centers = descriptors_filtration.database_descriptors_reduction(sample_bits, 1)
    n_classes = len(training_images)
    init_centers = np.reshape(init_centers, (n_classes, 256))
    all_bits = np.vstack(sample_bits)
    n_neurons_list = [len(sample_bits)]

    # Train a Kohonen Network using a common set of descriptors and initial centers we have got recently.
    generator = get_test_params_generator(n_neurons_list, test_params)
    generated_test_images, true_labels = make_test_images(training_images,
                                                          transformation_params,
                                                          with_labels=True,
                                                          plus_one=True)

    for n_neurons, epochs, alpha0, normalized, detector_params, decision_bound, _ in generator:
        print("\n" + "-" * 100)
        print(n_neurons, epochs, alpha0, normalized, detector_params, decision_bound)

        init_params = learning_interface.ClassifierInitParams(n_neurons,
                                                              epochs,
                                                              alpha0,
                                                              normalized=normalized)
        train_params = learning_interface.ClassifierTrainParams(class_names=class_names,
                                                                decision_bound=decision_bound)
        test_single_kohonen_net(all_bits, init_centers,
                                training_images,
                                generated_test_images,
                                true_labels,
                                init_params, train_params, plus_one)
    pass