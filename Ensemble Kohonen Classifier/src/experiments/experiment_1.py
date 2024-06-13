from sklearn.metrics import accuracy_score
import numpy as np
import sys
sys.path.append('..')

from .common import *
from models.utils.utils import *
from models import kohonen_network, ensemble, learning_interface, descriptors_filtration

"""### Тестування одиничної мережі Кохонена"""
DEFAULT_TEST_PARAMS = TestParams()


def test_single_kohonen_net(all_bits, init_centers,
                            training_images,
                            generated_test_images,
                            true_labels,
                            init_params, train_params,
                            plus_one=True):
    kohnet = kohonen_network.KohonenNet(init_params.n_neurons, 256, 'cityblock', normalized=True)
    kohnet.train(all_bits, init_params.epochs, init_params.alpha0, init=init_centers)
    centroids = kohnet.centroids

    classifier = learning_interface.EnsembleImageClassifier(init_params)
    classifier.set_class_names(train_params.class_names)
    n_classes = len(init_centers)
    classifier.ensemble = ensemble.ParallelSelfLearningEnsemble(n_classes, 1, 256, normalized=init_params.normalized)
    classifier.ensemble.centroids = centroids

    sample_time_avg = 0.0
    for i, (sample, class_name) in enumerate(zip(training_images, train_params.class_names)):
        sample_classification_start = perf_counter()
        print(class_name, classifier.classify_image(sample, train_params.decision_bound, by_name=True, with_stats=True))
        classification_time = perf_counter() - sample_classification_start
        print(f"\t{classification_time:.4f} секунд на класифікацію еталона")
        sample_time_avg += classification_time
    print(f"\t{sample_time_avg / len(training_images):.4f} середній час на класифікацію еталона")

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
    print(f"Час тестування: {testing_time:.4f} секунд. ~{testing_time / 360:.4f} секунд на зображення")
    print("Accuracy")
    print(accuracy_score(true_labels, pred_labels))
    print(np.unique(pred_labels, return_counts=True))
    return classifier


def run(training_images, class_names,
        test_params:TestParams=DEFAULT_TEST_PARAMS,
        transformation_params: ImageTransformationsParams=DEFAULT_TRANSFORMATION_PARAMS):
    # Один нейрон на еталон (сумісно)
    # 1. Обрати вектори топ 1 за унікальністю для кажного еталону
    N_FEATURES = test_params.n_features
    sample_descriptors, train_kp = get_descriptors_and_keypoints(training_images, N_FEATURES)
    sample_bits = np.asarray([np.unpackbits(etalon, axis=1) for etalon in sample_descriptors])

    init_centers = descriptors_filtration.database_descriptors_reduction(sample_bits, 1)
    n_classes = len(training_images)
    init_centers = np.reshape(init_centers, (n_classes, 256))
    all_bits = np.vstack(sample_bits)
    n_neurons_list = [len(sample_bits)]

    # 2. Навчити мережу Кохонена на об'єднаній выбоці всії дескрипторів
    generator = test_parameters_generator(n_neurons_list, test_params)
    generated_test_images, true_labels = make_test_images(training_images, transformation_params, with_labels=True, plus_one=True)

    for n_neurons, epochs, alpha0, normalized, detector_params, decision_bound, _ in generator:
        print("\n" + "-" * 100)
        print(n_neurons, epochs, alpha0, normalized, detector_params, decision_bound)

        init_params = learning_interface.ClassifierInitParams(n_neurons, epochs, alpha0, normalized=normalized)
        train_params = learning_interface.ClassifierTrainParams(class_names=class_names, decision_bound=decision_bound)
        test_single_kohonen_net(all_bits, init_centers,
                                training_images,
                                generated_test_images,
                                true_labels,
                                init_params, train_params)
    pass