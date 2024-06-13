from sklearn.metrics import accuracy_score
from time import perf_counter
from ..models.learning_interface import *


def demo(train_images, class_names):
    init_params = ClassifierInitParams(3, 25, 0.5, normalized=True)
    classifier = EnsembleImageClassifier(init_params)
    classifier.train(train_images, class_names=class_names, detector_params=500)
    for sample, class_name in zip(train_images, class_names):
        label_id, stats = classifier.classify_image(sample, 0.25, by_name=True, with_stats=True)

        plt.imshow(sample)
        title_color = 'green' if class_name == label_id else 'red'
        plt.title(label_id, color=title_color)
        plt.show()
        print(stats)

    generated_test_images, true_labels = make_test_images(train_images, with_labels=True)
    current_labels = []
    for test_image, class_number in zip(generated_test_images, true_labels):
        label_id, stats = classifier.classify_image(test_image, 0.23, with_stats=True)
        plt.imshow(test_image)
        title_color = 'green' if class_number == label_id else 'red'
        if label_id == -1:
            plt.title("Mismatch", color='red')
        else:
            plt.title(class_names[label_id], color=title_color)
        plt.show()
        print(stats)
        print(class_number, label_id)
        current_labels.append(label_id)
    print(accuracy_score(true_labels, current_labels))
    pass

def ensemble_classification_test(train_images, test_images, true_labels,
                                 init_params: ClassifierInitParams,
                                 train_params: ClassifierTrainParams, plus_one=False):
    classifier = EnsembleImageClassifier(init_params)
    start_training = perf_counter()
    classifier.train(train_images, train_params)
    training_time = perf_counter() - start_training
    print(f"Час тренування: {training_time:.4f} секунд")

    # Класифікація еталонів
    classification_time_avg = 0.0
    for sample, class_name in zip(train_images, train_params.class_names):
        classification_start = perf_counter()
        print(class_name, classifier.classify_image(sample,
                                                    train_params.decision_bound,
                                                    by_name=True, with_stats=True, with_error=True))
        sample_classification_time = perf_counter() - classification_start
        print(f"\t{sample_classification_time:.4f} секунд на класифікацію еталона")
        classification_time_avg += sample_classification_time
    print("\n")
    print(f"\t{classification_time_avg / len(train_images):.4f} середній час на класифікацію еталона")

    # Класифікація тестових зображень
    pred_labels = []
    testing_start = perf_counter()
    for test, true_label in zip(test_images, true_labels):
        label, density = classifier.classify_image(test, decision_bound, with_stats=True)
        if plus_one:
            label = -1 * true_label if label == -1 or label + 1 != true_label else label + 1
        else:
            label = label if label == true_label else -1 * true_label
        pred_labels.append(label)

    testing_time = perf_counter() - testing_start
    print(f"Час тестування: {testing_time} секунд. ~{testing_time / 360:.5f} секунд на зображення")

    print("Accuracy")
    acc = accuracy_score(true_labels, pred_labels)
    print(acc)
    print(np.unique(pred_labels, return_counts=True))
    return classifier, acc, training_time, testing_time


from itertools import product

n_neurons_list = [1, 3, 5, 10, 25, 50]
epochs_list = [10]
alpha0_list = [0.25]
reduction_number_list = [None]
normalized_list = [True]
detector_params_list = [500]
decision_bound_list = [0.25]

generator = product(n_neurons_list,
                    epochs_list,
                    alpha0_list,
                    reduction_number_list,
                    normalized_list,
                    detector_params_list,
                    decision_bound_list)
generated_test_images, true_labels = make_test_images(train_images, with_labels=True, plus_one=True)
accuracy_list = []
lt_list, tt_list = [], []

for n_neurons, epochs, alpha0, reduction_number, normalized, detector_params, decision_bound in generator:
    if reduction_number is not None and reduction_number > detector_params:
        continue
    print("\n--------------------------------")
    print(n_neurons, epochs, alpha0, reduction_number, normalized, detector_params, decision_bound)
    cl, acc, lt, tt = ensemble_classification_test(train_images,
                                                   generated_test_images,
                                                   true_labels,
                                                   n_neurons, epochs, alpha0, reduction_number, normalized, detector_params,
                                                   decision_bound, plus_one=True)

    accuracy_list.append(acc)
    lt_list.append(lt)
    tt_list.append(tt)

from matplotlib import pyplot as plt
from matplotlib import ticker

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes[0].plot(n_neurons_list, lt_list, marker='o', color='red')
axes[0].set_xlim(-0.05, 51)
axes[0].set_xlabel("Кількість нейронів на еталон")
axes[0].set_ylabel('Час на навчання, c')
axes[0].xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))

axes[1].plot(n_neurons_list, tt_list, marker='o', color='blue')
axes[1].set_xlim(-0.05, 51)
axes[1].set_xlabel("Кількість нейронів на еталон")
axes[1].set_ylabel('Час на класифікацію тестової вибірки, c')
axes[1].xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))
plt.plot()

from matplotlib import pyplot as plt
from matplotlib import ticker

fig, ax = plt.subplots()
ax.plot(n_neurons_list, accuracy_list, marker='o')
ax.set_xlim(-0.05, 51)
ax.set_ylim(0, 1)
ax.set_xlabel("Кількість нейронів на еталон")
ax.set_ylabel('Точність тестової класифікації')
ax.set_title("500 дескрипторів, 10 епох навчання, коефіцієнт навчання 0.25 ")
ax.xaxis.set_major_locator(ticker.FixedLocator(n_neurons_list))
plt.plot()
