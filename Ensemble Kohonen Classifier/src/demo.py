from sklearn.metrics import accuracy_score
from models.learning_interface import *
import matplotlib.pyplot as plt
from models.utils.utils import make_test_images


def demo(train_images, class_names, test_classification=True):
    init_params = ClassifierInitParams(10, 50, 0.5, normalized=True)
    classifier = EnsembleImageClassifier(init_params)
    train_params = ClassifierTrainParams(3, 25, 0.2, class_names=class_names, detector_params=500)
    classifier.train(train_images, train_params)
    for sample, class_name in zip(train_images, class_names):
        label_id, stats = classifier.classify_image(sample, 0.25, by_name=True, with_stats=True)

        plt.imshow(sample)
        title_color = 'green' if class_name == label_id else 'red'
        plt.title(label_id, color=title_color)
        plt.show()
        print(stats)

    if test_classification:
        transformations_dict = {
            "rotation_angles": [30],
            "scaling": [0.5],
            "shifting": [150, -150]
        }
        generated_test_images, true_labels = make_test_images(train_images, transformations_dict, with_labels=True)
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
