from models.utils import image_loading
from demo import demo
from experiments import experiment_1, experiment_3, experiment_4, experiment_5
from models.descriptors_filtration import show_reduction_impact

DEFAULT_IMAGES_PATH = r'..\resources\coins'
DEFAULT_CLASS_NAMES_PATH = r'..\resources\coins\classnames.txt'


def main():
    training_images, class_names = image_loading.load_training_images(DEFAULT_IMAGES_PATH, DEFAULT_CLASS_NAMES_PATH)
    # show_reduction_impact(training_images, 0, 500, 500, 2500)
    # demo(training_images, class_names, test_classification=True)
    # experiment_1.run(training_images, class_names)
    # experiment_2.run(training_images)
    # experiment_3.run(classifier, training_images)
    experiment_4.run(training_images, class_names)
    # experiment_5.run(training_images)
    

if __name__ == '__main__':
    main()
