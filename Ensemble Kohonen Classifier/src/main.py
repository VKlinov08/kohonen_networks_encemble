from models.utils import image_loading
from demo import *
from experiments import experiment_2, experiment_3, experiment_4, experiment_5

DEFAULT_IMAGES_PATH = r'..\resources\coins'
DEFAULT_CLASS_NAMES_PATH = r'..\resources\coins\classnames.txt'


def main():
    training_images, class_names = image_loading.load_training_images(DEFAULT_IMAGES_PATH, DEFAULT_CLASS_NAMES_PATH)
    demo(training_images, class_names, test_classification=False)
    # experiment_2.run(training_images)
    # experiment_3.run(classifier, training_images)
    # experiment_4.run(training_images)
    # experiment_5.run(training_images)
    

if __name__ == '__main__':
    main()
