import cv2
import glob


def load_images(file_paths: list) -> list:
    """
    Function for image loading from the computer by paths in the `file_paths` list.
    Since images can have different size, container `images` is a list and not a numpy.ndarray.
    :param file_paths: list of strings with paths to images on the computer.
    :return: list of numpy.ndarray matrices.
    """
    images = []
    for i, path in enumerate(file_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        images.append(img)
    return images


def get_clean_classname(file_path: str, extension: str = '.webp') -> str:
    """
    Function for extracting a class label from given filename.
    :param file_path: path string e.g. 'resources/apple.png'.
    :param extension: image file extension string.
    :return: string of a class label e.g 'apple' instead of 'resources/apple.png'.
    """
    ext_len = len(extension) if extension.startswith('.') else len(extension) + 1
    return file_path.split('\\')[-1][:-ext_len]


def filter_paths_by_classnames(file_paths: list, classnames: list, extension: str = '.webp'):
    """
    From a list of paths take only those paths which represent classes in `classnames` list.
    :param file_paths: list of strings with paths to images on the computer.
    :param classnames: list of strings of class labels for classification.
    :param extension: file extension of images.
    :return: list of paths for class labels in `classnames` list.
    """
    chosen_paths = list(filter(lambda path: get_clean_classname(path, extension) in classnames, file_paths))
    sorted_chosen_paths = sorted(chosen_paths, key=lambda o: classnames.index(get_clean_classname(o)))
    return sorted_chosen_paths


def filter_and_load(file_paths: list, classnames: list, extension: str = '.webp'):
    """ Function for loading images from `file_paths` list by `classnames` labels. """
    filtered_paths = filter_paths_by_classnames(file_paths, classnames, extension)
    images = load_images(filtered_paths)
    return images


def load_training_images(images_path, class_names_path, extension: str = '.webp'):
    """
    Function for loading
    :param images_path: path string to a folder where images are stored on the computer.
    :param class_names_path: path string to a file that stores all class labels for classification process.
    :param extension: file extension for images.
    :return: list of numpy.ndarray of images and list of class labels string.
    """
    # class_names = None
    with open(class_names_path, 'r') as file:
        class_names = file.readlines()

    def drop_last_n(path: str):
        if path[-1] == '\n':
            return path[:-1]
        return path

    extension = '.' + extension if extension[0] != '.' else extension
    reg_pattern = "\\*" + extension if images_path[-1] != '\\' else "*" + extension
    found_images_paths = glob.glob(images_path + reg_pattern)

    class_names = list(map(drop_last_n, class_names))
    training_images = filter_and_load(found_images_paths, class_names, extension)
    return training_images, class_names
