import cv2
import glob


def get_clean_classname(file_path: str, extention: str = '.webp'):
    """ Функція для отримання назви класу з рядку шляху до файлу зображення. """
    ext_len = len(extention) if extention.startswith('.') else len(extention) + 1
    return file_path.split('\\')[-1][:-ext_len]


def load_images(file_paths: list):
    images = []
    for i, path in enumerate(file_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        images.append(img)
    return images


def filter_paths_by_classnames(file_paths: list, classnames: list, extention:str='.webp'):
    chosen_paths = list(filter(lambda path: get_clean_classname(path, extention) in classnames, file_paths))
    sorted_chosen_paths = sorted(chosen_paths, key=lambda o: classnames.index(get_clean_classname(o)))
    return sorted_chosen_paths


def filter_and_load(file_paths:list, classnames:list, extention:str='.webp'):
    """ Функція для фільтрації необхідних зображень за назвою класу
        та завантаження їх даних. """
    filtered_paths = filter_paths_by_classnames(file_paths, classnames, extention)
    images = load_images(filtered_paths)
    return images


def load_training_images(images_path, class_names_path, extention:str='.webp'):
    class_names = None
    with open(class_names_path, 'r') as file:
        class_names = file.readlines()

    def drop_last_n(path: str):
        if path[-1] == '\n':
            return path[:-1]
        return path

    class_names = list(map(drop_last_n, class_names))
    extention = '.' + extention if extention[0] != '.' else extention
    reg_pattern = f"\*{extention}" if images_path[-1] != '\\' else f"*{extention}"
    found_images_paths = glob.glob(images_path + reg_pattern)
    training_images = filter_and_load(found_images_paths, class_names, extention)
    return training_images, class_names
