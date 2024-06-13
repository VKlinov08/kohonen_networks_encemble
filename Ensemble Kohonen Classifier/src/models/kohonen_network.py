from scipy.spatial.distance import cdist
import numpy as np


def binary_normalization(distance: str, x_float: np.ndarray):
    ord = 1 if distance == 'cityblock' else 2
    norms = np.linalg.norm(x_float, ord=ord, axis=1)
    normalized = np.divide(x_float, norms[:, np.newaxis])
    return normalized


def vq_error(x_array: np.ndarray, distance: str, centroids=None, normalize_input: bool = False):
    """
    Vector Quantazation Error evaluation
    :param x_array:
    :param distance:
    :param centroids:
    :param normalize_input:
    :return:
    """
    if not isinstance(x_array.dtype, float):
        x_array = x_array.astype(float)
    if normalize_input:
        x_array = binary_normalization(distance, x_array)
    norms = cdist(x_array, centroids, metric=distance)
    min_norms = np.amin(norms, axis=1)
    n = min_norms.shape[0]
    return np.square(min_norms).sum() / n


class KohonenNet:
    def __init__(self, n_neurons: int, ndim: int, distance: str = 'cityblock', dtype=np.float64,
                 normalized: bool = False):
        self.n_neurons = n_neurons
        self.centroids = np.empty((n_neurons, ndim), dtype=dtype)
        self.train_distance = distance
        self.normalized = normalized
        self.train_iterations = 0

    def binary_normalization(self, train_set_float: np.ndarray):
        ord = 1 if self.train_distance == 'cityblock' else 2
        norms = np.linalg.norm(train_set_float, ord=ord, axis=1)
        normalized = np.divide(train_set_float, norms[:, np.newaxis])
        return normalized

    def error(self, x_array, centroids=None, normalize_input: bool = False):
        if centroids is None:
            centroids = self.centroids
        # if not isinstance(x_array.dtype, float):
        #     x_array = x_array.astype(float)
        # if normalize_input:
        #     x_array = self.binary_normalization(x_array)
        #
        # norms = cdist(x_array, centroids, metric=self.train_distance)
        # min_norms = np.amin(norms, axis=1)
        # n = min_norms.shape[0]
        # return np.square(min_norms).sum() / n
        return vq_error(x_array, self.train_distance, centroids, normalize_input)

    def training_step_(self, x, alpha):
        prev_centroids = self.centroids.copy()
        winner = np.argmin(cdist(np.expand_dims(x, axis=0),
                                 self.centroids, metric=self.train_distance))
        self.centroids[winner] = self.centroids[winner] + alpha * (x - self.centroids[winner])
        return prev_centroids

    def training_by_epochs_(self, train_set, epochs: int, alpha0: float):
        alpha = alpha0
        prev_centroids = None
        iteration = 1
        size = len(train_set)
        for epoch in range(epochs):
            for x in train_set:
                prev_centroids = self.training_step_(x, alpha)
                alpha = alpha0 * (1.0 - iteration / (epochs * size))
                iteration += 1
        return iteration

    def training_by_eps_(self, train_set, eps: float, alpha0: float):
        from itertools import cycle
        alpha = alpha0
        prev_centroids = None
        iteration = 1
        for x in cycle(train_set):
            prev_centroids = self.training_step_(x, alpha)
            iteration += 1
            alpha = alpha0 / iteration

            current_error = self.error(train_set)
            prev_error = self.error(train_set, prev_centroids)
            if np.abs(prev_error - current_error) < eps:
                return iteration

    def train(self, train_set: np.ndarray,
              epochs: int,
              alpha0: float,
              eps: float = None,
              init=None):
        train_set_float = train_set.astype(self.centroids.dtype)
        train_set_float = self.binary_normalization(train_set_float) if self.normalized else train_set_float
        if init is not None:
            self.centroids = self.binary_normalization(init) if self.normalized else init
        else:
            self.centroids = train_set_float[np.random.randint(train_set.shape[0], size=(self.n_neurons))]

        self.train_iterations = self.training_by_epochs_(train_set_float, epochs, alpha0) - 1
        if eps is not None:
            self.train_iterations += self.training_by_eps_(train_set_float, eps, alpha0) - 1

    def classify(self, test_samples: np.ndarray) -> tuple:
        test_samples_float = test_samples.astype(self.centroids.dtype)
        test_set = self.binary_normalization(test_samples_float) if self.normalized else test_samples_float
        distances = cdist(test_set,
                          self.centroids, metric=self.train_distance)
        labels = np.argmin(distances, axis=1)
        minimums = np.asarray([distances[i, labels[i]] for i in np.arange(len(test_set))])
        return labels, minimums
