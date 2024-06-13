from .kohonen_network import *
from concurrent import futures


class SelfLearningEnsemble:
    def __init__(self, n_learners: int, n_neurons: int, ndim: int, distance='cityblock', normalized: bool = False):
        self.ensemble = [KohonenNet(n_neurons, ndim, distance=distance, normalized=normalized) for _ in
                         range(n_learners)]
        self.centroids = np.empty((n_learners * n_neurons, ndim), dtype=np.float64)
        self.class_indices = np.repeat(np.arange(n_learners), n_neurons)
        self.train_distance = distance
        self.n_neurons = n_neurons
        self.n_learners = n_learners
        self.normalized = normalized
        pass

    def binary_normalization(self, x_float: np.ndarray):
        return binary_normalization(self.train_distance, x_float)

    def train(self, train_sets: np.ndarray, epochs: int, alpha0: float, eps: float = None):
        for i, (learner, train_set) in enumerate(zip(self.ensemble, train_sets)):
            learner.train(train_set, epochs, alpha0, eps=eps)
            self.centroids[i * self.n_neurons:(i + 1) * self.n_neurons] = learner.centroids

    def classify_descriptors_(self, test_set: np.ndarray) -> tuple:
        test_set_float = test_set.astype(self.centroids.dtype)
        test_set_float = self.binary_normalization(test_set_float) if self.normalized else test_set_float
        distances = cdist(test_set_float,
                          self.centroids,
                          metric=self.train_distance)
        centroids_labels = np.argmin(distances, axis=1)
        min_distances = np.asarray([distances[i, centroids_labels[i]]
                                    for i in np.arange(len(test_set))])

        class_labels = self.class_indices[centroids_labels]
        return class_labels, min_distances

    def classify(self, test_set: np.ndarray, decision_bound: float, with_stats=False):
        class_labels, min_distances = self.classify_descriptors_(test_set)
        density, bins = np.histogram(class_labels,
                                     bins=np.arange(self.n_learners + 1),
                                     density=True)

        if np.amax(density) < decision_bound:
            if with_stats:
                return -1, density
            else:
                return -1
        if with_stats:
            return np.argmax(density), density
        return np.argmax(density)

    def error(self, x_array, centroids=None, normalize_input: bool = False):
        if centroids is None:
            centroids = self.centroids
        return vq_error(x_array, self.train_distance, centroids, normalize_input)


"""### Мультипотоковий ансамбль"""

def train_kohonen_net(net: KohonenNet,
                      train_set: np.ndarray,
                      epochs: int, alpha0: float,
                      eps: float = None) -> KohonenNet:
    net.train(train_set, epochs, alpha0, eps)
    return net


class ParallelSelfLearningEnsemble(SelfLearningEnsemble):
    def __init__(self, n_learners: int, n_neurons: int,
                 ndim: int, distance: str = 'cityblock', n_workers: int = 5, normalized: bool = False):
        super().__init__(n_learners, n_neurons, ndim, distance, normalized)
        self.n_workers = n_workers
        pass

    def train(self, train_sets: np.ndarray, epochs: int, alpha0: float, eps: float = None):
        with futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_tests = {}
            for i, (learner, train_set) in enumerate(zip(self.ensemble, train_sets)):
                future_to_tests[executor.submit(train_kohonen_net,
                                                learner,
                                                train_set,
                                                epochs, alpha0, eps=eps)] = i

            for future in futures.as_completed(future_to_tests):
                try:
                    class_id = future_to_tests[future]
                    learner = future.result()
                    self.centroids[class_id * self.n_neurons:(class_id + 1) * self.n_neurons] = learner.centroids
                except Exception as exc:
                    print('Згенеровано виключення під час навчання KohonenNet: %s' % (exc))
                    print(class_id, class_id * self.n_neurons, (class_id + 1) * self.n_neurons)
        return



