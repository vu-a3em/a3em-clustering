import numpy as np

class ClusterFilter:
    def __init__(self, max_clusters, max_weight, embedding_size, thresh):
        self.means = np.zeros((max_clusters, embedding_size))
        self.weights = np.zeros((max_clusters,))
        self.max_weight = max_weight
        self.max_clusters = max_clusters
        self.base_radius = thresh

    def insert(self, mean: np.ndarray) -> bool:
        mean = np.array(mean)
        assert mean.shape == self.means.shape[1:] and self.means.shape[0] == self.max_clusters and self.weights.shape == (self.max_clusters,)
        l2_norm = np.sqrt(np.sum((self.means - mean) ** 2, axis = 1))
        close = l2_norm <= np.sqrt(self.weights) * self.base_radius

        if np.any(close):
            center = (np.sum((self.means[close].T * self.weights[close]).T, axis = 0) + mean) / (np.sum(self.weights[close]) + 1)
            weight = min(np.sum(self.weights[close]) + 1, self.max_weight)
            self.means = np.concatenate([
                np.zeros((self.means.shape[0] - (np.sum(~close) + 1), self.means.shape[1])),
                self.means[~close],
                [ center ],
            ])
            self.weights = np.concatenate([
                np.zeros((self.weights.shape[0] - (np.sum(~close) + 1),)),
                self.weights[~close],
                [ weight ],
            ])
            return False
        else:
            self.means = np.concatenate([
                self.means[1:],
                [ mean ],
            ])
            self.weights = np.concatenate([
                self.weights[1:],
                [ 1 ],
            ])
            return True

if __name__ == '__main__':
    f = ClusterFilter(3, 5, 2, 0.5)
    def check(x, y):
        if not np.all(np.abs(f.means - x) < 0.0001) or not np.all(np.abs(f.weights - y) < 0.0001):
            raise RuntimeError(f'filter error:\n\n{f.means}\n\n{f.weights}')
    check([(0, 0), (0, 0), (0, 0)], [0, 0, 0])

    assert f.insert([1, 2]) == True
    check([(0, 0), (0, 0), (1, 2)], [0, 0, 1])

    assert f.insert([-3, 2]) == True
    check([(0, 0), (1, 2), (-3, 2)], [0, 1, 1])

    assert f.insert([-3.2, 2.1]) == False
    check([(0, 0), (1, 2), (-3.1, 2.05)], [0, 1, 2])

    assert f.insert([-4, 1]) == True
    check([(1, 2), (-3.1, 2.05), (-4, 1)], [1, 2, 1])

    assert f.insert([-4, 1.8]) == True
    check([(-3.1, 2.05), (-4, 1), (-4, 1.8)], [2, 1, 1])

    assert f.insert([-3.4, 2.2]) == False
    check([(-4, 1), (-4, 1.8), (-3.2, 2.1)], [1, 1, 3])

    assert f.insert([-4.1, 1.41]) == False
    check([(0, 0), (-3.2, 2.1), (-4.033333, 1.4033333)], [0, 3, 3])

    assert f.insert([-3, 2]) == False
    check([(0, 0), (-4.033333, 1.4033333), (-3.15, 2.075)], [0, 3, 4])

    assert f.insert([-3.59, 1.74]) == False
    check([(0, 0), (0, 0), (-3.53624875, 1.78125)], [0, 0, 5])

    print('passed all tests (no output means good)')
