from unittest import TestCase
import numpy as np
from stratified_group_kfold import StratifiedGroupKFold


def gen_p_dist(n_labels):
    p_label = np.array([np.random.randint(1, 100) for _ in range(n_labels)])
    p_label = p_label / np.sum(p_label)
    return p_label


# sample <n> times from distribution <dist>
def sample(n, dist):
    labels = np.expand_dims(np.random.rand(n), 1) - np.expand_dims(
        np.cumsum(dist), 0)
    return np.argmax(labels < 0, axis=1)


def calc_dist(samples):
    c = np.bincount(samples)
    return c / np.size(samples)


class TestStratifiedGroupKFold(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestStratifiedGroupKFold, self).__init__(*args, **kwargs)
        self.n_samples = 500
        self.n_labels = 5
        self.n_splits = 5
        self.n_groups = 100

        self.x = [i for i in range(self.n_samples)]
        self.labels = sample(self.n_samples, gen_p_dist(self.n_labels))
        self.labels_dist = calc_dist(self.labels)
        self.groups = sample(self.n_samples, gen_p_dist(self.n_groups))

    def test_determinancy(self):
        for seed in range(10):
            kf1 = StratifiedGroupKFold(self.n_splits,
                                       shuffle=True,
                                       random_state=seed)
            kf2 = StratifiedGroupKFold(self.n_splits,
                                       shuffle=True,
                                       random_state=seed)
            for (_, index1), (_, index2) in zip(
                    kf1.split(self.x, self.labels, self.groups),
                    kf2.split(self.x, self.labels, self.groups)):
                self.assertEqual(set(index1), set(index2))

    def test_random(self):
        # generate reference splits
        kf = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=3)
        ref_test_index_list = [
            set(test) for _, test in kf.split(self.x, self.labels, self.groups)
        ]

        from itertools import permutations

        def sum_intersections(sets1, sets2):
            return sum([len(s1 & s2) for s1, s2 in zip(sets1, sets2)])

        scores = []
        for seed in range(10):
            kf = StratifiedGroupKFold(self.n_splits,
                                      shuffle=True,
                                      random_state=seed)
            test_index_list = [
                set(test)
                for _, test in kf.split(self.x, self.labels, self.groups)
            ]
            score = max([
                sum_intersections(ref_test_index_list, s)
                for s in permutations(test_index_list)
            ]) / self.n_samples
            scores.append(score)
        self.assertTrue(np.mean(scores) < .5)
