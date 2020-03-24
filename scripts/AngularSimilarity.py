import math
import numpy as np
import pandas as pd
import scipy as sp
import point_cloud_utils as pcu
from sklearn.neighbors import KDTree
np.random.seed(42)


class AngularSimilarity:
    def __init__(self, a, b):
        """
        return angular similarity between point clouds
        :param a: point cloud A
        :param b: point cloud B
        """
        self.a = a
        self.b = b
        self.a_ = pcu.estimate_normals(a, k=10)
        self.b_ = pcu.estimate_normals(b, k=10)

    @staticmethod
    def compute_average(a, b, a_, b_):
        kdt = KDTree(b, leaf_size=30, metric="euclidean")
        indices = kdt.query(a, k=1, return_distance=False)

        s = np.arccos(abs(np.sum((a_ * b_[indices].reshape(-1, 3)), axis=1)))
        s = np.average(s[~np.isnan(s)])

        return s

    def compute_distance(self):
        res_1 = self.compute_average(self.a, self.b, self.a_, self.b_)
        res_2 = self.compute_average(self.b, self.a, self.b_, self.a_)
        return min(res_1, res_2)


if __name__ == '__main__':
    A = np.random.rand(100, 3)
    B = np.random.rand(100, 3)
    ang = AngularSimilarity(A, B)
    res = ang.compute_distance()
