import numpy as np
import scipy as sp
from scipy.spatial import distance

class Hausdorff:
    def __init__(self, u, v):
        self.u = u
        self.v = v
    def distance(self, d_type, criteria):
        if d_type == "euc":
            d = self.euclidean(self.u, self.v)
        elif d_type == "man":
            d = self.manhattan(self.u, self.v)
        elif d_type =="cheb":
            d = self.chebyshev(self.u, self.v)
        min_ = d.min(axis=1)

        # min_2 = d.min(axis=0)
        # min_ = np.concatenate((min_1, min_2))
        stop = 1
        if criteria == "max":
            return np.max(min_)
        elif criteria == "avg":
            return np.mean(min_)

    def euclidean(self,A,B):
        return distance.cdist(A, B)

    def manhattan(self, A,B):
        return distance.cdist(A, B, metric="cityblock")

    def chebyshev(self, A,B):
        return distance.cdist(A,B, metric='chebyshev')
