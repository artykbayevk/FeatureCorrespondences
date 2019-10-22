import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.spatial.distance import cdist


class Dataset:
    def __init__(self, max_value, min_value, number_of_p, number_of_q, samples=1, d_of_p=0.5):
        self.max_value = max_value
        self.min_value = min_value
        self.p_number = number_of_p
        self.q_number = number_of_q
        assert self.q_number / self.p_number >= 3
        assert self.p_number % 2 != 0

        self.ellipse = np.linspace(0, 2 * pi, self.q_number)
        self.origin_x = int(self.max_value / 2)
        self.origin_y = int(self.max_value / 2)
        self.samples = samples

        self.radius_on_x = 1.0
        self.radius_on_y = 0.5 * int(self.p_number / 2) + self.radius_on_x
        self.d_of_p = d_of_p

    def figure(self, angle, plot_figure=True):
        """

        :param origin_x:
        :param origin_y:
        :param ellipse:
        :param radius_x:
        :param radius_y:
        :param angle:
        :param plot_figure:
        :return: p_x, p_y, q_x, q_y - 4 arrays of points of P and Q figures
        """
        d = self.d_of_p
        p_n = self.p_number

        p_y, p_x = np.arange(-d * int(p_n / 2), d * int(p_n / 2) + d, d) + self.origin_x, np.zeros(p_n) + self.origin_x

        q_x = self.origin_x + self.radius_on_x * np.cos(self.ellipse)
        q_y = self.origin_y + self.radius_on_y * np.sin(self.ellipse)

        if plot_figure:
            plt.scatter(q_x, q_y, label="Q")
            plt.scatter(p_x, p_y, label="P")

            plt.xlabel("X")
            plt.ylabel("Y")

            plt.axis('equal')
            plt.grid(color='lightgray', linestyle='--')
            plt.legend('QP', loc='upper left')
            plt.show()

        P = np.array([
            p_x, p_y
        ])
        Q = np.array([
            q_x, q_y
        ])

        return P.T, Q.T

    def get_distances(self, P, Q):
        """

        :param p_x:
        :param p_y:
        :param q_x:
        :param q_y:
        :param norm_value:
        :return: dict of pairs
        """
        dist = cdist(P, Q) <= self.radius_on_x
        pos = np.where(dist)

        unique, index, counts = np.unique(pos[0], return_counts=True, return_index=True)

        P = pos[0]
        Q = pos[1]

        sample = np.zeros((np.unique(P).shape[0], 2))
        for i in range(P.shape[0]):
            print(P[i], Q[i])



        return self.min_value

    def get_solutions(self, pairs):
        """

        :param pairs:
        :return: ?
        """
        return pairs + self.p_number

    def get_value_for_solution(self, sol):
        """

        :param sol:
        :return:
        """

        return sol + self.q_number

    def generate(self):
        angle = np.pi
        P, Q = self.figure(angle=angle, plot_figure=False)
        dist = self.get_distances(P, Q)

    def __str__(self):
        return "Figure with P:{} and Q:{}\nOrigin Point: {}:{}\nRadius on X:{} and radius on Y:{}".format(
            self.p_number, self.q_number, self.origin_x, self.origin_y, self.radius_on_x, self.radius_on_y
        )


dataset = Dataset(
    max_value=100,
    min_value=0,
    number_of_p=5,
    number_of_q=21)
print(dataset)
dataset.generate()
