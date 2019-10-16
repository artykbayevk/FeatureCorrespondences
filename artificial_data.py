import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.spatial.distance import euclidean
from itertools import product


class Dataset:
    def __init__(self, max_value, min_value, number_of_p, number_of_q):
        self.max_value = max_value
        self.min_value = min_value
        self.p_number = number_of_p
        self.q_number = number_of_q

    def figure(self, origin_x, origin_y, ellipse, radius_x, radius_y, angle, plot_figure=True):
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
        return self.p_number + self.q_number

    def get_distances(self, p_x, p_y, q_x, q_y, norm_value):
        """

        :param p_x:
        :param p_y:
        :param q_x:
        :param q_y:
        :param norm_value:
        :return: dict of pairs
        """
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
