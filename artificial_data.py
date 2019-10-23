#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import pandas as pd

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

        :param angle:
        :param plot_figure:
        :return: p_x, p_y, q_x, q_y - 4 arrays of points of P and Q figures
        """
        d = self.d_of_p
        p_n = self.p_number

        p_y, p_x = np.arange(-d * int(p_n / 2), d * int(p_n / 2) + d, d) + self.origin_x, np.zeros(p_n) + self.origin_x

        q_x = self.origin_x + self.radius_on_x * np.cos(self.ellipse)
        q_y = self.origin_y + self.radius_on_y * np.sin(self.ellipse)

        p_y_ = -p_x * np.sin(np.pi - angle) + p_y * np.cos(np.pi - angle)
        p_x_ = p_x * np.cos(np.pi - angle) + p_y * np.sin(np.pi - angle)

        q_y_ = -q_x * np.sin(np.pi - angle) + q_y * np.cos(np.pi - angle)
        q_x_ = q_x * np.cos(np.pi - angle) + q_y * np.sin(np.pi - angle)

        if plot_figure:
            plt.scatter(q_x_, q_y_, label="Q")
            plt.scatter(p_x_, p_y_, label="P")

            plt.xlabel("X")
            plt.ylabel("Y")

            plt.axis('equal')
            plt.grid(color='lightgray', linestyle='--')
            plt.legend('QP', loc='upper left')
            plt.show()

        p = np.array([
            p_x_, p_y_
        ])
        q = np.array([
            q_x_, q_y_
        ])

        return p.T, q.T

    def get_solutions(self, p, q):
        """

        :param p:
        :param q:
        :return: dict of pairs
        """
        dist = cdist(p, q) <= self.radius_on_x
        pos = np.where(dist)

        p = pos[0]
        q = pos[1]

        d = [[] for i in range(np.unique(p).shape[0])]

        for i, j in zip(p, q):
            d[i].append(j)
        maximum = np.prod([len(x) for x in d])
        stop = 1
        res = []
        checker = set([])
        answer = []

        def solve(v):
            if len(res) == len(d):
                answer.append(res.copy())
            else:
                for i in d[v]:
                    if i not in checker:
                        checker.add(i)
                        res.append([v, i])
                        check = solve(v + 1)
                        if check is False:
                            res.pop(-1)
                            checker.remove(i)
                        else:
                            return True
            return False

        solve(0)
        solutions = np.array(answer)
        print("Generated: {} samples.".format(solutions.shape[0]))
        return solutions

    def get_value_for_solution(self, p, q, solutions):
        """

        :param sol:
        :return:
        """
        def value_signer(p, q):
            p_p = cdist(p,p)
            q_q = cdist(q,q)
            self.draw_sol_2(p,q)
            stop = 1

        y = np.zeros(shape=(solutions.shape[0], 1))


        res = np.array(list(map(lambda sub: value_signer(p, q[sub[:,1]]), solutions)))
        stop =  1
        return res

    def draw_solution(self, p, q, solutions):
        idx = np.random.randint(0, solutions.shape[0], 1)[0]
        sol = solutions[idx]
        positions = np.zeros((p.shape[0], 4))

        positions[:, 0:2] = p[sol[:, 0]]
        positions[:, 2:4] = q[sol[:, 1]]

        dist_P_Q = cdist(positions[:, 0:2], positions[:, 2:4])
        dist_P_P = cdist(positions[:, 0:2], positions[:, 0:2])
        dist_Q_Q = cdist(positions[:, 2:4], positions[:, 2:4])
        for pos in positions:
            plt.plot([pos[0], pos[2]], [pos[1], pos[3]])
        plt.scatter(positions[:, 0], positions[:, 1])
        plt.scatter(positions[:, 2], positions[:, 3])
        plt.grid(color='lightgray', linestyle='--')
        plt.axis('equal')
        plt.show()

    def draw_sol_2(self, p, q):
        for i in range(p.shape[0]):
            plt.plot([p[i,0],q[i, 0]], [p[i,1], q[i,1]])
        plt.scatter(p[:, 0], p[:,1])
        plt.scatter(q[:, 0], q[:,1])
        plt.grid(color='lightgray', linestyle='--')
        plt.axis('equal')
        plt.show()

    def save_features(self,P,Q):
        path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial'
        P = pd.DataFrame(P)
        Q = pd.DataFrame(Q)
        P.to_csv(os.path.join(path, "P_new.csv"), header=None, index=None)
        Q.to_csv(os.path.join(path, "Q_new.csv"), header=None, index=None)

    def generate(self):
        angle = np.pi
        p, q = self.figure(angle=angle, plot_figure=False)
        solutions = self.get_solutions(p, q)

        self.save_features(p,q)
        # self.draw_solution(p, q, solutions)

        data = self.get_value_for_solution(p, q, solutions)

    def __str__(self):
        return "Figure with P:{} and Q:{}\nOrigin Point: {}:{}\nRadius on X:{} and radius on Y:{}".format(
            self.p_number, self.q_number, self.origin_x , self.origin_y, self.radius_on_x, self.radius_on_y
        )
#%%


dataset = Dataset(
    max_value=100,
    min_value=0,
    number_of_p=7,
    number_of_q=23)
print(dataset)
#%%
dataset.generate()
