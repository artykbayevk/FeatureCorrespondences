#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import pandas as pd
import operator
import glob

class Dataset:
    def __init__(self, max_value, min_value, number_of_p, number_of_q, samples=1, d_of_p = 0.5):
        self.max_value = max_value
        self.min_value = min_value
        self.p_number = number_of_p
        self.q_number = number_of_q
        assert self.q_number / self.p_number >= 2
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
        Creating figure of P and Q features
        :param angle:
        :param plot_figure:
        :return: p_x, p_y, q_x, q_y - 4 arrays of points of P and Q figures
        """
        d = self.d_of_p
        p_n = self.p_number

        p_y, p_x = np.arange(-d * int(p_n / 2), d * int(p_n / 2) + d, d) , np.zeros(p_n)

        q_x = self.radius_on_x * np.cos(self.ellipse)
        q_y = self.radius_on_y * np.sin(self.ellipse)

        p_y_ = -p_x * np.sin(np.pi - angle) + p_y * np.cos(np.pi - angle) + self.origin_y
        p_x_ = p_x * np.cos(np.pi - angle) + p_y * np.sin(np.pi - angle) + self.origin_x

        q_y_ = -q_x * np.sin(np.pi - angle) + q_y * np.cos(np.pi - angle) + self.origin_y
        q_x_ = q_x * np.cos(np.pi - angle) + q_y * np.sin(np.pi - angle) + self.origin_x

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
        Function for generating all possible solutions with using Python
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

    def get_value_for_solution(self, p, q, solutions, type = "Py", angle = None):
        """
        choosing by the distances of P and Q features best or not best
        :param sol:
        :return:
        """
        def value_signer(p, q):
            p_p = cdist(p,p)
            q_q = cdist(q,q)
            best_sol = True
            counts = 0
            for i in range(0,p.shape[0] - 1):
                p_dists = p_p[i, i+1]
                q_dists = q_q[i, i+1]
                if p_dists * 2 <= q_dists:
                    best_sol = False
                    counts+=1

            new_counts = 0
            for i in range(0, p.shape[0] - 2):
                p1_to_p2 = p_p[i,i+1]
                q1_to_q2 = q_q[i,i+1]
                p2_to_p3 = p_p[i+1,i+2]
                q2_to_q3 = q_q[i+1,i+2]

                if p1_to_p2 * 2 <= q1_to_q2 and p2_to_p3* 2 <= q2_to_q3:
                    best_sol = False
                    new_counts+=1
            # self.draw_sol_2(p,q, "Best" if best_sol else "Opt")
            stop = 1
            if new_counts == 1 and counts == 2:
                best_sol = True
            elif counts == 1:
                best_sol = True

            # if best_sol:
            #     self.draw_sol_2(p, q, "Best" if best_sol else "Opt")

            return 1 if best_sol else 0

        def my_sort(mini_sol):
            axis = 1 # or can be 0
            phi = int(angle * 180/ np.pi)
            stop =1

            if phi in range(0, 45) or phi in range(135 ,225) or phi in range(315,361):
                axis = 1
            else:
                axis = 0
            stop =  1
            out = np.array(sorted(mini_sol, key=operator.itemgetter(axis)))
            return out

        if type == "Py":
            res = np.array(list(map(lambda sub: value_signer(p, q[sub[:,1]]), solutions)))
            best_idx = res > 0.0
            sols = solutions[best_idx]
            rand_sol = sols[np.random.randint(0, sols.shape[0] - 1, size=(1))[0]]
            self.draw_sol_2(p, q[rand_sol[:,1]])

            print("Best optimal solutions:{} and others:{} from:{}. {:.1f}/{:.1f} ".format(
                np.sum(res), res.shape[0] - np.sum(res), res.shape[0], np.sum(res)*100.0/res.shape[0],
                (res.shape[0]-np.sum(res))*100.0/res.shape[0]))
            return res
        elif type == 'LP':
            solutions = solutions.reshape(solutions.shape[0],int(solutions.shape[1]/4),4)
            solutions = np.array(list(map(lambda sub: my_sort(sub), solutions)))
            res = np.array(list(map(lambda sub: value_signer(sub[:, 0:2], sub[:,2:4]), solutions)))
            best_idx = res > 0.0
            sols = solutions[best_idx]
            rand_sol = sols[np.random.randint(0, sols.shape[0] - 1, size=(1))[0]]
            self.draw_sol_2(rand_sol[:,0:2], rand_sol[:, 2:4])
            print("Best optimal solutions:{} and others:{} from:{}. {:.1f}/{:.1f} ".format(
                np.sum(res), res.shape[0] - np.sum(res), res.shape[0], np.sum(res)*100.0/res.shape[0],
                (res.shape[0]-np.sum(res))*100.0/res.shape[0]))
            solutions = solutions.reshape(solutions.shape[0], solutions.shape[1] * solutions.shape[2])
            res = np.broadcast_to(np.array(res)[:, None], solutions.shape[:-1] + (1,))
            res = pd.DataFrame(np.concatenate((solutions, res), axis=-1))
            return res
        else:
            return None

    def draw_solution(self, p, q, solutions):
        idx = np.random.randint(0, solutions.shape[0], 1)[0]
        sol = solutions[idx]
        positions = np.zeros((p.shape[0], 4))

        positions[:, 0:2] = p[sol[:, 0]]
        positions[:, 2:4] = q[sol[:, 1]]

        for pos in positions:
            plt.plot([pos[0], pos[2]], [pos[1], pos[3]])
        plt.scatter(positions[:, 0], positions[:, 1])
        plt.scatter(positions[:, 2], positions[:, 3])
        plt.grid(color='lightgray', linestyle='--')
        plt.axis('equal')
        plt.show()

    def draw_sol_2(self, p, q, title=None):
        for i in range(p.shape[0]):
            plt.plot([p[i,0],q[i, 0]], [p[i,1], q[i,1]])
        plt.scatter(p[:, 0], p[:,1])
        plt.scatter(q[:, 0], q[:,1])
        if title:
            plt.title(title)
        plt.grid(color='lightgray', linestyle='--')
        plt.axis('equal')
        plt.show()

    def save_features(self,P,Q):
        path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial'
        P = pd.DataFrame(P)
        Q = pd.DataFrame(Q)
        P.to_csv(os.path.join(path, "P_new.csv"), header=None, index=None)
        Q.to_csv(os.path.join(path, "Q_new.csv"), header=None, index=None)

    def generate(self, LP = False):
        angle = np.pi / 6
        phi = int(angle * 180/ np.pi)
        p, q = self.figure(angle=angle, plot_figure=True)
        self.save_features(p,q)
        if LP:
            path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial'
            sols = pd.read_csv(os.path.join(path, 'solutions.csv'),header=None, index_col=None).values
            data = self.get_value_for_solution(p, q, sols, type = "LP", angle = angle)
            f_name = os.path.join(path, "data_{}.csv".format(phi))
            # data.to_csv(f_name, header=None, index=None)
        else:
            solutions = self.get_solutions(p, q)
            # draw random solution
            self.draw_solution(p, q, solutions)
            # choose best or not best optimal solution
            data = self.get_value_for_solution(p, q, solutions)

    def collect_data(self):
        """

        :return:
        """
        path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial'
        data_f_list =glob.glob(os.path.join(path, "data_*"))
        data = pd.concat([
            pd.read_csv(x, header=None) for x in data_f_list
        ], ignore_index=True)
        data.to_csv(os.path.join(path, "dataset.csv"), header = None, index= False)

    def dataset_info(self):
        """

        :return:
        """
        path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial'
        df = pd.read_csv(os.path.join(path, "dataset.csv"), header=None)
        sizeOfData = df.shape[0]
        sizeOfSample = df.shape[1] - 1
        countOfBest = df.iloc[:,-1][df.iloc[:,-1] > 0.0].shape[0]
        countOfOpt = df.iloc[:,-1][df.iloc[:, -1] == 0.0].shape[0]

        print("Total dataset size: {} samples per {} features. Best per Opt : {}/{}".format(sizeOfData, sizeOfSample, countOfBest, countOfOpt))



    def __str__(self):
        return "Figure with P:{} and Q:{}\nOrigin Point: {}:{}\nRadius on X:{} and radius on Y:{}".format(
            self.p_number, self.q_number, self.origin_x , self.origin_y, self.radius_on_x, self.radius_on_y
        )
#%%


dataset = Dataset(
    max_value=100,
    min_value=0,
    number_of_p=7,
    number_of_q=25)
print(dataset)
#%%
# dataset.generate(LP = True) # generating artificial dataset and run LP for choosing all optimal solutions
# dataset.collect_data() # collecting all generated data
dataset.dataset_info()