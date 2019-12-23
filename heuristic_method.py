#%% code for selecting best or not best using heuristic method
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from numpy import random,argsort,sqrt
from scipy.spatial.distance import cdist
np.random.seed(42)



class HeuristicMethod:
    """

    """
    def __init__(self, folder_path, size_of_sample, artificial_data_count , artifical_data=False):
        list_of_files = glob(os.path.join(folder_path, "*.csv"))
        dataset = np.zeros((len(list_of_files), size_of_sample))
        for idx, item in enumerate(list_of_files):
            dataset[idx] = pd.read_csv(item, header=None, delimiter=',').values.flatten()

        if artifical_data:
            gen_df = pd.DataFrame(columns=range(280), index=range(artificial_data_count))
            sample_df = np.copy(dataset)
            for i in range(artificial_data_count):
                rand_index = np.random.randint(0, sample_df.shape[0], size=(1,)).item()
                sample = sample_df[rand_index, :].reshape(140, 2)

                P = sample[::2]
                Q = sample[1::2]
                np.random.shuffle(P)
                np.random.shuffle(Q)

                sample = np.stack((P, Q), axis=1).reshape(280)
                gen_df.iloc[i] = sample
            dataset = np.concatenate((dataset, gen_df))

        self.df = pd.DataFrame(dataset)
        self.df['value'] =np.zeros(dataset.shape[0])
        self.df['class'] = np.zeros(dataset.shape[0])

    def assign_values(self):
        for idx, sample in self.df.iterrows():
            sample_ = np.array(sorted(sample.values[:-2].reshape(70,4), key = lambda k:[k[0], k[1]], reverse=True))
            P = sample_[:, :2] # P figure x and y
            Q = sample_[:, 2:] # Q figure x and y

            # plt.scatter(P[:,0],P[:,1])
            # plt.scatter(Q[:,0],Q[:,1])
            # plt.show()
            stop = 1
            count = 0
            p1_to_p2 = cdist(P,P)
            q1_to_q2 = cdist(Q,Q)
            for i in range(P.shape[0]-1):
                p_p = p1_to_p2[i, i+1]
                q_q = q1_to_q2[i, i+1]
                dists = [min(q_q, p_p), max(q_q, p_p)]
                try:
                    ratio = dists[1]/dists[0]
                except:
                    ratio = 0
                if ratio<=1.50:
                    count+=1

            # print("{:.2f} % out of {} correspondences are passed treshold".format(count*100/P.shape[0], P.shape[0]))
            self.df.iloc[idx, -2] = count*100/P.shape[0]
        # setting values on term of their count of
        plt.boxplot(self.df['value'])
        plt.show()

        plt.hist(self.df['value'])
        plt.show()


        max_value = np.max(self.df["value"])
        self.df['class'] = self.df['value'].apply(lambda x: 1.0 if x == max_value or x >= max_value - 1.0 else 0.0)
        print("{} num samples, {} samples {:.2f}% best and {} samples {:.2f}% nor-best".format(
            self.df.shape[0], np.sum(self.df['class']),np.sum(self.df['class'])*100/self.df.shape[0],
            self.df.shape[0] - np.sum(self.df['class']), (self.df.shape[0] - np.sum(self.df['class']))*100/self.df.shape[0]
        ))

    def rewrited_assign_values(self, n_neigbors):
        def knn_search(x, D, K):
            """ find K nearest neighbours of data among D """
            ndata = D.shape[1]
            K = K if K < ndata else ndata
            # euclidean distances from the other points
            sqd = sqrt(((D - x[:, :ndata]) ** 2).sum(axis=0))
            idx = argsort(sqd)  # sorting
            # return the indexes of K nearest neighbours
            return idx[:K+1]


        for idx, sample in self.df.iterrows():
            sample_ = np.array(sorted(sample.values[:-2].reshape(70, 4), key=lambda k: [k[0], k[1]], reverse=True))
            P = sample_[:, :2]  # P figure x and y
            Q = sample_[:, 2:]  # Q figure x and

            p1_to_p2 = cdist(P,P)
            q1_to_q2 = cdist(Q,Q)

            ## ranking algorithm implementation
            ## found Q - quality of mapping for each point
            Q_S = 0
            for p_id, p in enumerate(P):
                D = P.T
                x = p.reshape(-1,1)
                k_points = knn_search(x, P.T, n_neigbors)
                N_P = D[:, k_points]
                M_P = Q.T[:,k_points]

                D_N_P = np.sum(cdist(N_P,N_P))/n_neigbors**2
                D_M_P = np.sum(cdist(M_P,M_P))/n_neigbors**2
                D_Np_Mp = np.abs(D_N_P-D_M_P)
                Q_np_Mp = 1/(1+D_Np_Mp)
                Q_S+=Q_np_Mp
            # print(Q_S)
            self.df.iloc[idx, -2] = Q_S
        plt.boxplot(self.df['value'])
        plt.show()

        max_value = np.max(self.df["value"])
        self.df['class'] = self.df['value'].apply(lambda x: 1.0 if x == max_value else 0.0)
        print("{} num samples, {} samples {:.2f}% best and {} samples {:.2f}% nor-best".format(
            self.df.shape[0], np.sum(self.df['class']), np.sum(self.df['class']) * 100 / self.df.shape[0],
                                                        self.df.shape[0] - np.sum(self.df['class']),
                                                        (self.df.shape[0] - np.sum(self.df['class'])) * 100 /
                                                        self.df.shape[0]
        ))

    def save(self, out_path):
        df = self.df.drop("value", axis=1)
        df.to_csv(out_path, index=None, header=False)
        print("Saved")


folder_path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\pair_22\experiment'
HR = HeuristicMethod(
    folder_path, size_of_sample=280, artificial_data_count = 1000,artifical_data=False
)
HR.rewrited_assign_values(n_neigbors = 5)
# HR.assign_values()
# HR.save(out_path=r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_1.csv')


# TODO found 1000 optimal solutions
# TODO use heuristic method for labeling them
# TODO ratio between BEST/NOT BEST will be 1/3

