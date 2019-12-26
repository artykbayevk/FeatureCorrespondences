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
    def __init__(self, folder_path, size_of_sample):
        list_of_files = glob(os.path.join(folder_path, "*.csv"))
        dataset = np.zeros((len(list_of_files), size_of_sample))
        for idx, item in enumerate(list_of_files):
            dataset[idx] = pd.read_csv(item, header=None, delimiter=',').values.flatten()[:size_of_sample]


        self.df = pd.DataFrame(dataset)
        self.df['value'] =np.zeros(dataset.shape[0])
        self.df['class'] = np.zeros(dataset.shape[0])
        self.num_sols = int((self.df.shape[1]-1)/4)

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
            sample_ = np.array(sorted(sample.values[:-2].reshape(self.num_sols, 4), key=lambda k: [k[0], k[1]], reverse=True))
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
        # plt.boxplot(self.df['value'])
        # plt.show()

        dataset = sorted(self.df["value"])
        q1, q3 = np.percentile(dataset, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)

        max_value = np.max(self.df["value"])
        self.df['class'] = self.df['value'].apply(lambda x: 1.0 if x == max_value or x > upper_bound else 0.0)

        print("{} - {}/{}".format(self.df.shape[0],np.sum(self.df['class']),  self.df.shape[0] - np.sum(self.df['class'])))

    def save(self, out_path):
        df = self.df.drop("value", axis=1)
        if int(np.sum(self.df["class"])) == self.df.shape[0]:
            print("Class unbalancing\n")
        else:
            df.to_csv(out_path, index=None, header=False)
            print("Saved\n")

# HR.save(out_path=r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_1.csv')


# TODO found 1000 optimal solutions
# TODO use heuristic method for labeling them
# TODO ratio between BEST/NOT BEST will be 1/3

for i in range(1,23):
    sol_path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\pair_{}\experiment'.format(str(i))
    print("Solution: {}".format(str(i)))
    HR = HeuristicMethod(sol_path, size_of_sample=400)
    HR.rewrited_assign_values(n_neigbors=2)
    HR.save(out_path=r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_{}.csv'.format(str(i)))


