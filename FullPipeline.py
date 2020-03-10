import os
import glob
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.Triangulation.Depth import Triangulation
from scripts.Triangulation.HausdorffDist import Hausdorff
from scripts.email import send_email
from joblib import load

class Stereo:
    def __init__(self, path,n_components, plot_ground_truth = False, show_imgs = False, n_sols = 100):
        folder = glob.glob(os.path.join(path,'*'))
        '''
            collecting data from folder and put into folder_data variable
        '''
        folder_data = {}
        for item in folder:
            b_name = os.path.basename(item)
            if 'dataset' in b_name:
                folder_data['dataset'] = item
            elif 'left' in b_name:
                folder_data['left_img'] = item
            elif 'right' in b_name:
                folder_data['right_img'] = item

        with open(folder_data['dataset'], mode='r', encoding='utf-8') as f:
            content = f.readlines()
        content = [x for x in content if x.strip(' ') !='\n']
        K_data = [content[1:4],content[8:11]]
        RT_data = [content[4:7], content[11:]]
        K_data = np.array([
            np.array([ np.array([float(z) for z in y.strip('\n').strip(' ').split(' ')]) for y in x]) for x in K_data
        ])
        RT_data = np.array([
            np.array([ np.array([float(z) for z in y.strip('\n').strip(' ').split(' ')]) for y in x]) for x in RT_data
        ])

        '''
            assigning intrinsic and extrinsic parameters 
        '''

        self.main_path = path
        self.K = K_data[0]
        self.R1 = RT_data[1][:, :-1].T
        self.T1 = RT_data[1][:, -1]
        self.R2 = RT_data[0][:, :-1].T
        self.T2 = RT_data[0][:, -1]

        self.img1_path = folder_data['left_img']
        self.img2_path = folder_data['right_img']

        self.n_components = n_components
        self.draw_plot = plot_ground_truth
        self.show_imgs = show_imgs

        self.exp_dir = os.path.join(path, 'experiment')

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.limit_solutions = str(n_sols)



    def compute_ground_truth(self):
        opencv = Triangulation(K=self.K, R1=self.R1, R2=self.R2, T1=self.T1, T2=self.T2)
        opencv.load_imgs(self.img1_path, self.img2_path)
        opencv.findRootSIFTFeatures(n_components=self.n_components)
        opencv.matchingRootSIFTFeatures()
        opencv.findRTmatrices()
        if self.show_imgs:
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(opencv.img1, cmap="gray")
            f.add_subplot(1, 2, 2)
            plt.imshow(opencv.img2, cmap="gray")
            plt.show(block=True)
            matched_img = os.path.join(os.path.dirname(self.img1_path), "matchedGT.png")
            opencv.drawMathces(matched_img)
            matched = plt.imread(matched_img)
            plt.imshow(matched, cmap="gray")
            plt.show()
        opencv.point_cloud(plot=self.draw_plot, title="OpenCV")
        self.target = opencv.pts3D

    def full_evaluation(self, opt_solution_path, checkpoint_path, method):
        dataset = pd.read_csv(opt_solution_path, header=None, index_col=None)
        X = dataset.drop(dataset.columns[-1], axis=1).values
        Y = dataset[dataset.columns[-1]].values
        model = load(checkpoint_path)
        Y_ = model.predict(X)
        checker_flag = 0
        other_best_idx = []
        true_flag = int(np.sum(Y))
        all_sol_count = X.shape[0]
        cheb_all = np.zeros(all_sol_count)
        man_all = np.zeros(all_sol_count)
        euc_all = np.zeros(all_sol_count)
        idx_array = []

        res = np.sum(Y_[np.where(Y_ == Y)])

        if int(res) == 0:
            return False
        else:
            julia = Triangulation(K=self.K, R1=self.R1, R2=self.R2, T1=self.T1, T2=self.T2)
            julia.load_imgs(self.img1_path, self.img2_path)
            julia.findRootSIFTFeatures(n_components=self.n_components)

            for idx, item in enumerate(Y_):
                if Y[idx] == Y_[idx] and item == 1.0:
                    checker_flag+=1
                    idx_array.append(idx)
                else:
                    other_best_idx.append(idx)
                julia.matchingRootSIFTFeatures_advanced(X[idx])
                julia.findRTmatrices()
                julia.point_cloud(plot=self.draw_plot, title="Our method #{}".format(idx))
                pred = julia.pts3D
                metrics_1 = Hausdorff(u=pred, v=self.target)
                metrics_2 = Hausdorff(u=self.target, v=pred)

                if method == 'avg':

                    dist_cheb_avg = np.add(metrics_1.distance(d_type="cheb", criteria="avg"),
                                     metrics_2.distance(d_type="cheb", criteria="avg"))/2
                    dist_man_avg = np.add(metrics_1.distance(d_type="man", criteria="avg"),
                                    metrics_2.distance(d_type="man", criteria="avg"))/2
                    dist_euc_avg = np.add(metrics_1.distance(d_type="euc", criteria="avg"),
                                    metrics_2.distance(d_type="euc", criteria="avg"))/2
                elif method == 'min':
                    dist_cheb_avg = min(metrics_1.distance(d_type="cheb", criteria="avg"),
                                     metrics_2.distance(d_type="cheb", criteria="avg"))
                    dist_man_avg = min(metrics_1.distance(d_type="man", criteria="avg"),
                                    metrics_2.distance(d_type="man", criteria="avg"))
                    dist_euc_avg = min(metrics_1.distance(d_type="euc", criteria="avg"),
                                    metrics_2.distance(d_type="euc", criteria="avg"))
                elif method == "max":
                    dist_cheb_avg = max(metrics_1.distance(d_type="cheb", criteria="avg"),
                                     metrics_2.distance(d_type="cheb", criteria="avg"))
                    dist_man_avg = max(metrics_1.distance(d_type="man", criteria="avg"),
                                    metrics_2.distance(d_type="man", criteria="avg"))
                    dist_euc_avg = max(metrics_1.distance(d_type="euc", criteria="avg"),
                                    metrics_2.distance(d_type="euc", criteria="avg"))
                else:
                    return False
                cheb_all[idx] = dist_cheb_avg
                man_all[idx] = dist_man_avg
                euc_all[idx] = dist_euc_avg

            min_cheb_other = np.max(cheb_all[other_best_idx])
            min_man_other  = np.max(man_all[other_best_idx])
            min_euc_other  = np.max(euc_all[other_best_idx])

            min_cheb = np.max(cheb_all[idx_array])
            min_man = np.max(man_all[idx_array])
            min_euc = np.max(euc_all[idx_array])

            print("\nPredicted : {} out of actual: {}".format(checker_flag, true_flag))
            print("Other: Cheb: {:3f} Man: {:3f} Euc: {:3f}".format(min_cheb_other, min_man_other, min_euc_other))
            print("Our:   Cheb: {:3f} Man: {:3f} Euc: {:3f}".format(min_cheb, min_man, min_euc))
            print()
            if min_cheb < min_cheb_other or min_man < min_man_other or min_euc < min_euc_other:
                return True
            else:
                return False

with open(r"C:\Users\user\Documents\Research\FeatureCorrespondenes\config\config.json", 'r') as f:
    CONFIG = json.load(f)["config"]


ranges = {
    "fountain":[0,19],
    "herjzesu":[19,32],
    "entry":[32, 49],
    "castle":[49, 81]
}

full_dataset = {
    "fountain": 0,
    "herjzesu": 0,
    "entry": 0,
    "castle": 0
}

all_counters = {}
overall = {}
methods = ["avg","min", "max"]

selected = "castle"


r = ranges[selected]
for method in methods:
    checker = 0
    for i in range(r[0], r[1]):
        pair_path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset_2\main\pair_{}'.format(str(i))
        sol_path = r"C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset_2\main\stereo_heuristic_data\all\pair_{}.csv".format(str(i))

        if not os.path.exists(sol_path):
            continue
        stereo = Stereo(
            path = pair_path,
            n_components = int(CONFIG["SIFTFeatures"]),
            plot_ground_truth=False,
            show_imgs = False,
            n_sols=100
        )
        full_dataset[selected]+=1
        stereo.compute_ground_truth()
        print("PAIR #{}".format(str(i)))
        res = stereo.full_evaluation(sol_path,
                                     checkpoint_path=r"C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\combined_models\fountain_herjzesu_entry.joblib",
                                     method = method)
        if res:
            checker+=1
        else:
            print("Not predicted by the model or results higher than others\n")
    print("method is {}\n============================================".format(method))
    print(checker)
    all_counters[method] = checker

print("full result is ", all_counters)
print("full data is", int(full_dataset[selected]/3))