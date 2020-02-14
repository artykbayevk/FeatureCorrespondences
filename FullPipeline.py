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

    def full_evaluation(self, opt_solution_path, checkpoint_path):
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
        for idx, item in enumerate(Y_):
            if Y[idx] == Y_[idx] and item == 1.0:
                checker_flag+=1
                idx_array.append(idx)
            if Y[idx] == 1.0 and Y[idx] != Y_[idx]:
                other_best_idx.append(idx)

            julia = Triangulation(K=self.K, R1=self.R1, R2=self.R2, T1=self.T1, T2=self.T2)
            julia.load_imgs(self.img1_path, self.img2_path)
            julia.findRootSIFTFeatures(n_components=self.n_components)
            julia.matchingRootSIFTFeatures_advanced(X[idx])
            julia.findRTmatrices()
            julia.point_cloud(plot=self.draw_plot, title="Our method #{}".format(idx))
            pred = julia.pts3D
            metrics = Hausdorff(u=pred, v=self.target)
            dist_cheb_avg = metrics.distance(d_type="cheb", criteria="avg")
            dist_man_avg = metrics.distance(d_type="man", criteria="avg")
            dist_euc_avg = metrics.distance(d_type="euc", criteria="avg")
            cheb_all[idx] = dist_cheb_avg
            man_all[idx] = dist_man_avg
            euc_all[idx] = dist_euc_avg
            print(
                "\t\t#{}: \t\t {:5f} {:5f} {:5f} REAL: {} PRED: {}".format(idx, dist_cheb_avg,
                                                                           dist_man_avg,
                                                                           dist_euc_avg,
                                                                            Y[idx], Y_[idx]))
        max_cheb = np.max(cheb_all)
        min_cheb = np.min(cheb_all)
        max_man  = np.max(man_all)
        min_man  = np.min(man_all)
        max_euc  = np.max(euc_all)
        min_euc  = np.min(euc_all)

        selected_cheb = cheb_all[idx_array]
        selected_man = man_all[idx_array]
        selected_euc = euc_all[idx_array]

        print("All data\nMAX:{:3f} {:3f} {:3f}\nMIN:{:3f} {:3f} {:3f}".format(
            max_cheb, max_euc, max_man,min_cheb, min_euc, min_man
        ))
        max_cheb = np.max(selected_cheb)
        min_cheb = np.min(selected_cheb)
        max_man = np.max(selected_man)
        min_man = np.min(selected_man)
        max_euc = np.max(selected_euc)
        min_euc = np.min(selected_euc)

        print("\nPredicted : {} out of actual: {}".format(checker_flag, true_flag))
        print("Selected data\nMAX:{:3f} {:3f} {:3f}\nMIN:{:3f} {:3f} {:3f}\n".format(
            max_cheb, max_euc, max_man,min_cheb, min_euc, min_man
        ))

        if len(other_best_idx) != 0:

            selected_cheb = cheb_all[other_best_idx]
            selected_man = man_all[other_best_idx]
            selected_euc = euc_all[other_best_idx]

            max_cheb = np.max(selected_cheb)
            min_cheb = np.min(selected_cheb)
            max_man = np.max(selected_man)
            min_man = np.min(selected_man)
            max_euc = np.max(selected_euc)
            min_euc = np.min(selected_euc)

            print("Not selected data\nMAX:{:3f} {:3f} {:3f}\nMIN:{:3f} {:3f} {:3f}".format(
                max_cheb, max_euc, max_man,min_cheb, min_euc, min_man
            ))

with open(r"C:\Users\user\Documents\Research\FeatureCorrespondenes\config\config.json", 'r') as f:
    CONFIG = json.load(f)["config"]
stereo = Stereo(
    path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\pair_3',
    n_components = int(CONFIG["SIFTFeatures"]),
    plot_ground_truth=False,
    show_imgs = False,
    n_sols=100
)

stereo.compute_ground_truth()
stereo.full_evaluation(r"C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_3.csv",
                       checkpoint_path=r"C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\keras_model.joblib")

#TODO evaluate full process.
#TODO show the real difference between choosing best and not best optimal solution
#TODO compare minimum of selected best optimal solution with non-selected best optimal solutions/other solutions