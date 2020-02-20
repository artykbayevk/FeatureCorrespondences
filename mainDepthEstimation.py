import os
import glob
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scripts.Triangulation.Depth import Triangulation
from scripts.Triangulation.HausdorffDist import Hausdorff
from scripts.email import send_email

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

    def compute_LP(self):
        process = subprocess.Popen(
            ['julia', r'C:\Users\user\Documents\Research\FeatureCorrespondenes\scripts\DepthEstimation.jl',
             self.img1_path,
             self.img2_path,
             self.main_path,
             self.limit_solutions],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        # print(stdout, stderr)
        self.opt_solutions = len(glob.glob(os.path.join(self.exp_dir, "*.csv")))


    def julia_method(self, run_julia=True):
        if run_julia:
            self.compute_LP()

        for f_i in range(1, self.opt_solutions+1):
            julia= Triangulation(K=self.K, R1=self.R1, R2=self.R2, T1=self.T1, T2=self.T2)
            julia.load_imgs(self.img1_path, self.img2_path)
            julia.findRootSIFTFeatures(n_components=self.n_components)
            f_path = os.path.join(self.exp_dir, 'matchedPoints_' + str(f_i) + '.csv')
            julia.matchingRootSIFTFeatures(f_path, True)
            julia.findRTmatrices()
            julia.point_cloud(plot=self.draw_plot, title="Our method #{}".format(f_i))
            pred = julia.pts3D
            metrics = Hausdorff(u=pred, v=self.target)
            dist_cheb_avg = metrics.distance(d_type="cheb", criteria="avg")
            dist_cheb_max = metrics.distance(d_type="cheb", criteria="max")
            dist_man_avg = metrics.distance(d_type="man", criteria="avg")
            dist_man_max = metrics.distance(d_type="man", criteria="max")
            dist_euc_avg = metrics.distance(d_type="euc", criteria="avg")
            dist_euc_max = metrics.distance(d_type="euc", criteria="max")
            print(
                "\t\t#{}: \t\t {:5f} {:5f} {:5f} {:5f} {:5f} {:5f}".format(f_i, dist_cheb_avg, dist_cheb_max,
                                                                           dist_man_avg,
                                                                           dist_man_max, dist_euc_avg, dist_euc_max))

with open(r"C:\Users\user\Documents\Research\FeatureCorrespondenes\config\config.json", 'r') as f:
    CONFIG = json.load(f)["config"]

for i in range(0,32):
    stereo = Stereo(
        path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset_2\main\pair_{}'.format(str(i)),
        n_components = int(CONFIG["SIFTFeatures"]),
        plot_ground_truth=False,
        show_imgs = False,
        n_sols=100
    )

    stereo.compute_ground_truth()
    # stereo.julia_method(run_julia = True)
    stereo.compute_LP()
    print("Pair: {} finished".format(str(i)))




# send_email(
#     user="crm.kamalkhan@gmail.com",
#     pwd="Astana2019",
#     recipient="kamalkhan.artykbayev@nu.edu.kz",
#     subject="Deep Learning Model",
#     body="Its ready"
# )
