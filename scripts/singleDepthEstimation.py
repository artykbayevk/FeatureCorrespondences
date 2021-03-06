# %%
import json
import os

import numpy as np

from scripts.Triangulation.Depth import Triangulation
from scripts.Triangulation.HausdorffDist import Hausdorff

K = np.array([
    [919.8266666666666, 0.0, 506.89666666666665],
    [0.0, 921.8365624999999, 335.7672021484375],
    [0.0, 0.0, 1.0]
])

R1 = np.array([
    [0.450927, -0.0945642, -0.887537],
    [-0.892535, -0.0401974, -0.449183],
    [0.00679989, 0.994707, -0.102528]])
T1 = np.array([-7.28137, -7.57667, 0.204446])

R2 = np.array([
    [0.582226, -0.0983866, -0.807052],
    [-0.813027, -0.0706383, -0.577925],
    [-0.000148752, 0.992638, -0.121118]
])
T2 = np.array([-8.31326, -6.3181, 0.16107])

with open("/Users/artkvk/Documents/RA/FeatureCorrespondences/config/config.json", 'r') as f:
    CONFIG = json.load(f)["config"]
n_components = int(CONFIG["SIFTFeatures"])
plot = True if CONFIG['plot'] == 'true' else False

BASE = os.getcwd()
img1_path = os.path.join('/Users/artkvk/Documents/RA/FeatureCorrespondences/data/dense/0000-small-left.png')
img2_path = os.path.join('/Users/artkvk/Documents/RA/FeatureCorrespondences/data/dense/0001-small-right.png')

#  OPEN-CV METHOD
opencv = Triangulation(K=K, R1=R1, R2=R2, T1=T1, T2=T2)
opencv.load_imgs(img1_path, img2_path)
opencv.findRootSIFTFeatures(n_components=n_components)
opencv.matchingRootSIFTFeatures()
opencv.findRTmatrices()
opencv.point_cloud(plot=plot, title="OpenCV")
true = opencv.pts3D

#  JULIA METHOD
# path = '/Users/artkvk/Documents/RA/FeatureCorrespondences/data/dense/experiment'
path = "/Users/artkvk/Documents/RA/FeatureCorrespondences/data/dense/experiment"
print("# Optimal solution \t Cheb_avg Cheb_max Manh_avg Mang_max Eucl_avg Eucl_max")
for f_i in range(1, 7):
    julia = Triangulation(K=K, R1=R1, R2=R2, T1=T1, T2=T2)
    julia.load_imgs(img1_path, img2_path)
    julia.findRootSIFTFeatures(n_components=n_components)
    f_path = os.path.join(path, 'matchedPoints_' + str(f_i) + '.csv')
    julia.matchingRootSIFTFeatures(f_path, True)
    julia.findRTmatrices()
    julia.point_cloud(plot=plot, title="Our method #{}".format(f_i))
    pred = julia.pts3D
    metrics = Hausdorff(u=pred, v=true)
    dist_cheb_avg = metrics.distance(d_type="cheb", criteria="avg")
    dist_cheb_max = metrics.distance(d_type="cheb", criteria="max")
    dist_man_avg = metrics.distance(d_type="man", criteria="avg")
    dist_man_max = metrics.distance(d_type="man", criteria="max")
    dist_euc_avg = metrics.distance(d_type="euc", criteria="avg")
    dist_euc_max = metrics.distance(d_type="euc", criteria="max")
    print("\t\t#{}: \t\t {:5f} {:5f} {:5f} {:5f} {:5f} {:5f}".format(f_i, dist_cheb_avg, dist_cheb_max, dist_man_avg,
                                                                     dist_man_max, dist_euc_avg, dist_euc_max))
# %%
u = np.array([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
v = np.array([(2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -4.0)])

# TODO create dataset

# TODO rewrite DL model with heuristic method

# TODO show final results
