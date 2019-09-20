# %% importing libraries
import os

import numpy as np

from scripts.Triangulation.Depth import Triangulation

# %% intrinsic and extrinsic parameters of camera
fc_left = np.array([919.8266666666666, 921.8365624999999])
cc_left = np.array([506.89666666666665, 335.7672021484375])

fc_right = np.array([919.8266666666666, 921.8365624999999])
cc_right = np.array([506.89666666666665, 335.7672021484375])

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

R = R1 - R2
T = T1 - T2

# %%
BASE = os.getcwd()
img1_path = os.path.join(BASE, "data", "dense", "0000-small-left.png")
img2_path = os.path.join(BASE, "data", "dense", "0001-small-right.png")

estimator = Triangulation(fc_left, cc_left, fc_right, cc_right, R, T)
estimator.load_imgs(img1_path, img2_path)
estimator.findRootSIFTFeatures(200)
estimator.matchingRootSIFTFeatures()
estimator.drawMathces(os.path.join(BASE, "data", "dense", "matchesOpenCV.png"))

print("Number of SIFT features: {}".format(len(estimator.feature_1.kps)))
from scipy.spatial.distance import directed_hausdorff