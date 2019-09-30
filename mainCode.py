#%%
import os
import numpy as np
from scripts.Triangulation.Depth import Triangulation
from scipy.spatial.distance import directed_hausdorff


K = np.array([
    [919.8266666666666,0.0, 506.89666666666665],
    [0.0,921.8365624999999,335.7672021484375],
    [0.0,0.0,1.0 ]
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



#%%

BASE = os.getcwd()
img1_path = os.path.join(BASE, "data", "dense", "0000-small-left.png")
img2_path = os.path.join(BASE, "data", "dense", "0001-small-right.png")


### OPENCV METHOD
opencv = Triangulation(K = K, R1=R1, R2=R2, T1 = T1, T2 = T2)
opencv.load_imgs(img1_path, img2_path)
opencv.findRootSIFTFeatures(n_components=400)
opencv.matchingRootSIFTFeatures()
opencv.findRTmatrices()
opencv.point_cloud(plot = True)
true = opencv.pts3D

### JULIA METHOD
julia = Triangulation(K = K, R1=R1, R2=R2, T1 = T1, T2 = T2)
julia.load_imgs(img1_path, img2_path)
julia.findRootSIFTFeatures()

# path = "/Users/kamalsdu/Documents/Research/FeatureCorrespondences/data/dense/matchedPoints.csv"
path = r"C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dense\matchedPoints.csv"
julia.matchingRootSIFTFeatures(path, True)
julia.findRTmatrices()
julia.point_cloud(plot = True)
pred = julia.pts3D


#%%
distance = directed_hausdorff(pred, true)[0]
print(distance)
