# %% importing libraries
import os
import cv2
import pandas as pd
import numpy as np
from scripts.SceneReconstruction.Scene3D import SceneReconstruction3D

# %%
K = np.array([[2759.48 / 4, 0, 1520.69 / 4, 0, 2764.16 / 4,
               1006.81 / 4, 0, 0, 1]]).reshape(3, 3)
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
scene = SceneReconstruction3D(K, d)

# %%
BASE = os.getcwd()
img1_path = os.path.join(BASE, "data", "pair", "Left.png")
img2_path = os.path.join(BASE, "data", "pair", "Right.png")
scene.load_image_pair(img1_path, img2_path)

#%%
scene.findRootSIFTFeatures()
scene.matchingRootSIFTFeatures()
a = scene.matches[10][0]
b = cv2.DMatch()

# %%
data_path = os.path.join(BASE, "data", "pair", "matchedPoints.csv")
matchedPointsDF = pd.read_csv(data_path, sep=",", header=None)

PX = matchedPointsDF[0].values.reshape(-1, 1)
PY = matchedPointsDF[1].values.reshape(-1, 1)

QX = matchedPointsDF[2].values.reshape(-1, 1)
QY = matchedPointsDF[3].values.reshape(-1, 1)

matched_pts1 = np.float32(np.concatenate((PX, PY), axis=1))
matched_pts2 = np.float32(np.concatenate((QX, QY), axis=1))

scene.match_pts1 = matched_pts1
scene.match_pts2 = matched_pts2

# %%
scene.findRootSIFTFeatures()
scene.matchingRootSIFTFeatures()

# %%
scene.plot_point_cloud()

# %%
scene.plot_rectified_images()

#%%
pts1 = scene.match_pts1
pts2 = scene.match_pts2

df = pd.DataFrame(pts1)
df[2] = pts2[:,0]
df[3] = pts2[:,1]
df.head()
df.to_csv(os.path.join(BASE, "data", "pair", "matchedCVdata.csv"), header=None, index=None)

#%%
df1 = matchedPointsDF.values
df2 = df.values

#%%
cnt = 0
for i in range(df1.shape[0]):
    for j in range(df2.shape[0]):
        if(np.array_equal(np.float32(df1[i]), np.float32(df2[j])) == True):
            cnt+=1
print(cnt)
