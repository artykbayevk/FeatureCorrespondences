#%%
import numpy as np
import cv2
import os
#%%
BASE = os.getcwd()
img1_path = os.path.join(BASE, "data","test","001_L.png")
img2_path = os.path.join(BASE, "data","test","001_R.png")

#%%
class RootSIFT:
    def __init__(self):
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        (kps, descs) = self.extractor.compute(image, kps)
        if len(kps) == 0:
            return ([], None)

        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return (kps, descs)

#%%
def rootSIFT(img_path, resize = False, n_kp = 100):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if resize:
        scale_percent = 10    # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    if n_kp == 0:
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SIFT_create(n_kp)

    (kps, descs) = sift.detectAndCompute(gray, None)
    rs = RootSIFT()
    (kps, descs) = rs.compute(gray, kps)

    pos = [np.array([x.pt[0], x.pt[1]]) for x in kps]

    # return np.array(pos)
    return kps, descs


#%%
kp1, desc1 = rootSIFT(img1_path,n_kp=0)
kp2, desc2 = rootSIFT(img1_path,n_kp=0)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desc1,desc2,k=2)

#%%
good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

#%%

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

pts1.shape
pts2.shape




#%%
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

#%%
F
