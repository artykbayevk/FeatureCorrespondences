import cv2
import numpy as np
import os

class InnerFeatures:
    def __init__(self, kps, des, pos):
        self.kps = kps
        self.des = des
        self.pos = pos

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

class SceneReconstruction3D:
    def __init__(self, K, d):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.d = d

    def loadImgs(self, img1_path, img2_path, scale = 10):

        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        width = int(img1.shape[1] * scale / 100)
        height = int(img1.shape[0] * scale / 100)
        dim = (width, height)
        self.img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)
        width = int(img2.shape[1] * scale / 100)
        height = int(img2.shape[0] * scale / 100)
        dim = (width, height)
        self.img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    def rootSIFT(self):
        def innerRootSIFT(img):
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, descs) = sift.detectAndCompute(img, None)

            rs = RootSIFT()
            (kps, descs) = rs.compute(img, kps)
            pos = [np.array([x.pt[0], x.pt[1]]) for x in kps]

            return kps, descs, pos
        kps1, desc1, pos1 = innerRootSIFT(self.img1)
        kps2, desc2, pos2 = innerRootSIFT(self.img2)
        self.feature_1 = InnerFeatures(kps1, desc1, pos1)
        self.feature_2 = InnerFeatures(kps2, desc2, pos2)

    def matcher(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.feature_1.des, self.feature_2.des, k=2)

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(self.feature_2.kps[m.trainIdx].pt)
                pts1.append(self.feature_1.kps[m.queryIdx].pt)

        self.match_pts1 = np.int32(pts1)
        self.match_pts2 = np.int32(pts2)

    def LPMatcher(self, pts1, pts2, correspondences):
        print(pts1.shape, pts2.shape, correspondences.shape)



K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4,
                   1006.81/4, 0, 0, 1]]).reshape(3, 3)
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)


scene = SceneReconstruction3D(K,d)
BASE = os.getcwd()
img1_path = os.path.join(BASE, "data","test","001_L.png")
img2_path = os.path.join(BASE, "data","test","001_R.png")
scene.loadImgs(img1_path, img2_path)


#%%
scene.rootSIFT()
scene.matcher()
print(scene.match_pts1[0], scene.match_pts2[0])
