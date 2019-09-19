#%%

import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os

class Triangulation:
    def __init__(self, fc_left, cc_left, fc_right, cc_right, R ,T):
        self.fc_left = fc_left
        self.cc_left = cc_left
        self.fc_right = fc_right
        self.cc_right = cc_right
        self.R = R
        self.T = T

    def load_imgs(self, path1, path2):
        self.img1 = cv2.imread(path1, cv2.CV_8UC3)
        self.img2 = cv2.imread(path2, cv2.CV_8UC3)

        base = "/".join(path1.split("\\")[:-1])
        cv2.imwrite(os.path.join(base, "left_loaded.png"), self.img1)
        cv2.imwrite(os.path.join(base, "right_loaded.png"), self.img2)

    def findRootSIFTFeatures(self, n_components = None):

        class RootSIFT:
            def __init__(self):
                self.extractor = cv2.xfeatures2d.SIFT_create(n_components) if n_components != None else cv2.xfeatures2d.SIFT_create()
            def compute(self, image, kps, eps=1e-7):
                (kps, descs) = self.extractor.compute(image, kps)
                if len(kps) == 0:
                    return ([], None)

                descs /= (descs.sum(axis=1, keepdims=True) + eps)
                descs = np.sqrt(descs)
                return (kps, descs)

        class InnerFeatures:
            def __init__(self, kps, des, pos):
                self.kps = kps
                self.des = des
                self.pos = pos

        def innerRootSIFT(img):
            sift = cv2.xfeatures2d.SIFT_create(n_components) if n_components != None else cv2.xfeatures2d.SIFT_create()
            (kps, descs) = sift.detectAndCompute(img, None)

            rs = RootSIFT()
            (kps, descs) = rs.compute(img, kps)
            pos = [np.array([x.pt[0], x.pt[1]]) for x in kps]

            return kps, descs, pos

        kps1, desc1, pos1 = innerRootSIFT(self.img1)
        kps2, desc2, pos2 = innerRootSIFT(self.img2)
        self.feature_1 = InnerFeatures(kps1, desc1, pos1)
        self.feature_2 = InnerFeatures(kps2, desc2, pos2)

    def matchingRootSIFTFeatures(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.feature_1.des, self.feature_2.des, k=2)

        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(self.feature_2.kps[m.trainIdx].pt)
                pts1.append(self.feature_1.kps[m.queryIdx].pt)

        self.match_pts1 = np.round(pts1)
        self.match_pts2 = np.round(pts2)
        self.matches = good

    def drawMathces(self, path):
        OutImage = cv2.drawMatches(self.img1, self.feature_1.kps, self.img2, self.feature_2.kps, self.matches,outImg=None)
        cv2.imwrite(path,OutImage)
