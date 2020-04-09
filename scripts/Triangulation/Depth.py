#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
import sys
import cv2
import os

class Triangulation:
    def __init__(self, K, R1, R2, T1, T2):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.R1 = R1
        self.T1 = T1

        self.R2 = R2
        self.T2 = T2

    def load_imgs(self, path1, path2):
        self.img1 = cv2.imread(path1, cv2.CV_8UC3)
        self.img2 = cv2.imread(path2, cv2.CV_8UC3)

        base = "/".join(path1.split("\\")[:-1])
        # cv2.imwrite(os.path.join(base, "left_loaded.png"), self.img1)
        # cv2.imwrite(os.path.join(base, "right_loaded.png"), self.img2)

    def findK_centroids_closest(self, features, clusters):
        """
        This function just produce closest points for centroids

        :param features:
        :param clusters:
        :return:
        """
        class InnerFeatures:
            def __init__(self, kps, des, pos):
                self.kps = kps
                self.des = des
                self.pos = pos

        kmeans = KMeans(n_clusters=clusters)

        pts = np.array(features.pos)
        kps = np.array(features.kps)
        des = np.array(features.des)

        kmeans.fit(pts)
        m_clusters = kmeans.labels_.tolist()
        centers = np.array(kmeans.cluster_centers_)

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, pts)

        assert len(set(closest)) == clusters

        result = InnerFeatures(kps[closest], des[closest], pts[closest])
        return result

    def findK_centroids_average(self, features, clusters):

        """
        produce centroids with their kps and descriptions

        :param features:
        :param clusters:
        :return:
        """
        class InnerFeatures:
            def __init__(self, kps, des, pos):
                self.kps = kps
                self.des = des
                self.pos = pos
        kmeans = KMeans(n_clusters=clusters)

        pts = np.array(features.pos)
        kps = np.array(features.kps)
        des = np.array(features.des)

        kmeans.fit(pts)
        m_clusters = np.array(kmeans.labels_.tolist())
        centers = np.array(kmeans.cluster_centers_)

        # KeyPoint(x,y,size) -required

        final_kps = []
        final_des = []
        final_pts = []

        for cluster in range(clusters):
            indices = np.where(m_clusters == cluster)
            cluster_kps_size = np.mean(np.array([x.size for x in kps[indices]]))
            cluster_des = des[indices]

            average_des = np.mean(cluster_des, axis=0)
            cluster_kps = cv2.KeyPoint(x=centers[cluster][0], y=centers[cluster][1], _size=cluster_kps_size)

            final_kps.append(cluster_kps)
            final_des.append(average_des)
            final_pts.append([centers[cluster][0], centers[cluster][1]])

        final_pts = np.array(final_pts)
        final_des = np.array(final_des)
        final_kps = np.array(final_kps)

        result = InnerFeatures(kps = final_kps, des=final_des, pos = final_pts)
        return result


    def findRootSIFTFeatures(self, n_components = None):

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

        class InnerFeatures:
            def __init__(self, kps, des, pos):
                self.kps = kps
                self.des = des
                self.pos = pos

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

        ## next step is finding 100 CENTROIDS,
        self.feature_1 = self.findK_centroids_average(self.feature_1, n_components)
        self.feature_2 = self.findK_centroids_average(self.feature_2, n_components)


    def matchingRootSIFTFeatures_advanced(self,solution):
        solution = solution.reshape(-1,4)
        PX = solution[:,0].reshape(-1, 1)
        PY = solution[:,1].reshape(-1, 1)

        QX = solution[:,2].reshape(-1, 1)
        QY = solution[:,3].reshape(-1, 1)

        matched_pts1 = np.float32(np.concatenate((PX, PY), axis=1))
        matched_pts2 = np.float32(np.concatenate((QX, QY), axis=1))

        self.match_pts1 = matched_pts1
        self.match_pts2 = matched_pts2


    def matchingRootSIFTFeatures(self, pathToCsv=None ,fromJulia=False):
        if fromJulia:
            data_path = os.path.join(pathToCsv)
            matchedPointsDF = pd.read_csv(data_path, sep=",", header=None)

            PX = matchedPointsDF[0].values.reshape(-1, 1)
            PY = matchedPointsDF[1].values.reshape(-1, 1)

            QX = matchedPointsDF[2].values.reshape(-1, 1)
            QY = matchedPointsDF[3].values.reshape(-1, 1)

            matched_pts1 = np.float32(np.concatenate((PX, PY), axis=1))
            matched_pts2 = np.float32(np.concatenate((QX, QY), axis=1))

            self.match_pts1 = matched_pts1
            self.match_pts2 = matched_pts2

        else:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(self.feature_1.des, self.feature_2.des, k=2)

            good = []
            pts1 = []
            pts2 = []

            for i, (m, n) in enumerate(matches):
                if m.distance < 0.80 * n.distance:
                    good.append(m)
                    pts2.append(self.feature_2.kps[m.trainIdx].pt)
                    pts1.append(self.feature_1.kps[m.queryIdx].pt)

            self.match_pts1 = np.round(pts1)
            self.match_pts2 = np.round(pts2)
            self.matches = good
            print("Opencv Matched found :{} feature correspondences".format(len(self.matches)))

    def drawMathces(self, path):
        OutImage = cv2.drawMatches(self.img1, self.feature_1.kps, self.img2, self.feature_2.kps, self.matches,outImg=None)
        cv2.imwrite(path,OutImage)

    def findRTmatrices(self):

        pts1 = self.match_pts1
        pts2 = self.match_pts2
        left_points = np.zeros((pts1.shape[0],3))
        right_points = np.zeros((pts2.shape[0],3))

        R = np.dot(self.R2,self.R1)
        T = - np.dot(self.R2, self.R1).dot(self.T1) - self.T2

        # R = np.multiply(self.R2, self.R1.T)
        # T = -np.dot(np.multiply(self.R2, self.R1.T), self.T1) + self.T2
        stop = 1

        for i in range(self.match_pts1.shape[0]):
            norm1 = self.K_inv.dot([self.match_pts1[i][0],self.match_pts1[i][1], 1.0])
            norm2 = self.K_inv.dot([self.match_pts2[i][0],self.match_pts2[i][1], 1.0])

            left_points[i,:] = norm1
            right_points[i,:] = norm2

        self.norm_1 = left_points
        self.norm_2 = right_points

        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def point_cloud(self, title, plot = True):
        first_inliers = np.array(self.norm_1).reshape(-1, 3)[:, :2]
        second_inliers = np.array(self.norm_2).reshape(-1, 3)[:, :2]
        pts4D = cv2.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T,
                                      second_inliers.T).T
        pts4D = cv2.triangulatePoints(self.K.dot(self.Rt1), self.K.dot(self.Rt2), first_inliers.T,
                                      second_inliers.T).T
        pts3D = pts4D[:, :3] / np.repeat(pts4D[:, 3], 3).reshape(-1, 3)
        Ys = pts3D[:, 0]
        Zs = pts3D[:, 1]
        Xs = pts3D[:, 2]

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Xs, Ys, Zs, c='r', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('3D point cloud: {}'.format(title))
            plt.show()

        self.pts3D = pts3D