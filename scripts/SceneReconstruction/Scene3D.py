#%% Importing all that shit

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os

class SceneReconstruction3D:

    def __init__(self,K, dist):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.d = dist

    def load_image_pair(self, img_path1, img_path2, use_pyr_down=True):
        self.img1 = cv2.imread(img_path1, cv2.CV_8UC3)
        self.img2 = cv2.imread(img_path2, cv2.CV_8UC3)

        if self.img1 is None:
            sys.exit("Image " + img_path1 + " could not be loaded.")
        if self.img2 is None:
            sys.exit("Image " + img_path2 + " could not be loaded.")

        if len(self.img1.shape) == 2:
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)

        target_width = 200
        if use_pyr_down and self.img1.shape[1] > target_width:
            while self.img1.shape[1] > 2 * target_width:
                self.img1 = cv2.pyrDown(self.img1)
                self.img2 = cv2.pyrDown(self.img2)

        self.img1 = cv2.undistort(self.img1, self.K, self.d)
        self.img2 = cv2.undistort(self.img2, self.K, self.d)

        base = "/".join(img_path1.split("\\")[:-1])
        cv2.imwrite(os.path.join(base, "left_loaded.png"), self.img1)
        cv2.imwrite(os.path.join(base, "right_loaded.png"), self.img2)

    def plot_rectified_images(self):
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]
        # perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K, self.d,
                                                          self.K, self.d,
                                                          self.img1.shape[:2],
                                                          R, T, alpha=1.0)
        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.K, self.d, R1, self.K,
                                                   self.img1.shape[:2],
                                                   cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.K, self.d, R2, self.K,
                                                   self.img2.shape[:2],
                                                   cv2.CV_32F)
        img_rect1 = cv2.remap(self.img1, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.img2, mapx2, mapy2, cv2.INTER_LINEAR)

        # draw the images side by side
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

        cv2.imshow('imgRectified', img)
        cv2.waitKey(5000)

    def plot_point_cloud(self):
        """Plots 3D point cloud
            This method generates and plots a 3D point cloud of the recovered
            3D scene.
            :param feat_mode: whether to use rich descriptors for feature
                              matching ("surf") or optic flow ("flow")
        """
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        # triangulate points
        first_inliers = np.array(self.match_inliers1).reshape(-1, 3)[:, :2]
        second_inliers = np.array(self.match_inliers2).reshape(-1, 3)[:, :2]
        pts4D = cv2.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T,
                                      second_inliers.T).T

        # convert from homogeneous coordinates to 3D
        pts3D = pts4D[:, :3] / np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

        # plot with matplotlib
        Ys = pts3D[:, 0]
        Zs = pts3D[:, 1]
        Xs = pts3D[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c='r', marker='o')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_zlabel('X')
        plt.title('3D point cloud: Use pan axes button below to inspect')
        plt.show()

    def findRootSIFTFeatures(self):

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

    def _find_fundamental_matrix(self):
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)

    def _find_essential_matrix(self):
        self.E = self.K.T.dot(self.F).dot(self.K)

    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([self.match_pts1[i][0],
                                                     self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.K_inv.dot([self.match_pts2[i][0],
                                                      self.match_pts2[i][1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]

        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]

            if not self._in_front_of_both_cameras(first_inliers,
                                                  second_inliers, R, T):
                # Fourth choice: R = U * Wt * Vt, T = -u_3
                T = - U[:, 2]

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def _in_front_of_both_cameras(self, first_points, second_points, rot,trans):
        """Determines whether point correspondences are in front of both
           images"""
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :],trans) / np.dot(rot[0, :] - second[0] * rot[2, :],second)
            first_3d_point = np.array([first[0] * first_z,second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True