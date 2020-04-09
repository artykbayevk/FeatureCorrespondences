using PyCall
using Distances
using StatsBase
using LinearAlgebra
using JuMP
using Gurobi
using CSV
using DataFrames
using SparseArrays
using Printf
using JSON

py"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import sys
import cv2
import os

class Triangulation:
    def load_imgs(self, path1, path2):
        self.img1 = cv2.imread(path1, cv2.CV_8UC3)
        self.img2 = cv2.imread(path2, cv2.CV_8UC3)
        # base = "/".join(path1.split("\\")[:-1])
        # cv2.imwrite(os.path.join(base, "left_loaded.png"), self.img1)
        # cv2.imwrite(os.path.join(base, "right_loaded.png"), self.img2)

    def findK_centroids(self, features, clusters):
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
                self.extractor =  cv2.xfeatures2d.SIFT_create()
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

        ## TOP K CENTROIDS
        self.feature_1 = self.findK_centroids_average(self.feature_1, n_components)
        self.feature_2 = self.findK_centroids_average(self.feature_2, n_components)

    def drawMatches(self, path):
        self.outImage = cv2.drawMatches(self.img1, self.feature_1.kps, self.img2, self.feature_2.kps, self.matches,outImg=None)
        cv2.imwrite(path, self.outImage)
scene = Triangulation()
"""
n_components = JSON.parse(String(read(ARGS[5])))["config"]["SIFTFeatures"]

pair = 9

left = 1
right = 0
# img1_path = "C:/Users/user/Documents/Research/FeatureCorrespondenes/data/dataset/pair_$(pair)/left_000$(left)-small.png"
# img2_path = "C:/Users/user/Documents/Research/FeatureCorrespondenes/data/dataset/pair_$(pair)/right_000$(right)-small.png"

img1_path = ARGS[1]
img2_path = ARGS[2]


py"scene.load_imgs"(img1_path, img2_path)
py"scene.findRootSIFTFeatures"(n_components=n_components)

pts1 = py"list(scene.feature_1.pos)"
pts2 = py"list(scene.feature_2.pos)";

P_points = hcat(pts1...)';
Q_points = hcat(pts2...)';

# println("size P points", size(P_points))
# println("size Q points", size(Q_points))

cost = pairwise(Euclidean(), P_points, Q_points; dims=1);
# println(size(cost))
P = ones(size(P_points,1));
Q = ones(size(Q_points,1));

solCount = parse(Int32,ARGS[4])
# m = JuMP.direct_model(Gurobi.Optimizer(PoolSearchMode=2, PoolSolutions=solCount, SolutionNumber=0,PoolGap = 0.001))
m = JuMP.direct_model(Gurobi.Optimizer(PoolSearchMode=2, PoolSolutions=solCount, SolutionNumber=0));

@variable(m, X[axes(cost,1), axes(cost,2)] ≥ 0, Int);
@objective(m, Min, cost ⋅ X);
@constraint(m,sum(X) .== min(sum(P), sum(Q)));
@constraint(m, X * ones(Int, length(Q)) .<= P);
@constraint(m, X'ones(Int, length(P)) .<= Q);
optimize!(m);
solution_pool = zeros(solCount, length(P),length(Q))
obj = objective_value(m)
cnt = 0
for i in 0:(solCount-1)
    global cnt
    setparam!(m.moi_backend.inner,"SolutionNumber", i)
    xn = Gurobi.get_dblattrarray(m.moi_backend.inner, "Xn", 1, length(X))
    xn_val = Gurobi.get_dblattr(m.moi_backend.inner, "PoolObjVal")
    #if(floor(xn_val) != floor(obj))
    #    if floor(xn_val) - floor(obj) == 1 || floor(xn_val) - floor(obj) == 2
    #        default = zeros(length(P),length(Q))
    #        for i in 0:length(P)-1
    #            default[i+1,:] = xn[(i*length(Q))+1:(i+1)*length(Q)]
    #        end
    #        solution_pool[i+1,:,:] = default
    #        cnt+=1
    #        continue
    #    end
    #    println(i , " solution(s) selected")
    #    println(xn_val, " current objective value")
    #    println(floor(xn_val)," ",floor(obj))
    #    break
    #end
    default = zeros(length(P),length(Q))
    for i in 0:length(P)-1
        default[i+1,:] = xn[(i*length(Q))+1:(i+1)*length(Q)]
    end
    solution_pool[i+1,:,:] = default
    cnt+=1
end

sol_pool = deepcopy(solution_pool[1:cnt,:,:]);
for n_sol in 1:cnt
    solOther = sparse(sol_pool[n_sol,:,:])
    experiment_path = string(ARGS[3], "/experiment/matchedPoints_",n_sol,".csv")
    # experiment_path = "C:/Users/user/Documents/Research/FeatureCorrespondenes/data/dataset/pair_$(pair)/experiment/matchedPoints_$(n_sol).csv"
    sizeOf = min(size(P,1), size(Q,1))
    matched_pts1 = zeros(sizeOf,2)
    matched_pts2 = zeros(sizeOf,2)
    i = 1
    py"""
    arr = []
    """
    for (x,y,v) in zip(findnz(solOther)...)
        x_pos = [P_points'[:,x][1], Q_points'[:,y][1]]
        y_pos = [P_points'[:,x][2], Q_points'[:,y][2]]

        # dmatch creating
        queryId = x-1
        trainId = y-1
        distance = cost[x,y]
    #     if(distance <= 10)
        dmatch = py"cv2.DMatch($(queryId), $(trainId),$(distance))"
        py"arr.append"(dmatch)
        matched_pts1[i,:] = [floor(x_pos[1]) floor(y_pos[1])]
        matched_pts2[i,:] = [floor(x_pos[2]) floor(y_pos[2])]
        i+=1
    #     end
    end
    py"""
    scene.matches = arr
    """


    # path = "../data\\pair\\lastLPMatched.png"
    # py"scene.drawMatches"(path)


    matched_final_1 = deepcopy(matched_pts1[1:i-1, :])
    matched_final_2 = deepcopy(matched_pts2[1:i-1, :]);
    df = DataFrame()
    df.PX = matched_final_1[:,1]
    df.PY = matched_final_1[:,2]
    df.QX = matched_final_2[:,1]
    df.QY = matched_final_2[:,2];
    # print(size(df))

    CSV.write(experiment_path,  df, writeheader=false)
end
