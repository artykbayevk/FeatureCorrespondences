#%%
import cv2
import numpy as np

global base_path
base_path = "data\\test\\"


img1_path = "data/test/test_1.jpg"
img2_path = "data/test/test_2.jpg"


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


def rootSIFT(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    rs = RootSIFT()
    (kps, descs) = rs.compute(gray, kps)
    img = cv2.drawKeypoints(gray, kps, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(base_path+'test_kp_'+img_path.split("_")[1], img)

    pos = [np.array([x.pt[0], x.pt[1]]) for x in kps]

    return np.array(pos)
    #TODO delete unique samples from result

    #TODO copy and past this code to the julia
