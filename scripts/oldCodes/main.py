#%%
import cv2
import numpy as np
import scripts.Scene3D.SceneClass
import scripts.SceneReconstruction.Scene3D
global base_path
base_path = "data\\test\\"


img1_path = "data/test/001_L.png"
img2_path = "data/test/001_R.png"


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
    img = cv2.drawKeypoints(gray, kps, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(base_path+'test_kp_'+img_path.split("_")[1], img)

    pos = [np.array([x.pt[0], x.pt[1]]) for x in kps]

    return np.array(pos)

res = rootSIFT(img1_path, n_kp=0)
res = rootSIFT(img2_path, n_kp=0)
res.shape
