import os
import sys
import numpy as np
import pandas as pd

BASE = os.getcwd()
DATASET = 'fountain'
DATA_BASE = os.path.join(BASE, 'data','origin', DATASET)
OUT_DATA = os.path.join(BASE, 'data', 'dataset_2', DATASET)
main_dataset_path = os.path.join(DATA_BASE, 'dataset.txt')

def read_dataset_txt(path):
    data = pd.read_csv(path, sep=" ", header=None)
    data.columns = ["left", "right"]
    return data[::-1]

def read_from_camera_file(path):
    doc_path = path+'.camera'
    with open(doc_path, "r", encoding="utf8") as f:
        content = f.readlines()
    del content[3]
    del content[-1]

    I = np.array([[float(y) for y in x.split('\n')[0].strip().split(' ')] for x in content[:3]])
    E = np.array([[float(y) for y in x.split('\n')[0].strip().split(' ')] for x in content[3:]]).T


    stop = 1
connections = read_dataset_txt(main_dataset_path)

for idx, pair in connections.iterrows():
    pair_path = os.path.join(OUT_DATA, "pair_"+str(idx))
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)

    left_path = os.path.join(DATA_BASE, pair["left"])
    right_path = os.path.join(DATA_BASE, pair["right"])

    print(left_path, right_path)
    left_camera = read_from_camera_file(left_path)
    right_camera = read_from_camera_file(right_path)
    stop = 1