import os
import sys
import numpy as np
import pandas as pd
from shutil import copyfile

BASE = os.getcwd()
DATASET = 'castle'
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
    return I,E

def create_output_string(path, I, E):
    out = ""+path+'\n'
    for line in I:
        new_line = " ".join([str(x) for x in line])
        out+=new_line
        out+='\n'
    for line in E:
        new_line =" ".join([str(x) for x in line])
        out+=new_line
        out+='\n'
    print(out)
    return out

connections = read_dataset_txt(main_dataset_path)

for idx, pair in connections.iterrows():
    pair_path = os.path.join(OUT_DATA, "pair_"+str(idx))
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)

    # SOURCE OF IMAGES
    left_path = os.path.join(DATA_BASE, pair["left"])
    right_path = os.path.join(DATA_BASE, pair["right"])

    # READING THEIR CAMERA MATRICES
    L_I, L_E = read_from_camera_file(left_path)
    R_I, R_E = read_from_camera_file(right_path)

    # COPYING IMAGES FROM SOURCE TO DESTINATION
    copyfile(left_path, os.path.join(pair_path, "left_" + os.path.basename(left_path)))
    copyfile(right_path, os.path.join(pair_path, "right_" + os.path.basename(right_path)))

    # CREATING NEW DATASET.TXT FILE
    out_L = create_output_string(left_path, L_I, L_E)
    out_R = create_output_string(right_path, R_I, R_E)
    OUTPUT_DATASET_FILE_PATH = os.path.join(pair_path, 'dataset.txt')
    with open(OUTPUT_DATASET_FILE_PATH, 'w') as f:
        f.write(out_L)
        f.write('\n')
        f.write(out_R)
    f.close()