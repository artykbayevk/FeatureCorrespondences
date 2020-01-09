# %%
import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from joblib import dump, load
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

# %%


class Model:
    def __init__(self, data_folder, phase, typeOfModel, checkpoint):
        self.data_folder = data_folder
        self.phase = phase
        self.type = typeOfModel
        self.checkpoint = checkpoint

    def data_load(self):
        list_of_stereo = glob.glob(os.path.join(self.data_folder, "*.csv"))
        stereo = []
        for file in list_of_stereo:
            tmp_read_data = pd.read_csv(file, header=None, delimiter=',')
            stereo.append(tmp_read_data)
        stereo_dataset = pd.concat(stereo)
        Y = stereo_dataset[stereo_dataset.shape[1]-1].values
        X = stereo_dataset.drop(stereo_dataset.shape[1]-1, axis=1).values
        stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        stratSplit.get_n_splits(X,Y)

        for train_idx, test_idx in stratSplit.split(X,Y):
            X_train = X[train_idx]
            Y_train = Y[train_idx]

            X_test = X[test_idx]
            Y_test = Y[test_idx]

        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        self.train_data = (X_train, Y_train)
        self.test_data = (X_test, Y_test)
        self.full_data = (X,Y)

    def train_cnn(self):
        train_X, train_Y = self.train_data
        test_X, test_Y = self.test_data


    def train(self):
        print("Model configured and ready for train")
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier())
            ]
        )
        parameter_space = {
            'mlp__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'mlp__activation': ['tanh', 'relu'],
            'mlp__solver': ['sgd', 'adam'],
            'mlp__alpha': [0.0001, 0.05],
            'mlp__learning_rate': ['constant', 'adaptive']
        }
        model = GridSearchCV(pipeline, param_grid=parameter_space, scoring='f1',n_jobs=1 )
        model.fit(self.train_data[0], self.train_data[1])
        print("Best parameters of model: ", model.best_params_)
        dump(model, self.checkpoint)

    def evaluate(self):
        model = load(self.checkpoint)
        X = self.test_data[0]
        Y = self.test_data[1]
        pred = model.predict(X)
        score = model.score(X,Y)
        accuracy = model.score(X,Y)
        f1 = f1_score(y_true=Y, y_pred=pred, average=None)
        cm = confusion_matrix(y_true=Y, y_pred=pred)
        print(score, accuracy, f1)
        print(cm)

    def inference(self, solutions_path):
        pair = pd.read_csv(solutions_path, sep=",", header=None, index_col=None)
        X = pair.drop(pair.shape[1] - 1, axis=1).values
        Y = pair[pair.shape[1] - 1].values
        model = load(self.checkpoint)

        pred = model.predict(X)
        cm = confusion_matrix(y_true=Y, y_pred=pred)
        print(cm)

DATA_PATH = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data'
PHASE = 'inference' # or can be evaluate or inference
TYPE_OF_MODEL = 'sklearn' # or can be keras
CHECKPOINT = r"C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\filename.joblib" # or it can be keras.h5
SOLUTION_PATH = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_13.csv'


DL = Model(DATA_PATH, PHASE, TYPE_OF_MODEL, CHECKPOINT)

# in inference dont need to collect data
DL.data_load()
DL.train_cnn()



'''
    SIMPLE MLP/DNN/FCNetwork
'''
# train process
# DL.train()

# evaluate process
# DL.evaluate()

# inference on real data
# DL.inference(SOLUTION_PATH)

