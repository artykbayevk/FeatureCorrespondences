
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
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers

from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D, GlobalAveragePooling1D
from keras.layers import Dropout
import keras
from keras import backend as K
from scripts.email import send_email

# %%

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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

        self.full_data = (X,Y)
        print("data shape:", X.shape)
    def ratio_data_loader(self):
        """

        :return: train_data and test_data updated
        """
        test_size = 0.1
        num_sol = 100
        num_of_features = 200



        pair_num = int(self.full_data[0].shape[0]/num_sol) # 20 is the num of solutions

        X_TR = []
        X_TS = []
        Y_TR = []
        Y_TS = []

        dataset_X = self.full_data[0].reshape(pair_num, num_sol, num_of_features)
        dataset_Y = self.full_data[1].reshape(pair_num, num_sol)
        for idx, pair_X in enumerate(dataset_X):
            pair_Y = dataset_Y[idx]
            stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            stratSplit.get_n_splits(pair_X, pair_Y)
            for train_idx, test_idx in stratSplit.split(pair_X, pair_Y):
                X_train = pair_X[train_idx]
                Y_train = pair_Y[train_idx]

                X_test = pair_X[test_idx]
                Y_test = pair_Y[test_idx]

                X_TR.append(X_train)
                X_TS.append(X_test)
                Y_TR.append(Y_train)
                Y_TS.append(Y_test)
        X_TR = np.array(X_TR).reshape(-pair_num * int(test_size*num_sol), num_of_features)
        X_TS = np.array(X_TS).reshape(-pair_num * int(test_size*num_sol), num_of_features)
        Y_TR = np.array(Y_TR).reshape(-pair_num * int(test_size*num_sol),1)
        Y_TS = np.array(Y_TS).reshape(-pair_num * int(test_size*num_sol),1)

        self.train_data = (X_TR, Y_TR)
        self.test_data = (X_TS, Y_TS)

        print("train data shape: ", X_TR.shape)
        print("test data shape: ", X_TS.shape)

    def dl_model(self,input_size):
        model = Sequential()

        model.add(Dense(100, activation='relu', input_dim=input_size, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3, noise_shape=None, seed=None))

        model.add(Dense(25, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3, noise_shape=None, seed=None))

        model.add(Dense(1, activation='sigmoid'))

        return model
    def train_evaluate_dnn(self):
        size_of_sample = self.full_data[0].shape[1]
        model = self.dl_model(size_of_sample)
        optim = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', "mae", "mse"])
        model.summary()
        X = self.train_data[0]
        Y = self.train_data[1]
        self.scaler = StandardScaler().fit(X)
        X_norm = self.scaler.transform(X)
        model_output = model.fit(X_norm, Y, epochs=100,
                                 batch_size=10, verbose=1,validation_data=(self.scaler.transform(self.test_data[0]), self.test_data[1]))
        model.save(self.checkpoint)
        res = model.evaluate(self.scaler.transform(self.test_data[0]), self.test_data[1], verbose=0)


    def train(self):
        print("Model configured and ready for train.TRAINING")
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
        model = GridSearchCV(pipeline, param_grid=parameter_space, scoring='f1',n_jobs=1, cv=10)
        model.fit(self.full_data[0], self.full_data[1])
        print("Best parameters of model: ", model.best_params_)
        print("Best score: ", model.scorer_)
        dump(model, self.checkpoint)

    def evaluate(self):
        model = load(self.checkpoint)
        X = self.full_data[0]
        Y = self.full_data[1]
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



DATA_PATH = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset_2\main\stereo_heuristic_data\herjzesu_entry_castle'
PHASE = 'inference' # or can be evaluate or inference
TYPE_OF_MODEL = 'sklearn' # or can be keras
CHECKPOINT = r"C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\combined_models\herjzesu_entry_castle.joblib"

MODEL_TYPE = 'MLP'

DL = Model(DATA_PATH, PHASE, TYPE_OF_MODEL, CHECKPOINT)
DL.data_load()
#DL.ratio_data_loader()

'''
    DNN MODEL ON KERAS
'''
if MODEL_TYPE == "DNN":
    DL.train_evaluate_dnn()



'''
    SIMPLE MLP/DNN/FCNetwork
'''
if MODEL_TYPE == 'MLP':
    DL.train()
    DL.evaluate()
    all_stereo_data = glob.glob(r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset_2\main\stereo_heuristic_data\herjzesu_entry_castle\*.csv')
    for file in all_stereo_data:
        print(os.path.basename(file))
        DL.inference(file)



send_email(
    user="crm.kamalkhan@gmail.com",
    pwd="Astana2019",
    recipient="kamalkhan.artykbayev@nu.edu.kz",
    subject="Deep Learning Model",
    body="Its ready"
)