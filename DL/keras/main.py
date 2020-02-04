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
        stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        stratSplit.get_n_splits(X,Y)

        for train_idx, test_idx in stratSplit.split(X,Y):
            X_train = X[train_idx]
            Y_train = Y[train_idx]

            X_test = X[test_idx]
            Y_test = Y[test_idx]

        self.train_data = (X_train, Y_train)
        self.test_data = (X_test, Y_test)
        self.full_data = (X,Y)

    def model(self):
        model = keras.models.Sequential()
        model.add(Conv1D(100, kernel_size=2, strides=2, input_shape=(200, 2),activation='relu'))
        model.add(Conv1D(100, kernel_size=1, strides=1, activation='relu'))
        model.add(MaxPool1D(2))
        model.add(Conv1D(50, kernel_size=1, strides=1, activation='relu'))
        model.add(Conv1D(25, kernel_size=1, strides=1, activation='relu'))
        model.add(MaxPool1D(2))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='softmax'))
        return model

    def dl_model(self,input_size):
        model = Sequential()

        model.add(Dense(100, activation='relu', input_dim=input_size, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3, noise_shape=None, seed=None))
        #
        # model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.3, noise_shape=None, seed=None))
        #
        # model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.3, noise_shape=None, seed=None))

        model.add(Dense(25, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3, noise_shape=None, seed=None))

        model.add(Dense(1, activation='sigmoid'))

        return model
    def train_dnn(self):
        size_of_sample = self.full_data[0].shape[1]
        model = self.dl_model(size_of_sample)
        optim = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', "mae", "mse"])
        model.summary()
        model_output = model.fit(self.full_data[0], self.full_data[1], epochs=100,
                                 batch_size=10, verbose=1,validation_split=0.2)
        model.save(self.checkpoint)

    def evaluate_dnn(self):
        size_of_sample = self.full_data[0].shape[1]
        model = self.dl_model(input_size=size_of_sample)
        model.load_weights(self.checkpoint)
        optim = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', "mae", "mse"])
        model.summary()

        res = model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(res)

    def predict_pair(self, sols_path):
        data = pd.read_csv(sols_path, header=None, delimiter=',')
        X = data.drop(data.shape[1]-1, axis=1).values
        Y = data[data.shape[1]-1].values.astype(int)

        size_of_sample = self.full_data[0].shape[1]
        model = self.dl_model(input_size=size_of_sample)
        model.load_weights(self.checkpoint)
        optim = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', "mae", "mse"])
        model.summary()

        pred = model.predict_classes(X).reshape(-1)
        print("PRED")
        print(pred)
        print("ACTUAL")
        print(Y)

    def train_cnn(self):

        data_X, data_Y = self.full_data


        def func(sample):
            x_indices = np.arange(0, sample.shape[0], 2)
            y_indices = np.arange(1, sample.shape[0], 2)
            x = sample[x_indices]
            y = sample[y_indices]
            upd_sample = np.array([x,y]).T
            return upd_sample
        data_X = np.apply_along_axis(func, 1, data_X)
        print(data_X.shape, data_Y.shape)

        clf = self.model()
        clf.compile(loss='binary_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]
        BATCH_SIZE = 100
        EPOCHS = 1000

        history = clf.fit(data_X,
                              data_Y,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              callbacks=callbacks_list,
                              validation_split=0.2,
                              verbose=1)


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
CHECKPOINT = r"C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\keras_model.h5" # or it can be keras.h5
SOLUTION_PATH = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_3.csv'


DL = Model(DATA_PATH, PHASE, TYPE_OF_MODEL, CHECKPOINT)

# in inference dont need to collect data
DL.data_load()
DL.train_dnn()
DL.evaluate_dnn()
# DL.predict_pair(SOLUTION_PATH)
# DL.predict_pair(SOLUTION_PATH)
# DL.predict_pair(SOLUTION_PATH)
# DL.predict_pair(SOLUTION_PATH)
# DL.predict_pair(SOLUTION_PATH)
# DL.predict_pair(SOLUTION_PATH)

# DL.train_cnn()



'''
    SIMPLE MLP/DNN/FCNetwork
'''
# train process
# DL.train()

send_email(
    user="crm.kamalkhan@gmail.com",
    pwd="Astana2019",
    recipient="kamalkhan.artykbayev@nu.edu.kz",
    subject="Deep Learning Model",
    body="Its ready"
)

# evaluate process
# DL.evaluate()

# inference on real data
# DL.inference(SOLUTION_PATH)

