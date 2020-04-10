import os
import glob
import pandas as pd
import numpy as np
import warnings
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, validation_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.getcwd()))


class Model:
    def __init__(self, data_folder, checkpoint):
        self.data_folder = data_folder
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
        Y_TR = np.array(Y_TR).reshape(-pair_num * int(test_size*num_sol), 1)
        Y_TS = np.array(Y_TS).reshape(-pair_num * int(test_size*num_sol), 1)

        self.train_data = (X_TR, Y_TR)
        self.test_data = (X_TS, Y_TS)

        print("train data shape: ", X_TR.shape)
        print("test data shape: ", X_TS.shape)

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

        model = GridSearchCV(pipeline, param_grid=parameter_space, scoring='f1', n_jobs=1, cv=10)
        model.fit(self.full_data[0], self.full_data[1])
        print("Best parameters of model: ", model.best_params_)
        print("Best score: ", model.scorer_)
        dump(model, self.checkpoint)

    def train_val(self):
        """
        training with learning curve
        :return:
        """
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
        cv = StratifiedKFold(n_splits=10, random_state=42)
        model = GridSearchCV(pipeline, param_grid=parameter_space, scoring='f1', n_jobs=1, cv=cv)
        model.fit(self.train_data[0], self.train_data[1])
        print("Tuned rg best params: {}".format(model.best_params_))
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
    @staticmethod
    def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean + train_std,
                         train_mean - train_std, color='blue', alpha=alpha)
        plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
        plt.title(title)
        plt.xlabel('Number of training points')
        plt.ylabel('F-measure')
        plt.grid(ls='--')
        plt.legend(loc='best')
        plt.show()

    @staticmethod
    def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
        param_range = [x[1] for x in param_range]
        sort_idx = np.argsort(param_range)
        param_range = np.array(param_range)[sort_idx]
        train_mean = np.mean(train_scores, axis=1)[sort_idx]
        train_std = np.std(train_scores, axis=1)[sort_idx]
        test_mean = np.mean(test_scores, axis=1)[sort_idx]
        test_std = np.std(test_scores, axis=1)[sort_idx]
        plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
        plt.fill_between(param_range, train_mean + train_std,
                         train_mean - train_std, color='blue', alpha=alpha)
        plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
        plt.title(title)
        plt.grid(ls='--')
        plt.xlabel('Weight of class 2')
        plt.ylabel('Average values and standard deviation for F1-Score')
        plt.legend(loc='best')
        plt.show()


DATA_PATH = os.path.join(BASE, "data", "dataset", "stereo_heuristic_data")
CHECKPOINT_PATH = os.path.join(BASE, "DL", "keras", "submission_models", "dataset_1.joblib")

DL = Model(
    data_folder=DATA_PATH,
    checkpoint=CHECKPOINT_PATH
)
DL.data_load()
DL.ratio_data_loader()
DL.train_val()

#
# from collections import Counter
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, validation_curve, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     test_mean = np.mean(test_scores, axis=1)
#     test_std = np.std(test_scores, axis=1)
#     plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
#     plt.fill_between(train_sizes, train_mean + train_std,
#                      train_mean - train_std, color='blue', alpha=alpha)
#     plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
#
#     plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
#     plt.title(title)
#     plt.xlabel('Number of training points')
#     plt.ylabel('F-measure')
#     plt.grid(ls='--')
#     plt.legend(loc='best')
#     plt.show()
#
#
# def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
#     param_range = [x[1] for x in param_range]
#     print(param_range)
#     print(train_scores, test_scores)
#     sort_idx = np.argsort(param_range)
#     print(sort_idx)
#     param_range=np.array(param_range)[sort_idx]
#     train_mean = np.mean(train_scores, axis=1)[sort_idx]
#     train_std = np.std(train_scores, axis=1)[sort_idx]
#     test_mean = np.mean(test_scores, axis=1)[sort_idx]
#     test_std = np.std(test_scores, axis=1)[sort_idx]
#     plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
#     plt.fill_between(param_range, train_mean + train_std,
#                  train_mean - train_std, color='blue', alpha=alpha)
#     plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
#     plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
#     plt.title(title)
#     plt.grid(ls='--')
#     plt.xlabel('Weight of class 2')
#     plt.ylabel('Average values and standard deviation for F1-Score')
#     plt.legend(loc='best')
#     plt.show()
#
#
# if __name__ == '__main__':
#     X, y = make_classification(n_classes=2, class_sep=2, weights=[0.9, 0.1], n_informative=3, n_redundant=1, flip_y=0,
#                                n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
#     print('Original dataset shape {}'.format(Counter(y)))
#
#     ln = X.shape
#     names = ["x%s" % i for i in range(1, ln[1] + 1)]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#     st = StandardScaler()
#
#     rg = LogisticRegression(class_weight={0: 1, 1: 6.5}, random_state=42, solver='saga', max_iter=100, n_jobs=-1)
#
#     param_grid = {'clf__C': [0.001, 0.01, 0.1, 0.002, 0.02, 0.005, 0.0007, .0006, 0.0005],
#                   'clf__class_weight': [{0: 1, 1: 6}, {0: 1, 1: 4}, {0: 1, 1: 5.5}, {0: 1, 1: 4.5}, {0: 1, 1: 5}]}
#
#     pipeline = Pipeline(steps=[('scaler', st),
#                                ('clf', rg)])
#
#     cv = StratifiedKFold(n_splits=5, random_state=42)
#     rg_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1')
#     rg_cv.fit(X_train, y_train)
#     print("Tuned rg best params: {}".format(rg_cv.best_params_))
#
#     plt.figure(figsize=(9, 6))
#     param_range1 = [i/10000 for i in range(1, 11)]
#     param_range2 = [{0: 1, 1: 6}, {0: 1, 1: 4}, {0: 1, 1: 5.5}, {0: 1, 1: 4.5}, {0: 1, 1: 5}]
#
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator=rg_cv.best_estimator_, X=X_train, y=y_train,
#         train_sizes=np.arange(0.1, 1.1, 0.1), cv=cv, scoring='f1', n_jobs=- 1)
#
#     plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for Logistic Regression')
#
#     # train_scores, test_scores = validation_curve(
#     # #     estimator=rg_cv.best_estimator_, X=X_train, y=y_train, param_name="clf__C", param_range=param_range1,
#     # #     cv=cv, scoring="f1", n_jobs=-1)
#     # # plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Curve for C", alpha=0.1)
#
#     train_scores, test_scores = validation_curve(
#         estimator=rg_cv.best_estimator_, X=X_train, y=y_train, param_name="clf__class_weight", param_range=param_range2,
#         cv=cv, scoring="f1", n_jobs=-1)
#     plot_validation_curve(param_range2, train_scores, test_scores, title="Validation Curve for class_weight", alpha=0.1)