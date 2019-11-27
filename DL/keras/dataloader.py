import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

def transform_samples(dataset, mult, size_of_sample):
    data = np.zeros((dataset.shape[0], size_of_sample+1))
    for idx, sample in enumerate(dataset):
        label = sample[-1:]
        features = sample[:-1]
        tiled = np.tile(features, mult)
        sample_ = np.concatenate((tiled, label))
        data[idx] = sample_

    return data


def get_divided_data(path, ts_size, val_size, dummy_mult, ownScaler = True):
    dataset = pd.read_csv(path, header=None).values
    size_of_sample = (dataset.shape[1]-1) * dummy_mult
    dataset = transform_samples(dataset, dummy_mult ,size_of_sample)
    main_scaler = StandardScaler().fit(dataset[:, :-1])

    if ownScaler:
        df = np.copy(dataset[:, :-1]).reshape(dataset.shape[0], int(dataset[:,:-1].shape[1]/2), 2)
        min = df.min()
        max = df.max()
        df = np.interp(df, (min, max), (0, 100))
        mean_X = df[:, :, 0].mean()
        mean_Y = df[:, :, 1].mean()
        df[:, :, 0] = df[:, :, 0] - mean_X
        df[:, :, 1] = df[:, :, 1] - mean_Y
        df = df.reshape(dataset.shape[0], dataset[:,:-1].shape[1])
        X_raw = np.copy(df)
    else:
        X_raw = dataset[:, :-1]
    Y = dataset[:, -1]

    x_tr, x_val_origin, y_tr, y_val_origin = train_test_split(X_raw, Y, test_size=val_size, random_state=42)
    x_val, x_ts, y_val, y_ts = train_test_split(x_val_origin, y_val_origin, test_size=ts_size, random_state=42)

    if ownScaler == False:
        x_tr = main_scaler.transform(x_tr)
        x_ts = main_scaler.transform(x_ts)
        x_val = main_scaler.transform(x_val)

    print("X_train size:{}".format(x_tr.shape))
    print("X_train size:{}".format(x_val.shape))
    print("X_train size:{}".format(x_ts.shape))

    return (x_tr, y_tr), (x_val, y_val), (x_ts, y_ts), size_of_sample, main_scaler


def get_pair_features(folder, size_of_sample, ownScaler = True, artificial_data=False):
    list_of_files = glob.glob(os.path.join(folder, "*.csv"))
    dataset = np.zeros((len(list_of_files), size_of_sample))
    for idx, item in enumerate(list_of_files):
        dataset[idx] = pd.read_csv(item, header=None, delimiter=',').values.flatten()

    # generate new artificial not optimal solutions here
    if artificial_data:
        gen_df = pd.DataFrame(columns = range(280), index=range(100))
        sample_df = np.copy(dataset)
        for i in range(100):
            rand_index = np.random.randint(0, sample_df.shape[0], size=(1,)).item()
            sample = sample_df[rand_index, :].reshape(140,2)

            P = sample[::2]
            Q = sample[1::2]
            np.random.shuffle(P)
            np.random.shuffle(Q)

            sample = np.stack((P,Q), axis=1).reshape(280)
            gen_df.iloc[i] = sample

        dataset = np.concatenate((dataset, gen_df))

    if ownScaler:
        df = np.copy(dataset).reshape(dataset.shape[0], int(dataset.shape[1]/2), 2)
        df = np.interp(df, (df.min(), df.max()), (0, 100))
        mean_X = df[:, 0].mean()
        mean_Y = df[:, 1].mean()

        df[:, 0] = df[:, 0] - mean_X
        df[:, 1] = df[:, 1] - mean_Y
        df = df.reshape(df.shape[0],280)
        return df
    return dataset