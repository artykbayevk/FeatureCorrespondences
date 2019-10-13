import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def transform_samples(dataset, mult, size_of_sample):
    data = np.zeros((dataset.shape[0], size_of_sample+1))
    stop =1
    for idx, sample in enumerate(dataset):
        label = sample[-1:]
        features = sample[:-1]
        tiled = np.tile(features, mult)
        sample_ = np.concatenate((tiled, label))
        data[idx] = sample_

    return data


def get_divided_data(path, ts_size, val_size, dummy_mult):
    dataset = np.load(path)
    size_of_sample = (dataset.shape[1]-1) * dummy_mult
    dataset = transform_samples(dataset, dummy_mult ,size_of_sample)
    main_scaler = StandardScaler().fit(dataset[:, :-1])
    X_raw = dataset[:, :-1]
    Y = dataset[:, -1]

    x_tr, x_val_origin, y_tr, y_val_origin = train_test_split(X_raw, Y, test_size=val_size, random_state=42)
    x_val, x_ts, y_val, y_ts = train_test_split(x_val_origin, y_val_origin, test_size=ts_size, random_state=42)

    x_tr = main_scaler.transform(x_tr)
    x_ts = main_scaler.transform(x_ts)
    x_val = main_scaler.transform(x_val)

    return (x_tr, y_tr), (x_val, y_val), (x_ts, y_ts), size_of_sample