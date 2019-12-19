import os
import glob
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers

# from keras.layers import Dense, Dropout
# from keras.models import Sequential
# from keras.regularizers import l2
# from keras import optimizers


from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

def dl_model(input_size):
    model = Sequential()

    model.add(Dense(100, activation='relu', input_dim=input_size, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3, noise_shape=None, seed=None))

    model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3, noise_shape=None, seed=None))

    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3, noise_shape=None, seed=None))

    model.add(Dense(25, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3, noise_shape=None, seed=None))

    model.add(Dense(1, activation='sigmoid'))

    return model


def train_val(train_data, val_data, n_epochs, lr, batch_size, size_of_sample, checkpoint):
    model = dl_model(size_of_sample)
    optim = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])
    model.summary()
    x_tr, y_tr = train_data
    x_val, y_val = val_data

    model_output = model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size, verbose=1,
                             validation_data=(x_val, y_val), )
    model.save(checkpoint)


def test(test_data, checkpoint, size_of_sample):
    x_ts, y_ts = test_data
    model = dl_model(input_size=size_of_sample)
    model.load_weights(checkpoint)
    model.summary()

    pred = model.predict_classes(x_ts).reshape(-1)
    acc = accuracy_score(y_true=y_ts, y_pred=pred)
    cm = confusion_matrix(y_true=y_ts, y_pred=pred)
    f1 = f1_score(y_true=y_ts, y_pred=pred, average=None)
    print(cm)
    print("Test dataset: acc:{:5f}".format(acc))
    print("F1 score", f1)

def predict(inference_data, target,checkpoint, size_of_sample):
    x_inf = inference_data
    model = dl_model(input_size=size_of_sample)
    model.load_weights(checkpoint)
    model.summary()

    pred = model.predict_classes(x_inf)
    cm = confusion_matrix(y_true=target, y_pred=pred)
    return cm