from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow import keras

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


def train_val(train_data, val_data, n_epochs, lr, batch_size, size_of_sample):
    model = dl_model(size_of_sample)
    optim = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])
    model.summary()
    x_tr, y_tr = train_data
    x_val, y_val = val_data

    model_output = model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val), )