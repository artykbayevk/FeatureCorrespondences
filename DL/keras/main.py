# %%
import warnings

from DL.keras.dataloader import get_divided_data, get_pair_features
from DL.keras.model import train_val, test, predict

warnings.filterwarnings("ignore")

path = '/Users/artkvk/Documents/RA/FeatureCorrespondences/data/dataset/dataset.npy'
checkpoint = '/Users/artkvk/Documents/RA/FeatureCorrespondences/DL/keras/model.h5'
val_size = 0.3
test_size = 0.5  # out of validation dataset
n_epochs = 50
lr = 0.001
batch_size = 1000
dummy_mult = 10

phase = 'inferencePair'  # train or test or predict
train_data, val_data, test_data, size_of_sample, scaler = get_divided_data(path=path, val_size=val_size, ts_size=test_size,
                                                                   dummy_mult=dummy_mult)
if phase == 'train':
    train_val(train_data=train_data, val_data=val_data, n_epochs=n_epochs, lr=lr, batch_size=batch_size,
              size_of_sample=size_of_sample, checkpoint=checkpoint)
elif phase == 'test':
    test(test_data, checkpoint, size_of_sample)
elif phase == 'inferencePair':
    folder = '/Users/artkvk/Documents/RA/FeatureCorrespondences/data/dense/experiment'
    inference_data = get_pair_features(folder=folder, size_of_sample=size_of_sample)
    # inference_data = scaler.transform(inference_data)
    predict(inference_data, checkpoint, size_of_sample)

# TODO create folder with sub-folders of each pairs: folder consists of *.csv files with X-Y points

# TODO write data-loader, that download all feature correspondences into one dataset

# TODO test created dataset with trained DL model

# TODO evaluate each predicting

