#%%
from DL.keras.dataloader import get_divided_data
from DL.keras.model import train_val
import warnings
warnings.filterwarnings("ignore")


path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\dataset.npy'
val_size = 0.3
test_size = 0.5  # out of validation dataset
n_epochs = 50
lr = 0.001
batch_size = 1000
dummy_mult = 6
train_data, val_data, test_data,size_of_sample = get_divided_data(path=path, val_size=val_size, ts_size=test_size, dummy_mult=dummy_mult)

train_val(train_data=train_data, val_data=val_data, n_epochs=n_epochs, lr=lr, batch_size=batch_size, size_of_sample=size_of_sample)

#TODO create folder with subfolders of each pairs: folder consists of *.csv files with X-Y points

#TODO write dataloader, that download all feature correspondences into one dataset

#TODO test created dataset with trained DL model

#TODO evaluate each predicting