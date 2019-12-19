# %%
import warnings

from DL.keras.dataloader import get_divided_data, get_pair_features,merge_dataset, new_get_pair_features
from DL.keras.model import train_val, test, predict

warnings.filterwarnings("ignore")

path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial\dataset.csv'
checkpoint = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\model.h5'
stereo_images_features = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data'
val_size = 0.3
test_size = 0.5  # out of validation dataset
n_epochs = 500
lr = 0.001
batch_size = 100
dummy_mult = 10

phase = 'inferencePair'  # train or test or predict
train_data, val_data, test_data, size_of_sample, scaler = merge_dataset(
    artificial_data_path=path,ts_size=0.5, val_size=val_size,
    dummy_mult=dummy_mult, stereo_images_folder=stereo_images_features, mergeScale=False
)



# train_val(train_data=train_data, val_data=val_data, n_epochs=n_epochs, lr=lr, batch_size=batch_size,
#           size_of_sample=size_of_sample, checkpoint=checkpoint)
# test(test_data, checkpoint, size_of_sample)
# test(val_data, checkpoint, size_of_sample)
# test(train_data, checkpoint, size_of_sample)
# folder = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\pair_16\experiment'
folder = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data\pair_19.csv'
# # inference_data = get_pair_features(folder=folder, size_of_sample=size_of_sample, scaler=scaler,artificial_data = False)
X,Y = new_get_pair_features(file=folder,scaler = scaler)
conf_matrix = predict(X,Y, checkpoint, size_of_sample)
print(conf_matrix)