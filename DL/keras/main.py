# %%
import warnings

from DL.keras.dataloader import get_divided_data, get_pair_features,merge_dataset
from DL.keras.model import train_val, test, predict

warnings.filterwarnings("ignore")

path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial\dataset.csv'
checkpoint = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\model.h5'
stereo_images_features = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\stereo_heuristic_data'
val_size = 0.3
test_size = 0.5  # out of validation dataset
n_epochs = 100
lr = 0.001
batch_size = 10
dummy_mult = 10

phase = 'test'  # train or test or predict
train_data, val_data, test_data,size_of_sample = merge_dataset(
    artificial_data_path=path,ts_size=0.5, val_size=val_size,
    dummy_mult=dummy_mult, stereo_images_folder=stereo_images_features, mergeScale=False
)



if phase == 'train':
    train_val(train_data=train_data, val_data=val_data, n_epochs=n_epochs, lr=lr, batch_size=batch_size,
              size_of_sample=size_of_sample, checkpoint=checkpoint)
elif phase == 'test':
    test(test_data, checkpoint, size_of_sample)
elif phase == 'inferencePair':
    folder = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\pair_22\experiment'
    inference_data = get_pair_features(folder=folder, size_of_sample=size_of_sample, artificial_data = True)
    pred = predict(inference_data, checkpoint, size_of_sample)
    for i in range(pred.shape[0]):
        print(i, pred[i])