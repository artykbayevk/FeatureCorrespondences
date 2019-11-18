# %%
import warnings

from DL.keras.dataloader import get_divided_data, get_pair_features
from DL.keras.model import train_val, test, predict

warnings.filterwarnings("ignore")

path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\artificial\dataset.csv'
checkpoint = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\DL\keras\model.h5'
val_size = 0.3
test_size = 0.5  # out of validation dataset
n_epochs = 100
lr = 0.001
batch_size = 100
dummy_mult = 10

phase = 'noth'  # train or test or predict
train_data, val_data, test_data, size_of_sample, scaler = get_divided_data(path=path, val_size=val_size, ts_size=test_size,
                                                                   dummy_mult=dummy_mult)

if phase == 'train':
    train_val(train_data=train_data, val_data=val_data, n_epochs=n_epochs, lr=lr, batch_size=batch_size,
              size_of_sample=size_of_sample, checkpoint=checkpoint)
elif phase == 'test':
    test(test_data, checkpoint, size_of_sample)
elif phase == 'inferencePair':
    folder = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\pair_8\experiment'
    inference_data = get_pair_features(folder=folder, size_of_sample=size_of_sample)
    pred = predict(inference_data, checkpoint, size_of_sample)
    print(pred)
# TODO evaluate with stereo images
# TODO try to train with 0-mean
# TODO meeting w/ professor and ask about any other ways to implement that
Y = train_data[1]
print(sum(Y))
print(Y.shape)

Y = test_data[1]
print(sum(Y))
print(Y.shape)