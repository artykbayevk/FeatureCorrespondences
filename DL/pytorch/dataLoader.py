import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class SampleDataset(Dataset):
    def __init__(self, path, input_size):
        self.data = np.load(path)
        self.input_size = input_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = sample[:-1]
        label = sample[-1]

        ## transformation and adding dummy features
        mult = int(self.input_size/features.shape[0])
         # = np.zeros(input_size)
        dummied_features = np.tile(features, mult)
        ##

        features = torch.from_numpy(dummied_features).float()
        label = torch.LongTensor([label])
        return features, label

def prepare_samplers(set, val_size,test_size, shuffle = True):
    dataset_size = len(set)
    indices = list(range(dataset_size))
    split_ts = int(np.floor(test_size *  dataset_size))
    split_val = int(np.floor(split_ts+ val_size * (dataset_size)))
    if shuffle:
        np.random.shuffle(indices)

    test_indices, val_indices, train_indices = indices[:split_ts], indices[split_ts:split_val], indices[split_val:]
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices), SubsetRandomSampler(test_indices)

path = r'C:\Users\user\Documents\Research\FeatureCorrespondenes\data\dataset\dataset.npy'
batch_size = 1000
input_size = 28
test_size = 0.2
val_size = 0.1

dataset = SampleDataset(path, input_size)
tr_sampler, val_sampler, ts_sampler = prepare_samplers(dataset, val_size, test_size, shuffle=False)

test_loader = DataLoader(dataset = dataset, batch_size=batch_size, sampler=ts_sampler)
val_loader = DataLoader(dataset = dataset, batch_size = batch_size, sampler=val_sampler)
train_loader = DataLoader(dataset=dataset, batch_size = batch_size, sampler=tr_sampler)

stop = 1

print("Train loader size: {}, validation loader size: {}, test loader size: {}".format(
    len(train_loader.dataset),
    len(val_loader.dataset),
    len(test_loader.dataset)
))