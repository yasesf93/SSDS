import numpy as np
import torch
import json
from .dataloaderdel import DelDataLoader 

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

seed_num = config['random_seed']
np.random.seed(seed_num)


class DelDataLoaderIMG(DelDataLoader):
    """docstring for DataLoader"""
    def __init__(self, dataset, batch_size, shuffle=True):
        super(DelDataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def set_delta(self, index, deltas):
        if index == 'all':
            indexes = self.indexes
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i, idx in enumerate(indexes):
            self.dataset.set_delta(idx, deltas[i])   