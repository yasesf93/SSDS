import numpy as np
import torch
import json
from .dataloaderdel import DelDataLoader 


class DelDataLoaderIMG(DelDataLoader):
    """docstring for DataLoader"""
    def __init__(self, dataset, batch_size, shuffle=True):
        super(DelDataLoader, self).__init__(dataset, batch_size)
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
