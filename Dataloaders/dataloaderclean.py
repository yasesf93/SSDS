import numpy as np
import torch


class CustomDataLoader(object):
    """docstring for DataLoader"""
    def __init__(self, dataset, batch_size, shuffle=True):
        super(CustomDataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        """'Generate one batch of data'"""
        # Generate indexes of the batch
        #print("here")
        if index == 'all':
            indexes = self.indexes
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        I_batch, Target_batch = [], [], []
        for index in indexes:
            I_samp, Target_samp = self.dataset[index]
            I_batch.append(I_samp)
            Target_batch.append(Target_samp)
        I_batch = torch.stack(I_batch)
        return I_batch, torch.Tensor(Target_batch)