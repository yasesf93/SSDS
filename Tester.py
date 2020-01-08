import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)


class Tester(object):
    def __init__(self, model, testdataloader, optimizer, criterion, classes, device, n_epoch, **kwargs):
        self.model = model
        self.testdataloader = testdataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.classes = classes
        self.batchsizets = config['batchsizets']
        self.device = device
        self.best_acc = 0
        self.startepoch = 0
        self.n_epoch = n_epoch

    def test_minibatch(self, batch_idx):
        (I, targets) = self.testdataloader[batch_idx]
        I, targets = I.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(I)
        targets=targets.long()
        loss = self.criterion(outputs, targets)
        _,predicted = outputs.max(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)


    def test_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        loss_avg = 0
        for i in range(len(self.testdataloader)):
            batch_loss, batch_correct, batch_total = self.test_minibatch(i)
            test_loss += batch_loss
            total += batch_total
            correct += batch_correct
            loss_avg = test_loss/(i+1)
            progress_bar(i, len(self.testdataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_avg, 100.*correct/total, correct, total))
        
        #Save checkpoint Best
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = { 'net': self.model.state_dict(),'acc': acc, 'epoch': epoch,}
            if not os.path.isdir('.checkpoint'):
                os.mkdir('.checkpoint')
            torch.save(state, '.checkpoint/ckpt.testbest')
            best_acc = acc


    def test(self, epoch):
        print(epochs)
        for epoch in range(self.start_epoch, epochs+1):
            self.test_epoch(epoch)  