import torch.nn as nn
from ..Dataloaders import CustomDataLoader
from torch.optim.optimizer import Optimizer
from ..utils import progress_bar
from ..visualizations import PlotLoss, PlotAcc
import torch
import os

class Trainer(object):
    def __init__(self, model, traindataloader, testdataloader, optimizer, criterion, **kwargs):
        self.model = model
        self.traindataloader = traindataloader
        self.testdataloader = testdataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.classes = kwargs.get('classes')
        self.batchsizetr = kwargs.get('batchsizetr')
        self.batchsizets = kwargs.get('batchsizets')
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0
        self.startepoch = kwargs.get('startepoch')
        self.n_epoch = kwargs.get('n_epoch')


    def train_minibatch(self, batch_idx):
        (I, targets) = self.traindataloader[batch_idx]
        I, targets = I.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(I)
        targets=targets.long()
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _,predicted = outputs.max(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)


    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        loss_avg = 0
        for i in range(len(self.traindataloader)):
            batch_loss, batch_correct, batch_total = self.train_minibatch(i)
            train_loss += batch_loss
            total += batch_total
            correct += batch_correct
            loss_avg = train_loss/(i+1)
            progress_bar(i, len(self.traindataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_avg, 100.*correct/total, correct, total))
        
        #Save checkpoint Best
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = { 'net': self.model.state_dict(),'acc': acc, 'epoch': epoch,}
            if not os.path.isdir('.checkpoint'):
                os.mkdir('.checkpoint')
            torch.save(state, '.checkpoint/ckpt.trainbest')
            best_acc = acc


    def train(self, epochs=n_epoch):
        print(epochs)
        for epoch in range(self.start_epoch, epochs+1):
            self.train_epoch(epoch)  
    


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


    def test(self, epochs=n_epoch):
        print(epochs)
        for epoch in range(self.start_epoch, epochs+1):
            self.test_epoch(epoch)  