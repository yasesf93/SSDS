import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from Visualizations import PlotLoss,PlotAcc

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseTester(object):
    def __init__(self, model, testdataloader, optimizer, criterion, classes, n_epoch, testbatchsize, expid, checkepoch, pres, atmeth, **kwargs):
        self.model = model
        self.testdataloader = testdataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.classes = classes
        self.batchsizets = testbatchsize
        self.best_acc = 0
        self.startepoch = 0
        self.n_epoch = n_epoch
        self.expid = expid
        self.checkepoch = checkepoch
        self.pres = pres
        self.atmeth = atmeth
        self.log = {}
        self.log['test_acc'] = []   
        self.ts_acc_ls = []
             

    def test_minibatch(self, batch_idx):
        raise NotImplementedError('Base Class')



    def test_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        test_loss = 0
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
        if epoch != 0 and (epoch % self.checkepoch == 0) and (self.stepsize>0.0001):
            self.stepsize = self.stepsize*np.exp(-self.k*epoch)
        self.ts_acc_ls.append(100.*correct/total)
        self.test_acc = 100.*correct/total



    def test(self, epochs, model):
        print(epochs)
        for epoch in range(self.startepoch, epochs+1):
            self.test_epoch(epoch) 
            if self.atmeth in ['SSDS', 'NOLAG', 'NOLAM'] and (epoch)%self.checkepoch == 0:
                self.save_log(epoch)
                self.plot_log()

            if epoch != 0 and 0<(self.ts_acc_ls[epoch]-self.ts_acc_ls[epoch-1]) < self.pres:
                break
        return self.test_acc


    def save_log(self, epoch):
        self.log['epoch'] = epoch
        torch.save(self.log, '%s/checkpoint/testlog.pkl'%(self.expid))

    def plot_log(self):
        if not os.path.isdir('%s.test_results'%(self.expid)):
            os.mkdir('%s.test_results'%(self.expid))
        PlotAcc(self.ts_acc_ls,'%s.test_results/TestAcc.pdf'%(self.expid))