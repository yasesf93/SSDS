import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
import numpy as np
from Visualizations import PlotLoss,PlotAcc

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseTrainer(object):
    def __init__(self, model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, pres, stepsize, k, **kwargs):
        self.model = model
        self.traindataloader = traindataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.classes = classes
        self.batchsizetr = trainbatchsize
        self.best_acc = 0
        self.startepoch = 0
        self.n_epoch = n_epoch
        self.expid = expid
        self.checkepoch = checkepoch
        self.pres = pres
        self.stepsize = stepsize
        self.k = k
        self.log = {}
        self.log['train_loss'] = []
        self.log['train_acc'] = []   
        self.log['epoch'] = self.startepoch
             

    def train_minibatch(self, batch_idx):
        raise NotImplementedError('Base Class')



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
        if epoch != 0 and (epoch % self.checkepoch == 0) and (self.stepsize>0.0001):
            self.stepsize = self.stepsize*np.exp(-self.k*epoch)
        self.log['train_loss'].append(loss_avg)
        self.log['train_acc'].append(100.*correct/total)

        #Save checkpoint Best
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = { 'net': self.model.state_dict(),'acc': acc, 'epoch': epoch,}
            if not os.path.isdir('%s'%(self.expid)):
                os.mkdir('%s'%(self.expid))
            if not os.path.isdir('%s/checkpoint'%(self.expid)):
                os.mkdir('%s/checkpoint'%(self.expid))
            torch.save(state, '%s/checkpoint/ckpt.trainbest'%(self.expid))
            best_acc = acc

        #Save checkpoint last
        acc = 100.*correct/total
        if epoch == (self.n_epoch)-1:
            print('Saving..')
            state = { 'net': self.model.state_dict(),'acc': acc, 'epoch': epoch,}
            if not os.path.isdir('%s'%(self.expid)):
                os.mkdir('%s'%(self.expid))
            if not os.path.isdir('%s/checkpoint'%(self.expid)):
                os.mkdir('%s/checkpoint'%(self.expid))
            torch.save(state, '%s/checkpoint/ckpt.trainlast'%(self.expid))
            best_acc = acc


    def train(self, epochs, model):
        model.train()
        print(epochs)
        for epoch in range(self.startepoch, epochs+1):
            self.train_epoch(epoch) 
            if (epoch)%self.checkepoch == 0:
                self.save_log(epoch)
                self.plot_log()
                
            if epoch != 0 and 0<(self.log['train_acc'][epoch]-self.log['train_acc'][epoch-1]) < self.pres:
                self.save_log(epoch)
                self.plot_log()
                break


    def save_log(self, epoch):
        self.log['epoch'] = epoch
        torch.save(self.log,'%s/checkpoint/trainlog.pkl'%(self.expid))
    

    def plot_log(self):
        if not os.path.isdir('%s/train_results'%(self.expid)):
            os.mkdir('%s/train_results'%(self.expid))
        PlotLoss(self.log['train_loss'],'%s/train_results/TrainLoss.pdf'%(self.expid))
        PlotAcc(self.log['train_acc'],'%s/train_results/TrainAcc.pdf'%(self.expid))
    