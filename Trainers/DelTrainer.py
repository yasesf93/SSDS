import sys
sys.path.append("..")
import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from .BaseTrainer import BaseTrainer 
from Attacker import Attacker
from torch.autograd import Variable

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DelTrainer(BaseTrainer):
    def __init__(self, model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, atmeth, v, t, lam, c_1, c_2, eps, stepsize, k, **kwargs):
        super().__init__(model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, **kwargs)
        self.atmeth = atmeth
        self.v = v
        self.t = t 
        self.lam = lam
        self.c_1 = c_1
        self.c_2 = c_2
        self.best_acc = 0
        self.startepoch = 0
        self.eps = eps
        self.stepsize = stepsize
        self.k = k
        self.attacker = Attacker(self.atmeth, self.eps, self.model, self.stepsize, self.k, self.traindataloader, self.batchsizetr, self.optimizer, self.criterion, self.classes, self.n_epoch, self.expid, self.checkepoch, c_1=self.c_1, c_2=self.c_2, lam=self.lam)
        self.log['train_accuracy'] = {}
        self.log['train_loss'] = {}
        self.log['train_lambda'] = {}
        self.log['train_v'] = {}
        

    def train_minibatch(self, batch_idx):
        
        (I,delta), targets = self.traindataloader[batch_idx]
        I, targets = I.to(device), targets.to(device)
        targets = targets.long()
        delta = Variable(delta.to(device), requires_grad = True)
        indexes = self.traindataloader.indexes[batch_idx*self.batchsizetr:(batch_idx+1)*self.batchsizetr]
        self.v = self.v.to(device)
        v_batch = self.v[indexes].squeeze()
        if self.atmeth == 'SSDS':
            X, new_delta, new_v, new_lam, new_t = self.attacker.SSDSattack(I, targets, delta, v_batch, self.t, self.lam, self.optimizer)
            self.lam = new_lam
            self.t = new_t   
        if self.atmeth == 'NOLAM':
            X, new_delta, new_v = self.attacker.NOLAMattack(I, targets, delta, v_batch, self.optimizer)
        if self.atmeth == 'NOLAG':
            X, new_delta, = self.attacker.NOLAGattack(I, targets, delta, self.optimizer)
        self.optimizer.zero_grad()
        outputs = self.model(X)
        targets = targets.long()
        loss = self.criterion(outputs, targets)
        loss.backward()
        if self.optimizer == 'SubOpt':
            self.optimizer.step(self.lam)
        else:
            self.optimizer.step()        
        _,predicted = outputs.max(1)
        self.traindataloader.dataset.delta[indexes] = new_delta
        if self.atmeth == 'SSDS' or self.atmeth == 'NOLAM':
            self.v[indexes] = new_v.unsqueeze(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)
