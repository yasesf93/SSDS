import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from .BaseTrainer import BaseTrainer 
from Attacker import Attacker

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RegTrainer(BaseTrainer):
    def __init__(self, model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, atmeth, **kwargs):
        super().__init__(model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, **kwargs)
        self.atmeth = atmeth
        self.pert = []
        self.best_acc = 0
        self.startepoch = 0
        self.eps = kwargs.get('epsilon')
        self.nstep = kwargs.get('nstep')
        self.stepsize = kwargs.get('stepsize')
        self.k = kwargs.get('k')
        
    def train_minibatch(self, batch_idx):
        attacker = Attacker(self.atmeth, self.eps, self.model, self.stepsize, self.k, self.traindataloader, self.batchsizetr, self.optimizer, self.criterion, self.classes, self.n_epoch, self.expid, self.checkepoch,nstep=self.nstep)
        (I, targets) = self.traindataloader[batch_idx]
        I, targets = I.to(device), targets.to(device)
        targets=targets.long()

        if self.atmeth == "PGD":
            I , self.pert = attacker.PGDattack(I.cpu().numpy(),targets.cpu(),self.optimizer)
            I = torch.from_numpy(I)
            I = I.to(device)
    
        if self.atmeth == "FGSM":
            I = attacker.FGSMattack(I.cpu().numpy(),targets.cpu(),self.optimizer)
            I = torch.from_numpy(I)
            I = I.to(device)

        self.optimizer.zero_grad()
        outputs = self.model(I)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _,predicted = outputs.max(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)

