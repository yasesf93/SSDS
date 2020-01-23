import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from .BaseTrainer import BaseTrainer 
from Attacker import Attacker
from Loss.trades import trades_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RegTrainer(BaseTrainer):
    def __init__(self, model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, pres, stepsize, k, atmeth, c_1, c_2, eps, dataname, nstep, beta, **kwargs):
        super().__init__(model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, pres, stepsize, k, atmeth, c_1, c_2, eps, dataname, nstep, **kwargs)
        self.beta = beta
        self.pert = []
        self.best_acc = 0
        self.startepoch = 0
        self.attacker = Attacker(self.eps, self.model, self.stepsize, self.optimizer, self.criterion, self.c_1, self.c_2, self.nstep, self.dataname)


    def train_minibatch(self, batch_idx):
        (I, _), targets = self.traindataloader[batch_idx]
        I, targets = I.to(device), targets.to(device)
        targets = targets.long()

        if self.atmeth == "PGD":
            I , self.pert = self.attacker.PGDattack(I.cpu().numpy(),targets.cpu(),self.optimizer)
            I = torch.from_numpy(I)
            I = I.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(I)
            loss = self.criterion(outputs, targets)
    
        if self.atmeth == "FGSM":
            I = self.attacker.FGSMattack(I.cpu().numpy(),targets.cpu(),self.optimizer)
            I = torch.from_numpy(I)
            I = I.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(I)
            loss = self.criterion(outputs, targets)

        if self.atmeth == "REG":
            self.optimizer.zero_grad()
            outputs = self.model(I)
            loss = self.criterion(outputs, targets)
        
        if self.atmeth == 'TRADES':
            self.optimizer.zero_grad()
            outputs = self.model(I)
            self.optimizer.zero_grad()
            loss = trades_loss(model=self.model, x_natural=I, y=targets,optimizer=self.optimizer, step_size=self.stepsize, epsilon=self.eps, perturb_steps=self.nstep, beta=self.beta)

        loss.backward()
        self.optimizer.step()
        _,predicted = outputs.max(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)
