import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
import numpy as np
from .BaseTester import BaseTester 
from Attacker import Attacker
from torch.autograd import Variable





device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RegTester(BaseTester):
    def __init__(self, model, testdataloader, optimizer, criterion, n_epoch, testbatchsize, expid, checkepoch, pres, stepsize, k, atmeth, c_1, c_2, eps, dataname, nstep,criterion_att, **kwargs):
        super().__init__(model, testdataloader, optimizer, criterion, n_epoch, testbatchsize, expid, checkepoch, pres, stepsize, k, atmeth, c_1, c_2, eps, dataname, nstep, **kwargs)
        self.pert = []
        self.best_acc = 0
        self.startepoch = 0
        self.criterion_att = criterion_att
        self.attacker = Attacker(self.eps, self.model, self.stepsize, self.optimizer, self.criterion_att, self.c_1, self.c_2, self.nstep, self.dataname)
    def test_minibatch(self, batch_idx):
        (I, _), targets = self.testdataloader[batch_idx]
        I, targets = I.to(device), targets.to(device)
        targets = targets.long()
        if self.atmeth == 'REG':
            I = Variable(torch.clamp(I, 0, 1.0))
            I = I.to(device)

        if self.atmeth == "PGD":
            I, targets = Variable(I, requires_grad=True), Variable(targets)
            I , self.pert = self.attacker.PGDattack(I,targets,self.optimizer)
            I = I.to(device)
    
        if self.atmeth == "FGSM":
            I, targets = Variable(I, requires_grad=True), Variable(targets)
            I = self.attacker.FGSMattack(I,targets,self.optimizer)
            I = I.to(device)

        self.optimizer.zero_grad()
        outputs = self.model(I)
        loss = self.criterion(outputs, targets)
        _,predicted = outputs.max(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)

