import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from utils import to_var
import numpy as np
from torch.autograd import Variable



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attacker(object):
    def __init__(self, eps, model, stepsize, optimizer, criterion,c_1,c_2,nstep,dataname):
        self.eps = eps
        self.model = model
        self.stepsize = stepsize
        self.optimizer = optimizer
        self.criterion = criterion
        self.c_1 = c_1
        self.c_2 = c_2
        self.nstep = nstep
        self.dataname = dataname

    ############################################### PGD ############################################
    def PGDattack(self, X_nat, y, Optimizer):
        random_noise = torch.FloatTensor(*X_nat.shape).uniform_(-self.eps, self.eps).to(device)
        X = Variable(X_nat.data + random_noise, requires_grad=True)
        for i in range(self.nstep):
            Optimizer.zero_grad()
            with torch.enable_grad():
                scores = self.model(X)
                loss = self.criterion(scores, y)
            loss.backward()
            grad = X.grad.data.sign()
            per = self.stepsize*grad
            X = Variable(X.data + per, requires_grad=True)
            per = torch.clamp(X.data - X_nat.data, -self.eps, self.eps)
            X = Variable(X_nat.data + per, requires_grad=True)
            X = Variable(torch.clamp(X, 0, 1.0), requires_grad=True)
        return X, per

    ############################################### FGSM ############################################
    def FGSMattack(self, X_nat, y, Optimizer):
        X = X_nat 
        Optimizer.zero_grad()
        with torch.enable_grad():
            scores = self.model(X)
            loss = self.criterion(scores, y)  
        loss.backward()
        grad = X.grad.data.sign()
        per = self.eps*grad
        X = Variable(X.data + per, requires_grad=True)
        X = Variable(torch.clamp(X, 0, 1.0), requires_grad=True)
        return X

    ############################################### SSDS-p ############################################
    def SSDSattack(self, X_nat, y, delta, v, t, lam, Optimizer):
        #it gets v and delta for batch as inputs
        Optimizer.zero_grad()
        if self.dataname in ['MNIST', 'FashionMNIST']:
            rand_i = torch.from_numpy(np.random.uniform(low=-self.eps, high=self.eps, size=X_nat.size())).to(device) 
            rand_i = rand_i.float()
            pert = X_nat + rand_i
            pert = pert+delta
            pert = torch.where(pert>X_nat+self.eps,X_nat+self.eps,pert)
            pert = torch.where(pert<X_nat-self.eps,X_nat-self.eps,pert)
        else:
            pert = X_nat+delta
        pert = torch.clamp(pert,0,1)
        randpert = pert-X_nat
        outputs = self.model(pert) 
        loss = self.criterion(outputs,y)
        loss.backward()
        grad = delta.grad.data.clone().detach()
        s = torch.sign(delta)
        lag = v[:,None,None,None]*s/self.c_1
        ########## Update Delta ############
        new_delta = (delta + self.stepsize*(grad - lag)).detach().cpu()
        new_delta=torch.clamp(new_delta,-self.eps,self.eps)
        ############# Update v ##############
        normcheck = self.stepsize*(delta.norm(p=float('inf'),dim=(1,2,3)) - self.eps)
        new_v = v.clone().detach()
        new_v[v+normcheck<0] = 0
        new_v[v+normcheck>=0] = v[v+normcheck>=0] + normcheck[v+normcheck>=0]
        lamsum = (new_v*(delta.norm(p=float('inf'),dim=(1,2,3)) - self.eps)).sum()
        v = new_v.clone().detach()
        del new_v, lag, normcheck
        ############ Update t #############
        t=t+self.stepsize*(lam-1)

        ############# Lambda Update ############
        normlam = (self.stepsize/self.c_2)*(loss.item())-t-(lamsum.item())
        if lam+normlam <0:
            lam = 0
        else:
            lam = lam+normlam
        
        return pert.detach(), pert-X_nat.detach(), new_delta,  v, lam, t 

    ############################################### NOLAM ############################################
    def NOLAMattack(self, X_nat, y, delta, v, Optimizer):
        #it gets v and delta for batch as inputs
        if self.dataname in ['MNIST', 'FashionMNIST']:
            rand_i = torch.from_numpy(np.random.uniform(low=-self.eps, high=self.eps, size=X_nat.size())).to(device) 
            rand_i = rand_i.float()
            pert = X_nat + rand_i
            pert = pert+delta
            pert = torch.where(pert>X_nat+self.eps,X_nat+self.eps,pert)
            pert = torch.where(pert<X_nat-self.eps,X_nat-self.eps,pert)
        else:
            pert = X_nat+delta
        pert = torch.clamp(pert,0,1)
        randpert = pert-X_nat
        Optimizer.zero_grad()
        outputs = self.model(pert) 
        loss = self.criterion(outputs,y)
        loss.backward()
        grad = delta.grad.data.clone().detach()
        s = torch.sign(delta)
        lag = v[:,None,None,None]*s/self.c_1
        ########## Update Delta ############
        new_delta = (delta + self.stepsize*(grad - lag)).detach().cpu()
        new_delta=torch.clamp(new_delta,-self.eps,self.eps)
        ############# Update v ##############
        normcheck = self.stepsize*(delta.norm(p=float('inf'),dim=(1,2,3)) - self.eps)
        new_v = v.clone().detach()
        new_v[v+normcheck<0] = 0
        new_v[v+normcheck>=0] = v[v+normcheck>=0] + normcheck[v+normcheck>=0]
        v = new_v.clone().detach()
        del new_v, lag, normcheck

        return pert.detach(), pert-X_nat.detach(), new_delta, v

    ############################################### NOLAG ############################################
    def NOLAGattack(self, X_nat, y, delta, Optimizer):
        #it gets v and delta for batch as inputs
        if self.dataname in ['MNIST', 'FashionMNIST']:
            rand_i = torch.from_numpy(np.random.uniform(low=-self.eps, high=self.eps, size=X_nat.size())).to(device) 
            rand_i = rand_i.float()
            pert = X_nat + rand_i
            pert = pert+delta
            pert = torch.where(pert>X_nat+self.eps,X_nat+self.eps,pert)
            pert = torch.where(pert<X_nat-self.eps,X_nat-self.eps,pert)
        else:
            pert = X_nat+delta
        pert = torch.clamp(pert,0,1)
        randpert = pert-X_nat
        Optimizer.zero_grad()
        outputs = self.model(pert) 
        loss = self.criterion(outputs,y)
        loss.backward()
        grad = delta.grad.data.clone().detach()
        ########## Update Delta ############
        new_delta = (delta + self.stepsize*(grad)).detach().cpu()
        new_delta = torch.clamp(new_delta,-self.eps,self.eps)
        
        return pert.detach(), pert-X_nat.detach(), new_delta
