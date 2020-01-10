import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from utils import to_var
import numpy as np

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attacker(object):
    def __init__(self, eps, model, stepsize, optimizer, criterion, **kwargs):
        self.eps = eps
        self.model = model
        self.stepsize = stepsize
        self.optimizer = optimizer
        self.criterion = criterion
        self.c_1 = kwargs.get('c_1')
        self.c_2 = kwargs.get('c_2')
        self.lam = kwargs.get('lam')
        self.nstep = kwargs.get('nstep')
        self.v = kwargs.get('v_scale')       

    ############################################### PGD ############################################
    def PGDattack(self, X_nat, y, Optimizer):
        randi=np.random.uniform(-self.eps, self.eps, X_nat.shape).astype('float32')
        X = X_nat +randi
        X = np.clip(X, 0, 1) 
        for i in range(self.nstep):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))
            scores = self.model(X_var)
            Optimizer.zero_grad()
            loss = self.criterion(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            per = self.stepsize*np.sign(grad)
            X += per
            X = np.clip(X,X_nat-self.eps, X_nat+self.eps)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X, per

    ############################################### FGSM ############################################
    def FGSMattack(self, X_nat, y, Optimizer):
        X = np.copy(X_nat) 
        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))
        scores = self.model(X_var)
        Optimizer.zero_grad()
        loss = self.criterion(scores, y_var)
        loss.backward()
        grad = X_var.grad.data.cpu().numpy()
        per = self.eps*np.sign(grad)
        X += per
        X = np.clip(X, 0, 1) # ensure valid pixel range
        return X

    ############################################### SSDS-p ############################################
    def SSDSattack(self, X_nat, y, delta, v, t, lam, Optimizer):
        #it gets v and delta for batch as inputs
        Optimizer.zero_grad()
        pert = X_nat+delta
        pert = torch.clamp(pert,0,1)
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
        v = new_v
        ############ Update t #############
        t=t+self.stepsize*(lam-1)

        ############# Lambda Update ############
        normlam = (self.stepsize/self.c_2)*(loss.item())-t-(lamsum.item())
        if lam+normlam <0:
            lam = 0
        else:
            lam = lam+normlam
        
        return pert.detach(), new_delta,  v, lam, t 

    ############################################### NOLAM ############################################
    def NOLAMattack(self, X_nat, y, delta, v, Optimizer):
        #it gets v and delta for batch as inputs
        pert = X_nat+delta
        pert = torch.clamp(pert,0,1)
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
        v = new_v

        return pert.detach(), new_delta, v

    ############################################### NOLAG ############################################
    def NOLAGattack(self, X_nat, y, delta, Optimizer):
        #it gets v and delta for batch as inputs
        pert = X_nat+delta
        pert = torch.clamp(pert,0,1)
        Optimizer.zero_grad()
        outputs = self.model(pert) 
        loss = self.criterion(outputs,y)
        loss.backward()
        grad = delta.grad.data.clone().detach()
        ########## Update Delta ############
        new_delta = (delta + self.stepsize*(grad)).detach().cpu()
        new_delta = torch.clamp(new_delta,-self.eps,self.eps)
        
        return pert.detach(), new_delta
