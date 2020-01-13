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
    def __init__(self, model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, pres, stepsize, k, atmeth, v, t, lam, c_1, c_2, eps, **kwargs):
        super().__init__(model, traindataloader, optimizer, criterion, classes, n_epoch, trainbatchsize, expid, checkepoch, pres, stepsize, k, **kwargs)
        self.atmeth = atmeth
        self.v = v
        self.t = t 
        self.lam = lam
        self.c_1 = c_1
        self.c_2 = c_2
        self.best_acc = 0
        self.startepoch = 0
        self.eps = eps
        # self.differ = torch.zeros(self.traindataloader.dataset.delta.size())
        self.attacker = Attacker(self.eps, self.model, self.stepsize, self.optimizer, self.criterion, c_1=self.c_1, c_2=self.c_2, lam=self.lam)
        self.log['train_accuracy'] = []
        self.log['train_loss'] = []
        self.log['train_lambda'] = []
        self.log['train_t'] = []

        self.log['train_spec_delt_val_log'] = {}
        self.log['train_spec_delt_val_log']['id'] = [40000,1,16,16] # a value that works for all the datasets
        self.log['train_spec_delt_val_log']['val'] = [self.traindataloader.dataset[self.log['train_spec_delt_val_log']['id'][0]][1]]

        self.log['train_spec_img_log'] = {}
        self.log['train_spec_img_log']['ids'] = [10, 15000, 49990] # a value that works for all the datasets

        self.log['train_spec_img_log']['v'] = [[] for _ in range(len(self.log['train_spec_img_log']['ids']))]
        self.log['train_spec_img_log']['infnormdelta'] = [[] for _ in range(len(self.log['train_spec_img_log']['ids']))]
        self.log['train_spec_img_log']['2normdelt'] = [[] for _ in range(len(self.log['train_spec_img_log']['ids']))]

        self.log['train_spec_img_log']['2normdiff'] = [[] for _ in range(len(self.log['train_spec_img_log']['ids']))]
        self.log['train_spec_img_log']['1normdiff'] = [[] for _ in range(len(self.log['train_spec_img_log']['ids']))]
        
        self.log['train_img_vis_log'] = {}
        self.log['train_img_vis_log']['ids'] = [10, 15000, 49990, 100, 1000, 10000, 25000, 40000] # a value that works for all the datasets
        self.log['train_img_vis_log']['img_tuple'] = [[] for _ in range(len(self.log['train_img_vis_log']['ids']))]

    def train_minibatch(self, batch_idx):
        (I, delta), targets = self.traindataloader[batch_idx]
        I, targets = I.to(device), targets.to(device)
        targets = targets.long()
        delta = delta.to(device)
        delta = Variable(delta, requires_grad = True)
        indexes = self.traindataloader.indexes[batch_idx*self.batchsizetr:(batch_idx+1)*self.batchsizetr]
        v_batch = self.v[indexes].squeeze().to(device)
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
        _, predicted = outputs.max(1)
        # print(indexes)
        # self.differ[indexes] = (new_delta - self.traindataloader.dataset[indexes][1].cpu())
        for i, idx in enumerate(indexes):
            self.traindataloader.dataset[idx] = new_delta[i]
        if self.atmeth == 'SSDS' or self.atmeth == 'NOLAM':
            self.v[indexes] = new_v.unsqueeze(1).detach().cpu()
        del new_delta, X, 
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)


    def train_epoch(self, epoch):
        super(DelTrainer, self).train_epoch(epoch)       

        if self.atmeth is 'SSDS':
            self.log['train_t'].append(self.t)
            self.log['train_lambda'].append(self.lam)

        self.log['train_spec_delt_val_log']['val'].append(self.traindataloader.dataset.delta[self.log['train_spec_delt_val_log']['id']])

        for idx, img_id in enumerate(self.log['train_spec_img_log']['ids']):
            self.log['train_spec_img_log']['infnormdelta'][idx].append(self.traindataloader.dataset.delta[img_id].norm(p=float("inf")).item())
            self.log['train_spec_img_log']['2normdelt'][idx].append(self.traindataloader.dataset.delta[img_id].norm(p=2).item())
            self.log['train_spec_img_log']['2normdiff'][idx].append(self.differ[img_id].norm(p=2).item())
            self.log['train_spec_img_log']['1normdiff'][idx].append(self.differ[img_id].norm(p=1).item())
            if self.atmeth in ['NOLAM', 'SSDS']:
                self.log['train_spec_img_log']['v'][idx].append(self.v[img_id].item())

        
        for idx, img_id in enumerate(self.log['train_img_vis_log']['ids']):
            self.log['train_img_vis_log']['img_tuple'][idx].append(self.traindataloader.dataset[img_id])
        
        #print functions
        print('attack method', self.atmeth)
        print('infinity norm of delta value', self.traindataloader.dataset.delta[15000].norm(p=float("inf")).item())
        if self.atmeth is 'SSDS':
            print('lambda', self.lam)
        if self.atmeth in ['NOLAM', 'SSDS']:
            print('v', self.v[15000].item())
        print('step-size', self.stepsize)
        print('L-2 norm of delta difference', self.differ[15000].norm(p=2).item())