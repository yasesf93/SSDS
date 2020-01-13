import sys
sys.path.append("..")
import torch.nn as nn
import Dataloaders
from torch.optim.optimizer import Optimizer
from utils import progress_bar
import torch
import os
import json
from .BaseTester import BaseTester 
from Attacker import Attacker
from torch.autograd import Variable
import numpy as np
from Visualizations import PlotVal, PlotHist, PlotImg

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DelTester(BaseTester):
    def __init__(self, model, testdataloader, optimizer, criterion, classes, n_epoch, testbatchsize, expid, checkepoch, pres,atmeth, v, t, lam, c_1, c_2, eps, stepsize, k, **kwargs):
        super().__init__(model, testdataloader, optimizer, criterion, classes, n_epoch, testbatchsize, expid, checkepoch, pres, atmeth, **kwargs)
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
        self.differ = torch.zeros(self.testdataloader.dataset.delta.size())
        self.attacker = Attacker(self.eps, self.model, self.stepsize, self.optimizer, self.criterion, c_1=self.c_1, c_2=self.c_2, lam=self.lam)
        self.log['test_accuracy'] = []
        self.log['test_loss'] = []
        self.log['test_lambda'] = []
        self.log['test_t'] = []
        self.log['fnorm_test'] = []

        self.log['test_spec_delt_val_log'] = {}
        self.log['test_spec_delt_val_log']['id'] = [7000,1,16,16] # a value that works for all the datasets
        self.log['test_spec_delt_val_log']['val'] = [self.testdataloader.dataset.delta[self.log['test_spec_delt_val_log']['id']]]
        
        self.log['test_spec_img_log'] = {}
        self.log['test_spec_img_log']['ids'] = [10, 5000, 9990] # a value that works for all the datasets

        self.log['test_spec_img_log']['v'] = [[] for _ in range(len(self.log['test_spec_img_log']['ids']))]
        self.log['test_spec_img_log']['infnormdelta'] = [[] for _ in range(len(self.log['test_spec_img_log']['ids']))]
        self.log['test_spec_img_log']['2normdelt'] = [[] for _ in range(len(self.log['test_spec_img_log']['ids']))]

        self.log['test_spec_img_log']['2normdiff'] = [[] for _ in range(len(self.log['test_spec_img_log']['ids']))]
        self.log['test_spec_img_log']['1normdiff'] = [[] for _ in range(len(self.log['test_spec_img_log']['ids']))]
        
        self.log['test_img_vis_log'] = {}
        self.log['test_img_vis_log']['ids'] = [10, 5000, 9990, 100, 1000, 8000, 2500, 4000] # a value that works for all the datasets
        self.log['test_img_vis_log']['img_tuple'] = [[] for _ in range(len(self.log['test_img_vis_log']['ids']))]

    def test_minibatch(self, batch_idx):
        
        (I,delta), targets = self.testdataloader[batch_idx]
        I, targets = I.to(device), targets.to(device)
        targets = targets.long()
        delta = Variable(delta.to(device), requires_grad = True)
        indexes = self.testdataloader.indexes[batch_idx*self.batchsizets:(batch_idx+1)*self.batchsizets]
        self.v = self.v.to(device)
        v_batch = self.v[indexes].squeeze()
        if self.atmeth == 'SSDS':
            X, new_delta, new_v, new_lam, new_t = self.attacker.SSDSattack(I, targets, delta, v_batch, self.t, self.lam, self.optimizer)  
        if self.atmeth == 'NOLAM':
            X, new_delta, new_v = self.attacker.NOLAMattack(I, targets, delta, v_batch, self.optimizer)
        if self.atmeth == 'NOLAG':
            X, new_delta, = self.attacker.NOLAGattack(I, targets, delta, self.optimizer)
        self.optimizer.zero_grad()
        outputs = self.model(X)
        targets = targets.long()
        loss = self.criterion(outputs, targets)      
        _,predicted = outputs.max(1)
        self.differ[indexes] = (new_delta - self.testdataloader.dataset.delta[indexes].cpu())
        self.testdataloader.dataset.delta[indexes] = new_delta
        if self.atmeth == 'SSDS' or self.atmeth == 'NOLAM':
            self.v[indexes] = new_v.unsqueeze(1)
        return loss.item(), predicted.eq(targets).sum().item(), targets.size(0)


    def test_epoch(self, epoch):
        super(DelTester, self).test_epoch(epoch)
        self.log['fnorm_test'].append(torch.zeros(len(self.testdataloader.dataset),1))
        for z in range (self.testdataloader.dataset.delta.shape[0]):
            self.log['fnorm'][-1][z]=self.testdataloader.dataset.delta[z].norm(p=float("inf"))
            
        if self.atmeth is 'SSDS':
            self.log['test_t'].append(self.t)
            self.log['test_lambda'].append(self.lam)

        self.log['test_spec_delt_val_log']['val'].append(self.testdataloader.dataset.delta[self.log['test_spec_delt_val_log']['id']])

        for idx, img_id in enumerate(self.log['test_spec_img_log']['ids']):
            self.log['test_spec_img_log']['infnormdelta'][idx].append(self.testdataloader.dataset.delta[img_id].norm(p=float("inf")).item())
            self.log['test_spec_img_log']['2normdelt'][idx].append(self.testdataloader.dataset.delta[img_id].norm(p=2).item())
            self.log['test_spec_img_log']['2normdiff'][idx].append(self.differ[img_id].norm(p=2).item())
            self.log['test_spec_img_log']['1normdiff'][idx].append(self.differ[img_id].norm(p=1).item())
            if self.atmeth in ['NOLAM', 'SSDS']:
                self.log['test_spec_img_log']['v'][idx].append(self.v[img_id].item())

        
        for idx, img_id in enumerate(self.log['test_img_vis_log']['ids']):
            self.log['test_img_vis_log']['img_tuple'][idx].append(self.testdataloader.dataset[img_id])
        

        #print functions
        print('attack method', self.atmeth)
        print('infinity norm of delta value', self.testdataloader.dataset.delta[5000].norm(p=float("inf")).item())
        if self.atmeth is 'SSDS':
            print('lambda', self.lam)
        if self.atmeth in ['NOLAM', 'SSDS']:
            print('v', self.v[5000].item())
        print('step-size', self.stepsize)
        print('L-2 norm of delta difference', self.differ[5000].norm(p=2).item())


    def plot_log(self):
        super(DelTester, self).plot_log()
        deltls = [delt[0, 0, 16, 16].item() for delt in self.log['test_spec_delt_val_log']['val']]
        PlotVal(deltls, r'$\delta$', '%s.test_results/test_specific_delta_pixel.pdf'%(self.expid), hline=self.eps) #plot delta pixel
        PlotHist([self.log['fnorm_test'][1].numpy().flatten(), self.log['fnorm_test'][-1].numpy().flatten()], ['first', 'last'], 'fnorm_test',
            '%s.test_results/fnorm_hist.pdf'%(self.expid), vline=self.eps)
        
        for idx, img_id in enumerate(self.log['test_spec_img_log']['ids']): #plot v, delta and delta difference for each image
            PlotVal(self.log['test_spec_img_log']['infnormdelta'][idx], r'$|\delta|_{\infty}$', '%s.test_results/infnorm_delta_%s.pdf'%(self.expid, img_id), hline=self.eps)
            PlotVal(self.log['test_spec_img_log']['2normdelt'][idx], r'$|\delta|_{2}$', '%s.test_results/2norm_delta_%s.pdf'%(self.expid, img_id))
            PlotVal(self.log['test_spec_img_log']['2normdiff'][idx], r'$|\delta_{k+1} - \delta_{k}|_{2}$', '%s.test_results/2norm_diff_%s.pdf'%(self.expid, img_id))
            PlotVal(self.log['test_spec_img_log']['1normdiff'][idx], r'$|\delta_{k+1} - \delta_{k}|$', '%s.test_results/1norm_diff_%s.pdf'%(self.expid, img_id))
            if self.atmeth in ['NOLAM', 'SSDS']:
                PlotVal(self.log['test_spec_img_log']['v'][idx], 'v', '%s.test_results/v_%s.pdf'%(self.expid, img_id), hline=1.0)  
        

        for idx, img_id in enumerate(self.log['test_img_vis_log']['ids']): #Plot Images
            (I, delta), targets = self.log['test_img_vis_log']['img_tuple'][idx][-1]
            pert = I + delta
            pert = torch.clamp(pert,0,1)
            pert = pert.unsqueeze(0)
            pert = pert.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(pert)
            _, predicted = outputs.max(1)
            PlotImg(I.squeeze().cpu().numpy(), delta.squeeze().cpu().numpy(), pert.squeeze().cpu().numpy(), self.classes[targets], self.classes[predicted],
                '%s.test_results/img_%s.pdf'%(self.expid, img_id))