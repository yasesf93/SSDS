import json
import os
import copy

with open('config.json', 'r') as f:
    config = json.load(f)

optimizers = ['SubOpt', 'SubOptMOM', 'SGD', 'SGDMOM', 'SGD', 'SGDMOM', 'SGD', 'SGDMOM', 'SGD', 'SGDMOM', 'SGD', 'SGDMOM']
attack_method = ['SSDS', 'SSDS','NOLAM','NOLAM', 'NOLAG', 'NOLAG', 'PGD', 'PGD', 'FGSM', 'FGSM', 'REG', 'REG']
exp = 1
for model in ['Resnet50', 'WResnet']:
   for opt, atmeth in zip(optimizers, attack_method):
        cfg = copy.deepcopy(config)
        cfg['data_name'] = 'CIFAR10'
        cfg['attack_method'] = atmeth
        cfg['optimizer'] = opt
        cfg['model_architecture'] = model
        cfg['training_epochs'] = 500 
        with open('experiment%s.json'%exp,'w') as f:
            json.dump(cfg, f, indent=4)
        exp += 1
