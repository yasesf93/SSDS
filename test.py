import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
from Trainer import Trainer
import json
import os
import random
import numpy as np


with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)

os.environ['CUDA_VISIBLE_DEVICE'] = '3'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################################## using a specific seed ########################################
seed_num= config['random_seed']
def seed_everything(seed=seed_num):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()


######################################################## setting up the hyperparameters ########################################

lr = config['learning_rate_training']
wd = config['weight_decay']
momentum = config['momentum']
n_epoch = config['training_epochs']
batchsizetr = config['training_batch_size']
batchsizets = config['test_batch_size']
dataname = config['data_name']
loss = config['loss_function']
opt = config['optimizer']
model = config['model_architecture']
advtr = config['adversarial_training']


######################################################## Transformation ########################################
if config['transform']==True:
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])  
else:
    transform_train=None
    transform_test=None



######################################################## Data translation ########################################
if dataname=="CIFAR10":
    if advtr==False
        trainset = Datasets.cifar10clean(root='./data', train=True, download=True, transform=transform_train)
        trainloader = Dataloaders.CleanDataLoader(trainset, batch_size=batchsizetr, shuffle=True) 
        #testset = Datasets.cifar10clean(root='./data', train=False, download=True, transform=transform_test)
        #testloader = Dataloaders.CleanDataLoader(testset, batch_size=batchsizets, shuffle=True) 
    else:
        trainset = Datasets.cifar10adv(root='./data', train=True, download=True, transform=transform_train)
        trainloader = Dataloaders.CleanDataLoader(trainset, batch_size=batchsizetr, shuffle=True) 
        #testset = Datasets.cifar10adv(root='./data', train=False, download=True, transform=transform_test)
        #testloader = Dataloaders.CleanDataLoader(testset, batch_size=batchsizets, shuffle=True) 

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if dataname=="MNIST":


if dataname=="ImageNet":



######################################################## Models ########################################


if model=="Resnet_50":
    net = Models.ResNet50()

if model = "WideResnet_50":

######################################################## Loss Function ########################################

if loss == 'Xent':
    criterion = nn.CrossEntropyLoss()


if loss == 'CW':


if loss == 'TRADES':


######################################################## Optimizers ########################################

if opt =='sgd':
    optimizer = optim.SGD(net.parameters(), lr=lr)


if opt == 'adam':


if opt == 'momentum':



trainer = Trainer(net, trainloader, testloader, optimizer, criterion, classes, n_epoch, device)
trainer.train(epochs=n_epoch)