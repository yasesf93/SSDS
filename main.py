import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import Trainers
import json
import os
import random
import numpy as np
import Datasets
import Dataloaders 
import Models
import Optimizers
from Loss.trades import trades_loss
import Testers

with open('config.json') as config_file: # Reading the Config File 
    config = json.load(config_file)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################################## using a specific seed ########################################
seed_num = config['random_seed']
def seed_everything(seed=seed_num):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()


######################################################## setting up the hyperparameters ########################################
training = config['train']
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
atmeth = config['attack_method']
checkepoch = config['checkpoint_epochs']
pres = config['precision_bound']


eps = config['epsilon']
nstep = config['num_steps']
stepsize = config['step_size']
v_scale = config['v']
lam = config['lambda']
k = config['step_size_decay']
c_1 = config['c_1']
c_2 = config['c_2']
t = config['t']
n_ep_PGD = config['PGD_Restarts']

if not os.path.exists('Experiments'):
    os.makedirs('Experiments')
expid = 'Experiments/%s_%s_%s_%s_%s'%(str(atmeth), str(opt), str(loss), str(dataname), str(model))
print('expid',expid)
######################################################## Transformation ########################################
if config['transform']==True:
    if dataname == "CIFAR10":
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
    elif dataname == "IMAGENET":
        data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])}
else:
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])



######################################################## Data translation  ########################################
if dataname == "CIFAR10":
    if atmeth == 'SSDS' or atmeth == 'NOLAG' or atmeth == 'NOLAM':
        trainset = Datasets.CIFAR10del(root='./data', train=True, download=True, transform=transform_train)
        trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
        v_tr = v_scale*torch.ones(trainloader.dataset.data.shape[0], 1)     
        testset = Datasets.CIFAR10del(root='./data', train=False, download=True, transform=transform_test)
        testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
        v_ts = v_scale*torch.ones(testloader.dataset.data.shape[0], 1)    
    else:
        trainset = Datasets.cifar10clean(root='./data', train=True, download=True, transform=transform_train)
        trainloader = Dataloaders.CleanDataLoader(trainset, batch_size=batchsizetr, shuffle=True) 
        testset = Datasets.cifar10clean(root='./data', train=False, download=True, transform=transform_test)
        testloader = Dataloaders.CleanDataLoader(testset, batch_size=batchsizets, shuffle=True) 
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_channels = 3

if dataname == "IMAGENET":
    data_dir = '/data/tiny-imagenet-200/'
    image_datasets = {x: Datasets.ImageNet(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['train', 'val','test']}
    dataloaders = {x: Dataloaders.DelDataLoaderIMG(image_datasets[x], batch_size=batchsizetr, shuffle=True)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    v_tr = v_scale*torch.ones(len(image_datasets['train']), 1)     
    classes = image_datasets['train'].classes
    trainloader = dataloaders['train']
    testloader = dataloaders['test']
    num_channels = 3

if dataname == "MNIST":
    trainset = Datasets.MNISTdel(root='./data', train=True, download=True, transform=transform_train)
    trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
    v_tr = v_scale*torch.ones(trainloader.dataset.delta.shape[0],1)      
    testset = Datasets.MNISTdel(root='./data', train=False, download=True, transform=transform_test)
    testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
    v_ts = v_scale*torch.ones(testloader.dataset.delta.shape[0], 1).to(device)     
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    num_channels = 1

######################################################## Models ########################################


if model =="Resnet50":
    net = Models.ResNet50(num_classes=len(classes), num_channels=num_channels)

if model =="Resnet18":
    net = Models.ResNet18(num_classes=len(classes), num_channels=num_channels)

if model == "WResnet":
    net = Models.WideResNet(num_classes=len(classes), num_channels=num_channels)

if model == "Simple":
    net = Models.SmallCNN(num_classes=len(classes), num_channels=num_channels)

net = nn.DataParallel(net.to(device))

######################################################## Loss Function ########################################

if loss == 'Xent':
    criterion = nn.CrossEntropyLoss()

if loss == 'TRADES':
    criterion = trades_loss()



######################################################## Optimizers ########################################

if opt == 'SGD':
    optimizer = Optimizers.SGD(net.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

if opt == 'SGDMOM':
    optimizer = Optimizers.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

if opt == 'SubOpt':
    optimizer = Optimizers.SubOpt(net.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

if opt == 'SubOptMOM':
    optimizer = Optimizers.SubOpt(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)



###################################### Main ################################################################
#Training

if atmeth == 'PGD' or  atmeth == 'FGSM' or atmeth == 'REG' :
    trainer = Trainers.RegTrainer(net, trainloader, optimizer, criterion, classes, n_epoch, batchsizetr, expid, checkepoch, pres, stepsize, k, atmeth, epsilon=eps, nstep=nstep)
    trainer.train(epochs=n_epoch, model=net)
elif atmeth == 'SSDS' or atmeth == 'NOLAG' or atmeth == 'NOLAM':
    trainer = Trainers.DelTrainer(net, trainloader, optimizer, criterion, classes, n_epoch, batchsizetr, expid, checkepoch, pres, stepsize, k, atmeth, v_tr, t, lam, c_1, c_2, eps, dataname)
    trainer.train(epochs=n_epoch, model=net)

#Testing
tr_model = torch.load('%s/checkpoint/ckpt.trainbest'%(expid))
net.load_state_dict(tr_model['net'])
print(tr_model['epoch'])
print(tr_model['acc'])
ts_acc_mat = {}

for attack in ['REG', 'PGD', 'FGSM', 'SSDS','NOLAM', 'NOLAG']:
    atmeth = attack
    if atmeth in ['FGSM', 'REG']:
        n_ep_test = 1
    if atmeth in ['SSDS','NOLAM','NOLAG']:
        n_ep_test = n_epoch
    if atmeth == 'PGD':
        n_ep_test = n_ep_PGD
    if atmeth == 'PGD' or  atmeth == 'FGSM' or atmeth == 'REG' :
        tester = Testers.RegTester(net, testloader, optimizer, criterion, classes, n_ep_test, batchsizets, expid, checkepoch, pres, atmeth, epsilon=eps, nstep=nstep, stepsize=stepsize, k=k)
        test_accuracy = tester.test(epochs=n_ep_test, model=net)
        ts_acc_mat[attack] = test_accuracy
    elif atmeth == 'SSDS' or atmeth == 'NOLAG' or atmeth == 'NOLAM':
        tester = Testers.DelTester(net, testloader, optimizer, criterion, classes, n_ep_test, batchsizets, expid, checkepoch, pres, atmeth, v_ts, t, lam, c_1, c_2, eps, stepsize, k, dataname)
        test_accuracy = tester.test(epochs=n_ep_test, model=net)
        ts_acc_mat[attack] = test_accuracy

with open('%s/testresults.json'%(expid), 'w') as res:
    json.dump(ts_acc_mat, res, indent=4)
print(ts_acc_mat)