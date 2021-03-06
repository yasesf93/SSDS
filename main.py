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
import Loss
import Testers
import argparse 
from collections import OrderedDict
from Loss.cw import CWLoss

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, default='config.json')
parser.add_argument('-g', '--gpu', type=int)
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
with open(args.exp) as config_file: # Reading the Config File 
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
nstep_train = config['num_steps_train']
stepsize_ssds = config['step_size_SSDS']
stepsize_pgd = config['step_size_PGD']
v_scale = config['v']
lam = config['lambda']
k = config['step_size_decay']
c_1 = config['c_1']
c_2 = config['c_2']
t = config['t']
n_ep_PGD = config['PGD_Restarts']
beta = config['beta_TRADES']
blackbox = config['black_box']
sourcem = config['source_model']
targetm = config['target_model']

if not os.path.exists('Experiments'):
    os.makedirs('Experiments')
expid = 'Experiments/%s_%s_%s_%s_%s'%(str(atmeth), str(opt), str(loss), str(dataname), str(model))
print('expid',expid)
######################################################## Transformation ########################################
if config['transform']==True:
    if dataname  in ["CIFAR10", "CIFAR100"]:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataname == "SVHN":
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # [-1 1]
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # [-1 1]
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
    trainset = Datasets.CIFAR10del(root='./data', train=True, download=True, transform=transform_train)
    trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
    v_tr = v_scale*torch.ones(trainloader.dataset.data.shape[0], 1)     
    testset = Datasets.CIFAR10del(root='./data', train=False, download=True, transform=transform_test)
    testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
    v_ts = v_scale*torch.ones(testloader.dataset.data.shape[0], 1)    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    num_channels = 3

if dataname == "CIFAR100":
    trainset = Datasets.CIFAR100del(root='./data', train=True, download=True, transform=transform_train)
    trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
    v_tr = v_scale*torch.ones(trainloader.dataset.data.shape[0], 1)     
    testset = Datasets.CIFAR100del(root='./data', train=False, download=True, transform=transform_test)
    testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
    v_ts = v_scale*torch.ones(testloader.dataset.data.shape[0], 1)    
    num_classes = 100
    num_channels = 3

if dataname == "SVHN":
    trainset = Datasets.SVHNdel(root='./data',split='train',download=True,transform=transform_train)
    trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
    v_tr = v_scale*torch.ones(trainloader.dataset.data.shape[0], 1)     
    testset = Datasets.SVHNdel(root='./data',split='test',download=True,transform=transform_test)
    testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
    v_ts = v_scale*torch.ones(testloader.dataset.data.shape[0], 1)
    num_classes = 10
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
    num_classes = len(classes)

if dataname == "MNIST":
    trainset = Datasets.MNISTdel(root='./data', train=True, download=True, transform=transform_train)
    trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
    v_tr = v_scale*torch.ones(trainloader.dataset.delta.shape[0],1)      
    testset = Datasets.MNISTdel(root='./data', train=False, download=True, transform=transform_test)
    testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
    v_ts = v_scale*torch.ones(testloader.dataset.delta.shape[0], 1).to(device)     
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    num_channels = 1
    num_classes = len(classes)

if dataname == "FashionMNIST":
    trainset = Datasets.FMNISTdel(root='./data/FMNIST', train=True, download=True, transform=transform_train)
    trainloader = Dataloaders.DelDataLoader(trainset, batch_size=batchsizetr, shuffle=True)
    v_tr = v_scale*torch.ones(trainloader.dataset.delta.shape[0],1)      
    testset = Datasets.FMNISTdel(root='./data/FMNIST', train=False, download=True, transform=transform_test)
    testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
    v_ts = v_scale*torch.ones(testloader.dataset.delta.shape[0], 1).to(device) 
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']    
    num_channels = 1
    num_classes = len(classes)

######################################################## Models ########################################


if model =="Resnet50":
    net = Models.ResNet50(num_classes=num_classes, num_channels=num_channels)
    net_s = Models.ResNet50(num_classes=num_classes, num_channels=num_channels)

if model == "WResnet":
    net = Models.WideResNet(num_classes=num_classes, num_channels=num_channels)
    net_s = Models.WideResNet(num_classes=num_classes, num_channels=num_channels)

if model == "Simple":
    net = Models.SmallCNN(num_classes=num_classes, num_channels=num_channels)
    net_s = Models.SmallCNN(num_classes=num_classes, num_channels=num_channels)

if model == "VGG":
    net = Models.VGG('VGG19')
    net_s = Models.VGG('VGG19')

if dataname == 'IMAGENET':
    net = Models.ResNet18(num_classes=num_classes, num_channels=num_channels)
    net.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = net.linear.in_features
    print(num_ftrs)
    net.linear = nn.Linear(num_ftrs*49, 200)   

# net = nn.DataParallel(net)

net = net.to(device)
net_s = net_s.to(device)


######################################################## Optimizers ########################################

if opt in ['SGD']:
    optimizer = Optimizers.SGD(net.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

if opt in ['SGDMOM']:
    optimizer = Optimizers.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

if opt in ['SubOpt']:
    optimizer = Optimizers.SubOpt(net.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

if opt in ['SubOptMOM']:
    optimizer = Optimizers.SubOpt(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)




######################################################## Loss Function ########################################

if loss == 'Xent':
    criterion = nn.CrossEntropyLoss()


###################################### Main ################################################################
#Training
if training == True:

    if atmeth in ['PGD', 'FGSM', 'REG', 'TRADES', 'Madry']:
        trainer = Trainers.RegTrainer(net, trainloader, optimizer, criterion, n_epoch, batchsizetr, expid, checkepoch, pres, stepsize_pgd, k, atmeth, c_1, c_2, eps, dataname, nstep,lr, beta)
        trainer.train(epochs=n_epoch, model=net)
    elif atmeth == 'SSDS' or atmeth == 'NOLAG' or atmeth == 'NOLAM':
        trainer = Trainers.DelTrainer(net, trainloader, optimizer, criterion, n_epoch, batchsizetr, expid, checkepoch, pres, stepsize_ssds, k, atmeth, c_1, c_2, eps, dataname, nstep,lr, v_tr, t, lam)
        trainer.train(epochs=n_epoch, model=net)

#Testing
if blackbox == True:   
    if sourcem in ['SSDS50', 'SSDSWRN']:
        opt = 'SubOptMOM'
    else:
        opt = 'SGDMOM'
    if sourcem == 'Madry':
        source_model = torch.load('Trained Models/Madry50.pt', map_location='cuda:1')
        state_dict = source_model['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[13:] # remove `module.model`
            new_state_dict[name] = v
        net_s.load_state_dict(new_state_dict, strict=False)
    elif sourcem == 'TRADES':
        source_model = torch.load('Trained Models/TRADESWRN.pt', map_location='cuda:1')
        net_s.load_state_dict(source_model)
    elif sourcem == 'SSDS50':
        source_model = torch.load('Trained Models/SSDS50.train', map_location='cuda:1')
        net_s.load_state_dict(source_model['net'])
    else:
        source_model = torch.load('Trained Models/SSDSWRN.train', map_location='cuda:1')
        net_s.load_state_dict(source_model['net'])

    if targetm in ['SSDS50', 'SSDSWRN']:
        opt = 'SubOptMOM'
    else:
        opt = 'SGDMOM'
    if targetm == 'Madry':
        target_model = torch.load('Trained Models/Madry50.pt', map_location='cuda:1')
        state_dict_target = target_model['state_dict']
        new_state_dict_target = OrderedDict()
        for k, v in state_dict_target.items():
            name = k[13:] # remove `module.model`
            new_state_dict_target[name] = v
        net.load_state_dict(new_state_dict_target, strict=False)
    elif targetm == 'TRADES':
        target_model = torch.load('Trained Models/TRADESWRN.pt', map_location='cuda:1')
        net.load_state_dict(target_model)
    elif targetm == 'SSDS50':
        target_model = torch.load('Trained Models/SSDS50.train', map_location='cuda:1')
        net.load_state_dict(target_model['net'])
    else: 
        target_model = torch.load('Trained Models/SSDSWRN.train', map_location='cuda:1')
        net.load_state_dict(target_model['net'])
        print(target_model['epoch'])
        print(target_model['acc'])
    testerBB = Testers.RegTesterBB(net, testloader, optimizer, criterion, n_ep_PGD, batchsizets, expid, checkepoch, pres, stepsize_pgd, k, atmeth, c_1, c_2, eps, dataname, nstep, net_s)
    bb_test_accuracy = testerBB.test(epochs=1, model=net)
    print('source model', sourcem)
    print('target model', targetm)
    print('Black box accuracy', bb_test_accuracy)
else:
    #white_box_Testing
    tr_model = torch.load('%s/checkpoint/ckpt.trainbest'%(expid), map_location='cuda:0')
    net.load_state_dict(tr_model['net'])
    print(tr_model['epoch'])
    print(tr_model['acc'])
    ts_acc_mat = {}

    for attack in ['CW', 'PGD', 'REG', 'FGSM']:
        if dataname == "MNIST":
            testset = Datasets.MNISTdel(root='./data', train=False, download=True, transform=transform_test)
            testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
        if dataname == "FashionMNIST":
            testset = Datasets.FMNISTdel(root='./data/FMNIST', train=False, download=True, transform=transform_test)
            testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
        if dataname == "CIFAR10":
            testset = Datasets.CIFAR10del(root='./data', train=False, download=True, transform=transform_test)
            testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
        if dataname == "CIFAR100":
            testset = Datasets.CIFAR100del(root='./data', train=False, download=True, transform=transform_test)
            testloader = Dataloaders.DelDataLoader(testset, batch_size=batchsizets, shuffle=True)
        if dataname == "IMAGENET":
            testloader = dataloaders['test']
        atmeth = attack
        if atmeth in ['FGSM', 'REG', 'CW']:
            n_ep_test = 1
        if atmeth in ['SSDS','NOLAM','NOLAG']:
            n_ep_test = n_epoch
        if atmeth == 'PGD':
            n_ep_test = n_ep_PGD
        if atmeth == 'CW':
            atmeth_name = 'PGD'
            criterion_att = CWLoss(num_classes)
            tester = Testers.RegTester(net, testloader, optimizer, criterion, n_ep_test, batchsizets, expid, checkepoch, pres, stepsize_pgd, k, atmeth_name, c_1, c_2, eps, dataname, nstep, criterion_att)
            test_accuracy = tester.test(epochs=n_ep_test, model=net)
            ts_acc_mat[attack] = test_accuracy
        if atmeth == 'PGD' or  atmeth == 'FGSM' or atmeth == 'REG' :
            criterion_att = nn.CrossEntropyLoss()
            tester = Testers.RegTester(net, testloader, optimizer, criterion, n_ep_test, batchsizets, expid, checkepoch, pres, stepsize_pgd, k, atmeth, c_1, c_2, eps, dataname, nstep, criterion_att)
            test_accuracy = tester.test(epochs=n_ep_test, model=net)
            ts_acc_mat[attack] = test_accuracy
        elif atmeth == 'SSDS' or atmeth == 'NOLAG' or atmeth == 'NOLAM':
            tester = Testers.DelTester(net, testloader, optimizer, criterion, n_ep_test, batchsizets, expid, checkepoch, pres, stepsize_ssds, k, atmeth, c_1, c_2, eps, dataname, nstep, v_ts, t, lam)
            test_accuracy = tester.test(epochs=n_ep_test, model=net)
            ts_acc_mat[attack] = test_accuracy

    with open('%s/testresults.json'%(expid), 'w') as res:
        json.dump(ts_acc_mat, res, indent=4)
    print(ts_acc_mat)
