import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
from Trainer import Trainer
import json

with open('config.json') as config_file:
    config = json.load(config_file)
os.environ['CUDA_VISIBLE_DEVICE'] = '3'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_num= config['random_seed']
def seed_everything(seed=seed_num):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()


lr=config['learning_rate_training']
wd=config['weight_decay']
momentum=config['momentum']
n_epoch= config['training_epochs']
batchsizetr=config['training_batch_size']
batchsizets=config['test_batch_size']
expdata=config['data_path']
dataname=config['data_name']
dataldr=config['dataloader']
loss = config['loss_function']
opt = config['optimizer']
net = config['model_architecture']


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

trainset = expdata(root='./data', train=True, download=True, transform=transform_train)
trainloader = dataldr(trainset, batch_size=batchsizetr, shuffle=True)  
testset = expdata(root='./data', train=False, download=True, transform=transform_test)
testloader = dataldr(testset, batch_size=batchsizetr, shuffle=True)
if dataname=="CIFAR10":
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = net.to(device)

if loss == 'Xent':
    criterion = nn.CrossEntropyLoss()

if opt =='sgd':
    optimizer = optim.SGD(net.parameters(), lr=lr)


trainer = Trainer(net, trainloader, testloader, optimizer, criterion, classes, n_epoch, device)
trainer.train(epochs=n_epoch)