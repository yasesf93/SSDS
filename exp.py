import json
import os
import copy

with open('config.json', 'r') as f:
    config = json.load(f)

optimizers = ['SGDMOM','SGDMOM','SGDMOM']
attack_method = ['TRADES', 'TRADES', 'TRADES']
models = ['Resnet50', 'WResnet', 'Simple']
data_names = ['CIFAR10', 'CIFAR10', 'MNIST']
step_sizes = [0.007, 0.007, 0.01]
num_steps = [10, 10, 40]
learning_rates = [0.1, 0.1, 0.01]
epsilons = [0.03, 0.03, 0.3]
betas = [6, 6, 1]
exp = 1
for data_name, opt, atmeth, model, eps, num_step, step_size, beta, lr in zip(data_names, optimizers, attack_method, models, epsilons, num_steps, step_sizes, betas, learning_rates):
    cfg = copy.deepcopy(config)
    cfg['data_name'] = data_name
    cfg['optimizer'] = opt
    cfg['attack_method'] = atmeth
    cfg['model_architecture'] = model
    cfg['step_size_PGD'] = step_size
    cfg['num_step'] = num_step
    cfg['beta_TRADES'] = beta
    cfg['learning_rate_training'] = lr
    cfg['training_epochs'] = 200
    cfg['transform'] == False
    with open('experiment%s.json'%exp,'w') as f:
        json.dump(cfg, f, indent=4)
    print(exp)
    exp += 1
