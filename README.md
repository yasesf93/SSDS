# SSDS (Stochastic Saddle-Point Dynamic System)

This is the reference code for the paper "A Fast Saddle-Point Dynamical System Approach to Robust Deep Learning" (https://arxiv.org/pdf/1910.08623.pdf) by Yasaman Esfandiari (ISU), Aditya Balu (ISU), Keivan Ebrahimi (ISU), Umesh Vaidya (University of Clemson), Nicola Elia (University of Minnesota), and Soumik Sarkar (ISU).

Using this repository you can train naturally trained and adversarially trained (FGSM, PGD, SGDA, SSDS, TRADES, etc) models for various datasets with different combinations of hyper parameters and model architectures. 

## Prerequisites
* Python (3.6.4)
* Pytorch (0.4.1)
* CUDA
* numpy


## How to use SSDS?

### set the hyper parameters in 'config.json' file

* ```data_name```: the dataset used for the experiments (MNIST, Fashion MNIST, CIFAR-10, CIFAR-100, IMAGENET)
* ```attack_method```: The method for computing the perturbations (Regular(no perturbation), FGSM, PGD, NOLAG(SGDA), NOLAM, SSDS, TRADES)
* ```learning_rate_training```: learning rate for training
* ```optimizer```: optimizer for training (SGD, SGDMOM(Momentum SGD), SubOpt(SSDS optimizer), SubOptMOM (Momentum SSDS optimizer))
* ```model_architecture```: model architecture used for training (ResNet50, Wide ResNet, VGG19, Simple)
* ```epsilon```: limit on the perturbation size
* ```num_steps```: number of perturbation iterations PGD and TRADES
* ```step_size_SSDS```: step size for generating the perturbations using SSDS
* ```step_size_PGD```: step size for generating the perturbations using PGD and TRADES
* ```v```: initial v value for SSDS
* ```lambda```: initial lambda value for SSDS
* ```C_1```: C_1 coefficient for SSDS
* ```C_2```: C_2 coefficient for SSDS
* ```beta_TRADES```: trade-off regularization parameter for TRADES

### Train the model
set ```train = true``` in the 'config.json' file and 
```bash
  $ python main.py -g 0
```

### Test the trained model (white-box)
set ```train = false```  and ```black_box = false``` in the 'config.json' file and 
```bash
  $ python main.py -g 0
```
This will test your model against clean input and different white-box attacks (FGSM, PGD, SGDA, SSDS)

### black-box test
```bash
cd SSDS
mkdir Trained Models
cd Trained Models
```
Download our pre-trained models from ['here'](https://drive.google.com/drive/folders/14Xl9k4Alsz6M9d_S_vwS8jZ5aBW7ZYRF?usp=sharing) and publicly available ['Madry'](https://github.com/MadryLab/robustness) and ['TRADES'](https://github.com/yaodongyu/TRADES/blob/master/README.md) models. set ```train = false``` and ```black_box = true``` in the 'config.json' file.

```bash
  $ python main.py -g 0
```
This will test our pre-trained SSDS models against publicly available PGD and TRADES models and vice versa.

## Want to attack SSDS models?
you can download our 'SSDS50.train' and 'SSDSWRN.train' models from ['here'](https://drive.google.com/drive/folders/14Xl9k4Alsz6M9d_S_vwS8jZ5aBW7ZYRF?usp=sharing)

