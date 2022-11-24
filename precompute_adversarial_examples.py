import os
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import pickle
from custom_utils import get_model# , compute_gaussian_newton, get_matrix_modulus_feature
import torchattacks
import argparse
from tqdm import tqdm
from custom_dataloader import cereate_adversarial_dataset
# from custom_utils import get_matrix_modulus_feature, get_matrix_modulus_feature_for_training
import argparse

def load_model(net, dataset, n_classes):
    folder = os.path.join('clean_train', dataset, net)
    model_path = os.path.join(folder, 'epoch_120.pth.tar')
    model = get_model(net, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--adv_type', type=str)
parser.add_argument('--net', type=str, default='vgg16')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()

# 1. arguments
adv_type = args.adv_type
net = args.net
dataset = args.dataset
split = args.split

if dataset == 'CIFAR100':
    n_classes = 100
else:
    n_classes = 10

# 2. precompute adversarial images
gn_model = load_model(net, dataset, n_classes)
adv_train_set_file = f'precomputed_adv_images_cross_model/{dataset}/{net}_{adv_type}_train.pkl'
adv_test_set_file = f'precomputed_adv_images_cross_model/{dataset}/{net}_{adv_type}_test.pkl'

if split == 'train':
    if dataset == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root='./cifar-data', train=True, download=True)
    elif dataset == 'CIFAR100':
        train_set = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True)
    elif dataset == 'SVHN':
        train_set = torchvision.datasets.SVHN(root='./svhn-data', split='train', download=True)
    cereate_adversarial_dataset(gn_model, train_set, adv_type, adv_train_set_file, n_classes)
    
if split == 'test':
    if dataset == 'CIFAR10':
        test_set = torchvision.datasets.CIFAR10(root='./cifar-data', train=False, download=True)
    elif dataset == 'CIFAR100':
        test_set = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True)
    elif dataset == 'SVHN':
        test_set = torchvision.datasets.SVHN(root='./svhn-data', split='test', download=True)
    cereate_adversarial_dataset(gn_model, test_set, adv_type, adv_test_set_file, n_classes)

