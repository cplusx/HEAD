import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
import torchvision
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()

split = args.split
dataset = args.dataset

save_path = f'precomputed_LSCF_model/{dataset}_{split}.pkl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if os.path.exists(save_path):
    exit()

### load images
if split == 'train':
    if dataset == 'CIFAR10':
        data = torchvision.datasets.CIFAR10(root='./cifar-data', train=True, download=True)
    elif dataset == 'CIFAR100':
        data = torchvision.datasets.CIFAR100(root='./ifar100-data', train=True, download=True)
    elif dataset == 'SVHN':
        data = torchvision.datasets.SVHN(root='./svhn-data', split='train', download=True)
if split == 'test':
    if dataset == 'CIFAR10':
        data = torchvision.datasets.CIFAR10(root='./cifar-data', train=False, download=True)
    elif dataset == 'CIFAR100':
        data = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True)
    elif dataset == 'SVHN':
        data = torchvision.datasets.SVHN(root='./svhn-data', split='test', download=True)

        
### format the image shape to n, num pixels
images = data.data
if images.shape[-1] == 3:
    # permute shape for CIFAR
    images = images.transpose(0, 3, 1, 2)
images = images / 255.
num_samples = images.shape[0]
images = images.reshape(num_samples, -1)

### compute PCA
pca = PCA(n_components=images.shape[-1])
pca.fit(images)

with open(save_path, 'wb') as OUT:
    pickle.dump(pca, OUT)