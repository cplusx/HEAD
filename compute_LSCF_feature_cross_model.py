import os
import numpy as np
from sklearn.decomposition import PCA
import torchvision
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--adv_type', type=str, default='benign')
parser.add_argument('--net', type=str, default='vgg16')
parser.add_argument('--feat_dim', type=int, default=32)
args = parser.parse_args()

split = args.split
dataset = args.dataset
adv_type = args.adv_type
net = args.net
feat_dim = args.feat_dim
model_path= f'precomputed_LSCF_model/{dataset}_train.pkl'
save_path = f'precomputed_LSCF_feature/{dataset}/{net}_{adv_type}_{split}.pkl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if adv_type == 'benign':
    ### load images
    if split == 'train':
        if dataset == 'CIFAR10':
            data = torchvision.datasets.CIFAR10(root='./cifar-data', train=True, download=True)
        elif dataset == 'CIFAR100':
            data = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True)
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
        # permute shape for CIFAR, ensure the shape is num_sample, 3, h, w
        images = images.transpose(0, 3, 1, 2)
    images = images / 255.
    num_samples = images.shape[0]
    images = images.reshape(num_samples, -1)
else:
    adv_file_path = f'precomputed_adv_images_cross_model/{dataset}/{net}_{adv_type}_{split}.pkl'
    with open(adv_file_path, 'rb') as IN:
        res = pickle.load(IN)
    images = res['adv_images']
    images = images.transpose(0, 3, 1, 2)
    num_samples = images.shape[0]
    images = images.reshape(num_samples, -1)
    
with open(model_path, 'rb') as IN:
    pca_model = pickle.load(IN)

images_feat = pca_model.transform(images)

with open(save_path, 'wb') as OUT:
    pickle.dump({
        'major': images_feat[:, :feat_dim],
        'minor': images_feat[:, -feat_dim:]
    }, OUT)