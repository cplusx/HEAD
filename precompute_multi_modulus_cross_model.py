import os
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import pickle
import argparse
from tqdm import tqdm
from custom_utils import get_model, compute_gaussian_newton, matrix_modulus
from custom_models import VGG16_splitted

def get_benign_images(dataset, split):
    if split == 'train':
        if dataset == 'CIFAR10':
            img_set = torchvision.datasets.CIFAR10(root='./cifar-data', train=True, download=True)
        elif dataset == 'CIFAR100':
            img_set = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True)
        elif dataset == 'SVHN':
            img_set = torchvision.datasets.SVHN(root='./svhn-data', split='train', download=True)
    elif split == 'test':
        if dataset == 'CIFAR10':
            img_set = torchvision.datasets.CIFAR10(root='./cifar-data', train=False, download=True)
        elif dataset == 'CIFAR100':
            img_set = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True)
        elif dataset == 'SVHN':
            img_set = torchvision.datasets.SVHN(root='./svhn-data', split='test', download=True)
    return img_set

def load_model(net, dataset, n_classes):
    folder = os.path.join('clean_train', dataset, net)
    model_path = os.path.join(folder, 'epoch_120.pth.tar')
    model = get_model(net, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    return model

def compute_gaussian_newton_features(
    model, images, save_path, 
    modulus_mode='l1'):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, 'rb') as IN:
            res = pickle.load(IN)
        feature = res['feature']
        start_idx = res['num_processed']
        print('Resumed from idx', start_idx)
    else:
        start_idx = 0
        feature = np.zeros((len(images), 13), dtype=np.float32) # 13 is for VGG_splitted

        
    for idx in tqdm(range(start_idx, len(images))):
        this_image = images[idx: idx+1]
        if this_image.shape[-1] == 3:
            this_image = this_image.transpose(0, 3, 1, 2)
        this_image = torch.tensor(this_image, requires_grad=True).contiguous().cuda()
        feature[idx] = np.array(
            model.compute_layerwise_gaussian_newton(this_image, [0], modulus_mode)
        )
        # gn = compute_gaussian_newton(model, this_image, 0) # 0 is dummy target
        # feature[idx] = matrix_modulus(gn, mode, modulus_mode)
    
        if (idx + 1) % 100 == 0:
            with open(save_path, 'wb') as OUT:
                pickle.dump({
                    'feature': feature,
                    'num_processed': idx + 1
                }, OUT)
    with open(save_path, 'wb') as OUT:
        pickle.dump({
            'feature': feature,
            'num_processed': len(images)
        }, OUT)


parser = argparse.ArgumentParser()
parser.add_argument('--adv_type', type=str, default='PGD10')
parser.add_argument('--net', type=str, default='vgg16')
parser.add_argument('--net_of_adv', type=str, default='resnet18')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--modulus_mode', type=str, default='l1')
parser.add_argument('--out_dir', type=str, default='precomputed_GGN_modulus')
args = parser.parse_args()

# 1. arguments
adv_type = args.adv_type
net = args.net
net_of_adv = args.net_of_adv
dataset = args.dataset
split = args.split
modulus_mode = args.modulus_mode
out_dir = args.out_dir

save_path = f'{out_dir}/multi_modulus_cross_model/{dataset}/{net_of_adv}_on_{net}_{adv_type}_{split}.pkl'


if dataset == 'CIFAR100':
    n_classes = 100
else:
    n_classes = 10

gn_model = load_model(net, dataset, n_classes)
gn_model = gn_model.eval()
if net == 'vgg16':
    gn_model = VGG16_splitted(gn_model)
else:
    raise NotImplementedError

# 2. load test clean image and adversarial image
if adv_type == 'benign':
    img_set = get_benign_images(dataset, split)
    images = (img_set.data.astype(np.float32) / 255)
else:
    # adv images
    adv_file_path = f'precomputed_adv_images_cross_model/{dataset}/{net_of_adv}_{adv_type}_{split}.pkl'

    with open(adv_file_path, 'rb') as IN:
        res = pickle.load(IN)
        images = res['adv_images']
        num_valid_images = res['num_processed']
        if num_valid_images != len(images):
            # make sure all adv images are generated
            raise RuntimeError('Wait for {} to complete'.format(adv_file_path))
# 3. precompute gaussian newton features
compute_gaussian_newton_features(
    gn_model, images, save_path,
    modulus_mode=modulus_mode
)