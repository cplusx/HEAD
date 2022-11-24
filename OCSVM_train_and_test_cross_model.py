import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.svm import OneClassSVM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--kernel', type=str, default='rbf')
parser.add_argument('--nu', type=float, default=0.5)
parser.add_argument('--adv_type', type=str, default='overall')
parser.add_argument('--hessian_dim', type=int, default=13)
parser.add_argument('--LSCF_dim', type=int, default=32)
#
parser.add_argument('--modulus_mode', type=str, default='l1')
parser.add_argument('--norm_feat', type=int, default=0)
args = parser.parse_args()

# hyper params
modulus_mode = args.modulus_mode
dataset = args.dataset
kernel = args.kernel
nu = args.nu
adv_type = args.adv_type
hessian_dim = args.hessian_dim
LSCF_dim = args.LSCF_dim
norm_feat = args.norm_feat

save_path = f'OCSVM_cross_model/{dataset}/Hessian_{hessian_dim}-LSCF_{LSCF_dim}-norm_{norm_feat}/{kernel}_{nu}_{adv_type}.txt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
if os.path.exists(save_path):
    exit()

adv_types = ['PGD10', 'PGD10L2', 'FGSM', 'BIM', 'CW', 'DeepFool', 'AutoAttack', 'OnePixel', 'SparseFool']
net_of_adv = 'resnet18'
net = 'vgg16'

def format_data(x):
    x = np.array(x)
    num_samples = x.shape[0]
    return x.reshape(num_samples, -1)
# read data
def load_data(
    dataset='CIFAR10', net_of_adv='resnet18', net='vgg16', 
    adv_type='benign', split='train', 
    hessian_dim=13, LSCF_dim=32,
    modulus_mode='l1'
):
    assert hessian_dim + LSCF_dim > 0, 'At least one dim is required'
    hessian_feature = None
    if hessian_dim > 0:
        if split == 'test':
            # use cross model adversarial examples during test
            file_path = f'precomputed_GGN_modulus/multi_modulus_cross_model/{dataset}/{net_of_adv}_on_{net}_{adv_type}_{split}.pkl'
        else:
            # use benign image inliers
            file_path = f'precomputed_GGN_modulus/multi_modulus/{dataset}/{net}_{adv_type}_{split}.pkl'
        with open(file_path, 'rb') as IN:
            res = pickle.load(IN)
        valid_samples = res['num_processed']
        hessian_feature = format_data(res['feature'][:valid_samples])[:, : hessian_dim] # feature counts from the input to deepest layer
        hessian_feature = np.log(np.abs(hessian_feature) + 1e-10)
        
    LSCF_feature = None
    if LSCF_dim > 0:
        file_path = f'precomputed_LSCF_feature/{dataset}/{net_of_adv}_{adv_type}_{split}.pkl'
        with open(file_path, 'rb') as IN:
            res = pickle.load(IN)
        LSCF_feature = format_data(res['minor'])[:, -LSCF_dim:] # feature counts from the smallest
        
    if hessian_feature is None:
        feature = LSCF_feature
    elif LSCF_feature is None:
        feature = hessian_feature
    else:
        feature = np.concatenate([hessian_feature, LSCF_feature[:valid_samples]], axis=1)
    
    return feature

train_benign = load_data(dataset=dataset, net_of_adv=net_of_adv, net=net, adv_type='benign', split='train', hessian_dim=hessian_dim, LSCF_dim=LSCF_dim, modulus_mode=modulus_mode)
test_benign = load_data(dataset=dataset, net_of_adv=net_of_adv, net=net, adv_type='benign', split='test', hessian_dim=hessian_dim, LSCF_dim=LSCF_dim, modulus_mode=modulus_mode)
test_adv = {
    'overall': []
}
for at in adv_types:
    test_adv[at] = load_data(dataset=dataset, net_of_adv=net_of_adv, net=net, adv_type=at, split='test', hessian_dim=hessian_dim, LSCF_dim=LSCF_dim, modulus_mode=modulus_mode)
    test_adv['overall'].append(test_adv[at])
test_adv['overall'] = np.concatenate(test_adv['overall'], axis=0)

# roc auc
def get_roc_from_data(ocsvm, inlier, outlier):
    y_score = ocsvm.score_samples(np.concatenate([inlier, outlier]))
    # min_val = -1e10
    # y_score = np.nan_to_num(y_score, nan=min_val, posinf=min_val, neginf=min_val)
    y_true = np.array([1]*len(inlier) + [0]*len(outlier))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc


# may add normalization before fit the KDE/OCSVM model?
if norm_feat == 1:
    mean = np.mean(train_benign, axis=0).reshape(1, -1)
    std = np.std(train_benign, axis=0).reshape(1, -1)
    train_benign = (train_benign - mean) / std
    test_benign = (test_benign - mean) / std
    for k, v in test_adv.items():
        test_adv[k] = (v - mean) / std

ocsvm = OneClassSVM(kernel=kernel, nu=nu).fit(train_benign)
inlier = test_benign
outlier = test_adv[adv_type]
_, _, auc = get_roc_from_data(ocsvm, inlier, outlier)


with open(save_path, 'w') as OUT:
    OUT.write('{:.3f}'.format(auc))