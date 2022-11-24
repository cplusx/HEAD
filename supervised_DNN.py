import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import torch
from torch import nn
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--adv_type', type=str, default='PGD10')
parser.add_argument('--hessian_dim', type=int, default=13)
parser.add_argument('--PCA_dim', type=int, default=32)
parser.add_argument('--modulus_mode', type=str, default='l1')
parser.add_argument('--norm_feat', type=int, default=1)
args = parser.parse_args()
# hyper params
modulus_mode = args.modulus_mode
dataset = args.dataset
adv_type = args.adv_type
hessian_dim = args.hessian_dim
PCA_dim = args.PCA_dim
norm_feat = args.norm_feat

save_path = f'SimpleDNN/{dataset}/Hessian_{hessian_dim}-PCA_{PCA_dim}-norm_{norm_feat}/{adv_type}.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# # roc auc
# def get_roc_from_data(clf, inlier, outlier):
#     y_score = clf.predict_proba(np.concatenate([inlier, outlier]))
#     y_score = np.max(y_score, axis=1)
#     y_true = np.array([1]*len(inlier) + [0]*len(outlier))
#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     auc = roc_auc_score(y_true, y_score)
#     return fpr, tpr, auc

def test_model(model, X, y):
    # compute ROC AUC
    model.eval()
    with torch.no_grad():
        y_score = model(X).ravel().numpy()
    y_true = y
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return auc

def train_and_test_model(X, y, X_test, y_test, batch_size = 100, epochs = 100):
    num_sample, feat_dim = X.shape
    num_batches = num_sample // batch_size
    model = nn.Sequential(
        nn.Linear(feat_dim, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32, 8),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    roc_auc = {}
    for epoch in range(epochs):
        print(epoch+1, '/', epochs)
        model.train()
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            data, target= X[start: end], y[start: end]
            data = torch.from_numpy(data.astype(np.float32))
            target = torch.tensor(target).unsqueeze(-1).to(torch.float) # shape (100,) -> (100, 1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss(reduction='mean')(output, target)
            loss.backward()
            optimizer.step()
        this_roc_auc = test_model(model, torch.from_numpy(X_test.astype(np.float32)), y_test)
        roc_auc[epoch+1] = '{:.3f}'.format(this_roc_auc)
    return roc_auc
        

def format_data(x):
    x = np.array(x)
    num_samples = x.shape[0]
    return x.reshape(num_samples, -1)
# read data
def load_data(
    dataset='CIFAR10', net='vgg16', 
    adv_type='benign', split='train', 
    hessian_dim=13, PCA_dim=32,
    modulus_mode='l1'
):
    assert hessian_dim + PCA_dim > 0, 'At least one dim is required'
    hessian_feature = None
    if hessian_dim > 0:
        file_path = f'precomputed_GGN_modulus/multi_modulus/{dataset}/{net}_{adv_type}_{split}.pkl'
        with open(file_path, 'rb') as IN:
            res = pickle.load(IN)
        valid_samples = res['num_processed']
        hessian_feature = format_data(res['feature'][:valid_samples])[:, : hessian_dim] # feature counts from the input to deepest layer
        hessian_feature = np.log(np.abs(hessian_feature) + 1e-10)
        
    PCA_feature = None
    if PCA_dim > 0:
        file_path = f'baseline_PCA/precomputed_PCA_feature/{dataset}/{net}_{adv_type}_{split}.pkl'
        with open(file_path, 'rb') as IN:
            res = pickle.load(IN)
        PCA_feature = format_data(res['minor'])[:, -PCA_dim:] # feature counts from the smallest
        
    if hessian_feature is None:
        feature = PCA_feature
    elif PCA_feature is None:
        feature = hessian_feature
    else:
        feature = np.concatenate([hessian_feature, PCA_feature[:valid_samples]], axis=1)
    
    return feature

# wait for data to be prepared
# train_benign = load_data(dataset=dataset, adv_type='benign', split='train', hessian_dim=hessian_dim, PCA_dim=PCA_dim, modulus_mode=modulus_mode)
# test_benign = load_data(dataset=dataset, adv_type='benign', split='test', hessian_dim=hessian_dim, PCA_dim=PCA_dim, modulus_mode=modulus_mode)

# train_adv = load_data(dataset=dataset, adv_type=adv_type, split='train', hessian_dim=hessian_dim, PCA_dim=PCA_dim, modulus_mode=modulus_mode)
# test_adv = load_data(dataset=dataset, adv_type=adv_type, split='test', hessian_dim=hessian_dim, PCA_dim=PCA_dim, modulus_mode=modulus_mode)

# temperary
all_benign = load_data(dataset=dataset, adv_type='benign', split='test', hessian_dim=hessian_dim, PCA_dim=PCA_dim, modulus_mode=modulus_mode)
all_adv = load_data(dataset=dataset, adv_type=adv_type, split='test', hessian_dim=hessian_dim, PCA_dim=PCA_dim, modulus_mode=modulus_mode)

num_images = len(all_benign)
num_train = int(0.8 * num_images)
num_test = num_images - num_train
train_benign = all_benign[:num_train]
train_adv = all_adv[:num_train]
test_benign = all_benign[num_train:]
test_adv = all_adv[num_train:]



# continue from here
train_data = np.concatenate([train_benign, train_adv], axis=0)
train_label = [1] * num_train + [0] * num_train
train_rnd_idx = np.arange(len(train_data))
np.random.shuffle(train_rnd_idx)
train_data = train_data[train_rnd_idx]
train_label = [train_label[i] for i in train_rnd_idx]

test_data = np.concatenate([test_benign, test_adv], axis=0)
test_label = [1] * num_test + [0] * num_test

if norm_feat == 1:
    mean = np.mean(train_data, axis=0).reshape(1, -1)
    std = np.std(train_data, axis=0).reshape(1, -1)
    train_data = (train_data-mean) / std
    test_data = (test_data-mean) / std

res = train_and_test_model(train_data, train_label, test_data, test_label)

print(f'AUC {res}')
with open(save_path, 'w') as OUT:
    json.dump(res, OUT)