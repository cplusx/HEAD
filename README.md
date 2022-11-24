Code release for [Attack-Agnostic Adversarial Detection](https://openreview.net/pdf?id=I6qYtguqZoR) (Accepted at NeurIPS 2022 Workshop)

This repository provides the code to reproduce the main results in the paper

Please follow the steps. 

## ==To produce result in Table 1==

The following code shows steps on CIFAR10, change --dataset to CIFAR100 or SVHN to see other results

We provide a bash file [for_table1_result.sh](for_table1_result.sh) that contains the commands to obtain results.

### 1. Train a target model (CNN)

This model will be the target of the attacker.
```
python train_clean.py --dataset CIFAR10 --net vgg16 --out_dir ./clean_train/
```

### 2. Compute adversarial images

This needs to be done for testing set for anomaly detection
```
python precompute_adversarial_examples.py --net vgg16 --adv_type PGD10 --split test --dataset CIFAR10
```

see [for_table1_result.sh](for_table1_result.sh) for other attacks

### 3. Compute Least Significant Component Feature
First compute the PCA model of benign images
```
python compute_LSCF_model.py --split train --dataset CIFAR10
```

Then use PCA model to extract the least significant component features of benign and adversarial images. We will need benign images for both training and testing, but adversarial images only for testing.
```
python compute_LSCF_feature.py --adv_type benign --split train --dataset CIFAR10 --feat_dim 32
python compute_LSCF_feature.py --adv_type benign --split test --dataset CIFAR10 --feat_dim 32
python compute_LSCF_feature.py --adv_type PGD10 --split test --dataset CIFAR10 --feat_dim 32
```
see [for_table1_result.sh](for_table1_result.sh) for other attacks

### 4. Compute Hessian Feature
Compute the Hessian feature of benign and adversarial images. We will need benign images for both training and testing, but adversarial images only for testing
```
python precompute_multi_modulus.py --net vgg16 --adv_type benign --split train --dataset CIFAR10 --modulus_mode l1
python precompute_multi_modulus.py --net vgg16 --adv_type benign --split test --dataset CIFAR10 --modulus_mode l1
python precompute_multi_modulus.py --net vgg16 --adv_type PGD10 --split test --dataset CIFAR10 --modulus_mode l1
```
see [for_table1_result.sh](for_table1_result.sh) for other attacks

### 5. Result of KDE
Train a KDE model on the concatenated feature
```
python KDE_train_and_test.py --adv_type overall --dataset CIFAR10 --kernel gaussian --bandwidth 4. --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
```
see [for_table1_result.sh](for_table1_result.sh) for evaluation on single attacks
### 6. Result of OCSVM
Train a OCSVM model on the concatenated feature
```
python OCSVM_train_and_test.py --adv_type overall --dataset CIFAR10 --kernel rbf --nu 0.7 --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
```
see [for_table1_result.sh](for_table1_result.sh) for evaluation on single attacks


## == To produce cross model result in Table 2 ==
### 1. Train a ResNet18 model that will generate cross model adversarial examples

```
python train_clean.py --dataset CIFAR10 --net resnet18 --out_dir ./clean_train/
```

### 2. Compute Least Significant Component Feature
```
python compute_LSCF_feature_cross_model.py --net resnet18 --adv_type PGD10 --split test --dataset CIFAR10 --feat_dim 32
```
### 3. Compute Hessian Feature

Compute the Hessian feature of benign and adversarial images. We only need test examples since anomaly detector will be trained on the same benign images as above
```
python precompute_multi_modulus_cross_model.py --net_of_adv resnet18  --net vgg16 --adv_type benign --split test --dataset CIFAR10 --modulus_mode l1
python precompute_multi_modulus_cross_model.py --net_of_adv resnet18  --net vgg16 --adv_type PGD10 --split test --dataset CIFAR10 --modulus_mode l1
```
### 4. Result of KDE
Train a KDE model on the concatenated feature
```
python KDE_train_and_test_cross_model.py --adv_type overall --dataset CIFAR10 --kernel gaussian --bandwidth 4. --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
```

### 5. Result of OCSVM
Train a OCSVM model on the concatenated feature
```
python OCSVM_train_and_test_cross_model.py --adv_type overall --dataset CIFAR10 --kernel rbf --nu 0.7 --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
```

## == To produce supervised training result in Figure 6 ==
### 1. Compute adversarial images for training

Change adv_type to see result of different attacks
```
python precompute_adversarial_examples.py --net vgg16 --adv_type PGD10 --split train --dataset CIFAR10
python supervised_DNN.py --adv_type PGD10 --dataset CIFAR10 --norm_feat 1
```
