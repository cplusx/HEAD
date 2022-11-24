import torch
from torch.utils.data import Dataset, DataLoader
from custom_utils import compute_gaussian_newton # , get_matrix_modulus_feature
import torchattacks
import numpy as np
from tqdm import tqdm
import os
import pickle

class GaussianNoise():
    # this is harmless random noise for ablation on gaussian vs adv
    def __init__(self, mean=0, std=3/255):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, label):
        bs, d, h, w = image.shape
        perturb = self.mean + torch.randn(bs, d, h, w) * self.std
        image = image + perturb.to(image.device)
        image = torch.clamp(image, 0, 1)
        return image
    
class UniformNoise():
    # this is harmless random noise for ablation on uniform vs adv
    def __init__(self, mean=0, max_perturb=3/255):
        self.mean = mean
        self.max_perturb = max_perturb
        
    def __call__(self, image, label):
        bs, d, h, w = image.shape
        perturb = self.mean + (torch.rand(bs, d, h, w)-0.5) * 2 * self.max_perturb
        image = image + perturb.to(image.device)
        image = torch.clamp(image, 0, 1)
        return image
    

def get_attack_method(model, adv_type, n_classes=10):
    if adv_type == 'PGD10':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    elif adv_type == 'FGSM':
        atk = torchattacks.FGSM(model, eps=8/255)
    elif adv_type == 'BIM':
        atk = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
    elif adv_type == 'CW':
        atk = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    elif adv_type == 'DeepFool':
        atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    elif adv_type == 'AutoAttack':
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=n_classes, seed=None, verbose=False)
    elif adv_type == 'OnePixel':
        atk = torchattacks.OnePixel(model, pixels=1, steps=75, popsize=400, inf_batch=128)
    elif adv_type == 'SparseFool':
        atk = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
    elif adv_type == 'Gaussian':
        atk = GaussianNoise()
    elif adv_type == 'Uniform':
        atk = UniformNoise()
    return atk

def get_adv_example(atk, image, target):
    if isinstance(target, list):
        target = torch.tensor(target).cuda()
    elif isinstance(target, int):
        target = torch.tensor([target]).cuda()
    image = image.contiguous().cuda()
    adv_image = atk(image, target)
    return adv_image

def pred_acc(model, image, target):
    if isinstance(target, list):
        target = torch.tensor(target).cuda()
    elif isinstance(target, int):
        target = torch.tensor([target]).cuda()
    image = image.cuda()
    model.eval()
    output = model(image)
    pred = torch.argmax(output, dim=-1)
    acc = torch.eq(pred, target)
    return acc.cpu().detach().numpy()

def cereate_adversarial_dataset(model, dataset, adv_type, save_path, n_classes):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # dataset
    data = torch.from_numpy(dataset.data)# torch tensor
    data = data.to(torch.float)
    if data.shape[-1] == 3:
        # permute shape for CIFAR
        data = data.permute(0, 3, 1, 2)
        labels = dataset.targets
    else:
        labels = dataset.labels
    data = data / 255.
    
    # attack method
    atk = get_attack_method(model, adv_type, n_classes)
    
    if os.path.exists(save_path):
        with open(save_path, 'rb') as IN:
            res = pickle.load(IN)
        adv_images = res['adv_images'].transpose(0, 3, 1, 2).astype(np.float32)
        acc = res['acc']
        start_idx = res['num_processed']
        print('Resumed from idx', start_idx)
    else:
        adv_images = np.zeros_like(data).astype(np.float32)
        acc = []
        start_idx = 0
    for i in tqdm(range(start_idx, len(data))):
        this_data = data[i: i+1]
        this_label = labels[i: i+1]
        this_data = torch.tensor(this_data).cuda()
        this_label = torch.tensor(this_label).cuda()
        this_adv = get_adv_example(atk, this_data, this_label).detach().cpu().numpy()
        this_acc = pred_acc(model, torch.from_numpy(this_adv), this_label)
        adv_images[i] = this_adv
        acc.append(this_acc)
    
        if (i+1) % 2000 == 0:
            with open(save_path, 'wb') as OUT:
                pickle.dump({
                    'adv_images': adv_images.transpose(0,2,3,1).astype(np.float32),
                    'labels': labels, # this is the original label of images
                    'acc': acc,
                    'num_processed': i+1
                }, OUT)
    with open(save_path, 'wb') as OUT:
        pickle.dump({
            'adv_images': adv_images.transpose(0,2,3,1).astype(np.float32),
            'labels': labels, # this is the original label of images
            'acc': acc,
            'num_processed': len(data)
        }, OUT)
        
class GaussianNewtonFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        gn_feature_model,
        images, 
        features_used = ['min_diag_feat', 'max_diag_feat', 'min_feat', 'max_feat'],
        num_ele_each_feat = 16
    ):
        self.gn_feature_model = gn_feature_model
        self.images = images
        self.features_used = features_used
        self.num_ele_each_feat = num_ele_each_feat
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        self.gn_feature_model.eval()
        image = torch.from_numpy(self.images[idx:idx+1])
        image = image.permute(0, 3, 1, 2).contiguous().cuda()
        gn = compute_gaussian_newton(self.gn_feature_model, image, [0]) # for target, just give a dummy target
        gn_features = get_matrix_modulus_feature(gn, self.num_ele_each_feat)
        
        cat_feat = []
        for key in self.features_used:
            cat_feat.append(gn_features[key])
        cat_feat = np.concatenate(cat_feat, axis=-1)
        return torch.from_numpy(cat_feat)
    
if __name__ == '__main__':
    import os
    import numpy as np
    import pickle
    import torchvision
    from custom_utils import get_model
    
    def load_model(net):
        folder = os.path.join('clean_train', f'net_{net}')
        model_path = os.path.join(folder, 'epoch_100.pth.tar')
        model = get_model(net)
        model.load_state_dict(torch.load(model_path)['state_dict'])
        return model
    
    net = 'vgg16'
    model = load_model(net)
    
    adv_type = 'PGD10'
    adv_train_set_file = f'precomputed_adv_images/{adv_type}_train.pkl'
    if not os.path.exists(adv_train_set_file):
        train_set = torchvision.datasets.CIFAR10(root='../cifar-data', train=True, download=True)
        cereate_adversarial_dataset(model, train_set, adv_type, adv_train_set_file)
        
    batch_size = 2
    train_set = torchvision.datasets.CIFAR10(root='../cifar-data', train=True, download=True)
    clean_images = train_set.data
    clean_images = clean_images.astype(np.float32) / 255

    clean_set = GaussianNewtonFeatureDataset(model, clean_images)
    clean_loader = torch.utils.data.DataLoader(clean_set, batch_size=batch_size, shuffle=True, num_workers=0)

    with open(adv_train_set_file, 'rb') as IN:
        adv_images = pickle.load(IN)['adv_images']
    adv_set = GaussianNewtonFeatureDataset(model, adv_images)
    adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=batch_size, shuffle=True, num_workers=0)
    
    for clean_images, adv_images in zip(clean_loader, adv_loader):
        print(clean_images.shape)
        print(adv_images.shape)
        break