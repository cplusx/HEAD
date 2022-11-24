import numpy as np
import torch
from torch import nn
from torch.autograd.functional import hessian, jacobian

class VGG16_splitted(nn.Module):
    def __init__(self, model):
        super(VGG16_splitted, self).__init__()
        self.split_model(model)
        
    def forward(self, x):
        for k, part in self.vgg_parts.items():
            x = part(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_part['classifier'](x)
        return x
    
    def split_model(self, model):
        features = model.features
        classifier = model.classifier
        self.vgg_parts = nn.ModuleDict()
        self.classifier_part = nn.ModuleDict()
        part_idx = 0
        last_idx = 0
        for layer_idx, layer in enumerate(features):
            if isinstance(layer, torch.nn.ReLU):
                self.vgg_parts[f'vgg_part{part_idx}'] = features[last_idx: layer_idx+1]
                last_idx = layer_idx + 1
                part_idx += 1
        if last_idx < layer_idx:
            self.vgg_parts[f'vgg_part{part_idx}'] = features[last_idx: layer_idx+1]
            
        self.classifier_part['classifier'] = classifier
        
    def compute_one_gaussian_newton(self, start_idx, image, target, modulus_mode):
        x = image.clone().detach()
        x.requires_grad_()
        self.zero_grad()
        
        if start_idx > 0:
            for idx in range(start_idx):
                x = self.vgg_parts[f'vgg_part{idx}'](x)
        
        def forward_func(x):
            with torch.enable_grad():
                for idx in range(start_idx, len(self.vgg_parts)):
                    x = self.vgg_parts[f'vgg_part{idx}'](x)
                x = x.view(x.size(0), -1)
                x = self.classifier_part['classifier'](x)
            return x
        def loss_func(logits):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss(reduction="mean")(logits, target)
            return loss
        
        logits = forward_func(x)
        
        logits_hessian = hessian(loss_func, logits).clone().detach()
        jacobian_logits_input = jacobian(forward_func, x).clone().detach()
        bs, num_logits, _, input_dim, input_h, input_w = jacobian_logits_input.shape
        assert bs == 1, 'plz ensure batch size is 1'
        jacobian_logits_input = jacobian_logits_input.view(num_logits, input_dim*input_h*input_w)
        logits_hessian = logits_hessian.view(num_logits, num_logits)
        m1 = jacobian_logits_input.t() @ logits_hessian
        m2 = jacobian_logits_input
        num_logits, num_ele = m2.shape
        res = 0
        for i in range(num_ele):
            this_m = m1 @ m2[:, i:i+1]
            if modulus_mode == 'l1':
                res += torch.sum(torch.abs(this_m))
            elif modulus_mode == 'l2':
                res += torch.sum(this_m**2)
        if modulus_mode == 'l2':
            res = torch.sqrt(res)
        return res.cpu().detach()
        
    def compute_layerwise_gaussian_newton(self, image, target, modulus_mode='l1'):
        self.eval()
        if isinstance(target, list):
            target = torch.tensor(target).to(image.device)
        elif isinstance(target, int):
            target = torch.tensor([target]).to(image.device)
        
        gaussian_newton = []
        for start_idx in range(len(self.vgg_parts)-1):
            gaussian_newton.append(self.compute_one_gaussian_newton(start_idx, image, target, modulus_mode))
        return gaussian_newton