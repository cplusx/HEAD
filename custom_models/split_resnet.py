import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import hessian, jacobian

class ResNet18_splitted(nn.Module):
    def __init__(self, model, num_blocks, num_classes=10):
        super(ResNet_splitted, self).__init__()
        self.model = model
        
    def forward(self, x):
        out = self.forward_before_linear(x)
        out = self.model.linear(out)
        return out
    
    def forward_before_linear(self, x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
        
    def compute_input_gaussian_newton(image, target, modulus_mode):
        x = image.clone().detach()
        x.requires_grad_()
        self.zero_grad()
        
        def loss_func(logits):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss(reduction="mean")(logits, target)
            return loss
        
        logits = self.forward(x)
        
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
    
    def compute_logit_gaussian_newton(image, target, modulus_mode):
        x = image.clone().detach()
        x.requires_grad_()
        self.zero_grad()
        
        x = self.forward_before_linear(x)
        
        def forward_func(x):
            with torch.enable_grad():
                x = self.model.linear(x)
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
        gaussian_newton.append(self.compute_input_gaussian_newton(image, target, modulus_mode))
        gaussian_newton.append(self.compute_logit_gaussian_newton(image, target, modulus_mode))
        return gaussian_newton
    
    
def ResNet18_splitted(num_classes=10):
    return ResNet_splitted(BasicBlock, [2,2,2,2], num_classes=num_classes)

