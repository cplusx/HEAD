from custom_models import *
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import hessian, jacobian

def get_model(net, n_classes, cuda=True):
    if net == "resnet18":
        model = ResNet18(num_classes=n_classes)
    if net == "resnet101":
        model = ResNet101(num_classes=n_classes)
    if net == "vgg16":
        model = vgg16(num_classes=n_classes)
    if net == "vgg19":
        model = vgg19(num_classes=n_classes)
    if net == "WRN":
        model = Wide_ResNet(depth=34, num_classes=n_classes, widen_factor=10, dropRate=0)
    if net == 'WRN_madry':
        model = Wide_ResNet_Madry(depth=32, num_classes=n_classes, widen_factor=10, dropRate=0)
    if net == 'densenet121':
        model = DenseNet121(num_classes=n_classes)
    if net == 'densenet201':
        model = DenseNet201(num_classes=n_classes)
    if net == 'lenet':
        model = LeNet(num_classes=n_classes)
    if cuda:
        model = model.cuda()
    return model

def compute_gaussian_newton(model, image, target):
    if isinstance(target, list):
        target = torch.tensor(target).to(image.device)
    elif isinstance(target, int):
        target = torch.tensor([target]).to(image.device)
    model.eval()
    x = image.clone().detach()
    x.requires_grad_()
    model.zero_grad()
    
    def forward_func(x):
        with torch.enable_grad():
            logits = model(x)
        return logits
    def loss_func(logits):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss(reduction="mean")(logits, target)
        return loss
    
    logits = forward_func(x)
    
    logits_hessian = hessian(loss_func, logits).clone().detach()
    jacobian_logits_input = jacobian(forward_func, x).clone().detach()
    # shape, logits_hessian: bs, 10, bs, 10. jacobian: bs, 10, bs, |3x32x32|
    bs, num_logits, _, input_dim, input_h, input_w = jacobian_logits_input.shape
    assert bs == 1, 'plz ensure batch size is 1'
    jacobian_logits_input = jacobian_logits_input.view(num_logits, input_dim*input_h*input_w)
    logits_hessian = logits_hessian.view(num_logits, num_logits)
    res = jacobian_logits_input.t() @ logits_hessian @ jacobian_logits_input # |input|, |input|
    return res.cpu().detach()


def compute_hessian_of_logits(model, image, target):
    image = image.cuda()
    if isinstance(target, list):
        target = torch.tensor(target).cuda()
    elif isinstance(target, int):
        target = torch.tensor([target]).cuda()
    model.eval()
    x = image.clone().detach()
    x.requires_grad_()
    model.zero_grad()
    
    def forward_func(x):
        with torch.enable_grad():
            logits = model(x)
        return logits
    def loss_func(logits):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss(reduction="mean")(logits, target)
        return loss
    
    logits = forward_func(x)
    
    logits_hessian = hessian(loss_func, logits).clone().detach()
    return logits_hessian.cpu().detach()

def compute_gradients(model, image, target):
    image = image.cuda()
    if isinstance(target, list):
        target = torch.tensor(target).cuda()
    elif isinstance(target, int):
        target = torch.tensor([target]).cuda()
    model.eval()
    x = image.clone().detach()
    x.requires_grad_()
    model.zero_grad()
    
    with torch.enable_grad():
        logits = model(x)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, target)
    loss.backward()
    grad = x.grad.clone().detach().cpu()
    return grad
    

# def matrix_modulus(x):
#     return torch.sum(x).cpu().detach().numpy()

def matrix_modulus(
    x, 
    mode='modulus', # modulus or resized
    modulus_mode='l1',
    resize_mode=32,
):
    if mode == 'modulus':
        if modulus_mode == 'l1':
            return torch.sum(torch.abs(x)).cpu().detach().numpy()
        elif modulus_mode == 'l2':
            return torch.sqrt(torch.sum(x**2)).cpu().detach().numpy()
        else:
            raise RuntimeError('Unknown modulus mode {}'.format(modulus_mode))
    elif mode == 'resized':
        # print('call resize', resize_mode)
        gn = x
        gn_size = gn.shape[0]
        assert gn_size % resize_mode == 0
        bin_size = gn_size // resize_mode
        
        # fast resize
        gn = gn.reshape((resize_mode, bin_size, resize_mode, bin_size)).mean(3).mean(1)
        return gn
    else:
        raise RuntimeError
