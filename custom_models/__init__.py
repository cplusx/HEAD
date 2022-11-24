from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .resnet import *
from .resnext import *
from .densenet import *
from .googlenet import *
from .mobilenet import *
from .shufflenet import *
from .preact_resnet import *
from .wide_resnet import *
from .small_cnn import *
from .wrn_madry import *
from .non_linear_resnet import *
from .split_vgg import *
from .split_resnet import *

def get_model(net):
    if net == "resnet18":
        model = ResNet18().cuda()
    if net == "resnet101":
        model = ResNet101().cuda()
    if net == "vgg16":
        model = vgg16().cuda()
    if net == "vgg19":
        model = vgg19().cuda()
    if net == "WRN":
        model = Wide_ResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.).cuda()
    if net == 'WRN_madry':
        model = Wide_ResNet_Madry(depth=32, num_classes=10, widen_factor=10, dropRate=0.).cuda()
    if net == 'densenet121':
        model = DenseNet121().cuda()
    if net == 'densenet201':
        model = DenseNet201().cuda()
    if net == 'lenet':
        model = LeNet().cuda()
    return model