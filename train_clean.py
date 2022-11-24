# train on clean dataset
import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from custom_models import *
import numpy as np
import glob

parser = argparse.ArgumentParser(description='Regular training on clean data')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--net', type=str, default="WRN_madry",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="CIFAR10", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./clean_model_results', help='dir of output')

args = parser.parse_args()

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = os.path.join(args.out_dir, args.dataset, args.net)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def train(model, train_loader, optimizer):
    loss_sum = 0
    correct = 0
    num_sample = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        # calculate standard adversarial training loss
        loss = nn.CrossEntropyLoss(reduction='mean')(output, target)
        loss_sum += loss.item()
        loss.backward()
        correct += (target == torch.argmax(output, -1)).sum()
        num_sample += len(target)
        optimizer.step()

    acc = correct / num_sample
    return loss_sum, acc

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "CIFAR10":
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root='./cifar-data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar-data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "SVHN":
    num_classes = 10
    trainset = torchvision.datasets.SVHN(root='./svhn-data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./svhn-data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "CIFAR100":
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "resnet18":
    model = ResNet18(num_classes=num_classes).cuda()
    net = "resnet18"
if args.net == "resnet101":
    model = ResNet101(num_classes=num_classes).cuda()
    net = "resnet101"
if args.net == "vgg16":
    model = vgg16(num_classes=num_classes).cuda()
    net = "vgg16"
if args.net == "vgg19":
    model = vgg19(num_classes=num_classes).cuda()
    net = "vgg19"
if args.net == "WRN":
  # e.g., WRN-34-10
    model = Wide_ResNet(depth=args.depth, num_classes=num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.net == 'WRN_madry':
  # e.g., WRN-32-10
    args.depth = 32
    model = Wide_ResNet_Madry(depth=args.depth, num_classes=num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.net == 'densenet121':
    model = DenseNet121(num_classes=num_classes).cuda()
    net = args.net
if args.net == 'densenet201':
    model = DenseNet201(num_classes=num_classes).cuda()
    net = args.net
if args.net == 'lenet':
    model = LeNet(num_classes=num_classes).cuda()
    net = args.net
print(net)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


# auto Resume
ckpts = glob.glob(f'{out_dir}/epoch_*.pth.tar')
if len(ckpts) > 0:
    ckpts.sort(key=os.path.getmtime) # use the latest ckpt to resume if possible
    ckpt = ckpts[-1]
    checkpoint = torch.load(ckpt)
    print('resume from ckpt')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print('==> start from scratch')
    start_epoch = 0
    
for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch + 1)
    train_loss, acc = train(model, train_loader, optimizer)
    print(acc)
    if (epoch+1) % 10 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': acc.item(),
        }, filename='epoch_{}.pth.tar'.format(epoch+1))