#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse #解析命令行参数
import csv #用于CSV格式的读写操作
import os #文件路径处理

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn #包含神经网络各类层和损失函数等模块
import torch.optim as optim #优化器模块，包括SGD，Adam
import torchvision.transforms as transforms
import torchvision.datasets as datasets 

import models
from utils import progress_bar #进度条

#创建参数解析器，用于处理命令行参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#添加参数
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run') #总共需要训练多少个周期
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)') #插值系数，alpha
args = parser.parse_args()

use_cuda = torch.cuda.is_available() #检查是否有可用的设备

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#如果传入的随机种子不为0，则从对应的随机种子开始，为了可重复性
if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
#根据args.augment的值，来决定训练数据的预处理和增广方式
#如果augment=1,裁剪，随机水平翻转，标准化
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
#如果augment=0，仅仅标准化
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
#标准化的值是固定的吗？(0.4914, 0.4822, 0.4465)这些数字是怎么计算出来的

#测试集数据仅仅进行标准化处理，不进行数据增强
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#加载数据集，地址为与data，加载的是训练集，不用自动下载，
trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
#这里加载的是CIFAR10训练集对应的label，按照顺序排列的
#在这里如果我想导入的数据集是CIFARN我应该怎么操作呢？

#用Dataloader包装训练集                          
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)


# Model
if args.resume: #根据args.resume判断是否需要从检查点恢复训练
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else: #否则根据传入的args.model从models中实例化一个网络
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv') #定义日志文件名称，包含模型类名，运行名称，随机种子信息

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net) #多GPU上并行训练
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss() #定义损失函数，交叉熵函数
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)
#定义优化器，随机梯度下降Stochastic Gradient Descent(SGD)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    #如果alpha不为0，从beta分布中随机抽取一个数作为lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    #把该batch的图像，与该batch的图像做随机的mixup，index，代表的是图像的mixup
    #mixup做的是什么操作？是简单的把图像乘上一个对应的系数，做mixup
    mixed_x = lam * x + (1 - lam) * x[index, :]#size (batch_size, channel , W, H)
    y_a, y_b = y, y[index]#size (batch_size 128)
    return mixed_x, y_a, y_b, lam

#混合的损失函数的定义，把预测值与两个混合的值做交叉熵计算，再加权求和
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0 #正则损失
    correct = 0
    total = 0

    #bootsraping报错是怎么解决的？
    #An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
    #This probably means that you are not using fork to start your child processes and you have forgotten to use the proper idiom.
    #怎么解决上述问题？使用 if __name__ == '__main__': 来保护主运行代码
    #那都有哪些函数应该写到main()里面去呢？
    #GPT把除了mixup_data和mixup_criterion以外的所有数据函数都写进了main()函数里
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        #map函数有什么用？对每个元素，应用一个制定的函数。在这里是转化为variable
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播清空梯度
        optimizer.step() #更新参数

        #终端显示训练进度
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc: #如果是最后一个epoch，或者当前准确度超过历史最佳则保存当前的checkpoint
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc #更新best_acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    } #将模型状态，准确率，当前的epoch，随机数状态打包成字典
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') #如果没有checkpoint目录就创建
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed)) #保存字典

#调整学习率，递减
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#创建日志文件
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

#主训练循环
for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',') #将数据写入csv文件
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])