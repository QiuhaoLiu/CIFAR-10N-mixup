from __future__ import print_function

import torch.utils
import torch.utils.data
# -*- coding:utf-8 -*-

'''代码原型mixup-cifar10-main\train_modified_GPT.py

这个文件主要是实现以下功能
1. 加载cifar-N数据集，多个档位的标签√
数据集储存在CIFAR-10_human.pt和CIFAR-100_human.pt，包括了图片和需要几个noise rate的标签
2. implement，confident learning
现在使用最简单可行的做法（方法一），只需拿到概率，用cleanlab的find_label_issues
不需要让cleanlab接管整个训练过程
'''
#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
'''代码修改自mixup-cifar10-main\train.py
在原有基础上，把CIFAR-10数据集替换成了CIFAR-N
'''


import argparse
import csv
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms #图像
import torchvision.datasets as datasets #导入Toronto的torchvision.datasets

import models
from utils import progress_bar

from data.datasets import input_dataset #这个输入的数据集是CIFARN，data文件夹是直接从C:\Users\Qiuhao Liu\OneDrive\Desktop\Jiaheng论文及代码\code\cifar-10-100n-main\data，粘贴过来的
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore") #不想看到过多的警告可以启用该函数
import random
from sklearn.model_selection import StratifiedKFold

import cleanlab
from cleanlab.classification import CleanLearning
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab.internal.util import value_counts
from cleanlab.internal.latent_algebra import compute_inv_noise_matrix
from cleanlab.filter import find_label_issues


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    '''Input: 
        x: 输入图像张量，形状为 (batch_size, channels, height, width)
        y: 输入标签张量，形状为 (batch_size,)。
        alpha: Beta 分布的参数，控制混合比例。
        use_cuda: 是否使用 CUDA 加速
    '''
    '''Output:
        mixed_x: 混合后的图像，形状为 (batch_size, channels, height, width)
        y_a: 原始标签，形状为 (batch_size,)。
        y_b: 生成的随机index所对应的标签，形状为 (batch_size,)。
        lam: 混合比例
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) #混合比列lambda是从beta分布中随机抽取的值
    else:
        lam = 1

    #我现在先不用mixup，只用resnet做交叉验证
    lam = 1
    batch_size = x.size()[0] #128
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda() #index大小为batch_size#生成一个为从0到batch_size-1的随机张量，便于之后的随机混合
    else:
        index = torch.randperm(batch_size)
    
    #把该batch的图像，与该batch的图像做随机的mixup，index，代表的是图像的mixup
    #mixup做的是什么操作？是简单的把图像乘上一个对应的系数，做mixup
    mixed_x = lam * x + (1 - lam) * x[index, :] #size (batch_size, channel , W, H)
    y_a, y_b = y, y[index] #size (batchsize 128) 
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''计算Mixup损失
    Input:
    criterion: 损失函数，在这里是交叉熵损失。
    pred: 模型预测结果
    y_a: 原始标签
    y_b: 被混合的标签
    lam: 混合比例
    Output: 
    将两个标签按照损失比例加权计算损失值
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_probs(model, loader, use_cuda = True):
    '''
    用于把holdout集做out-of-sample推断并返回概率，返回softmax概率矩阵（N, num_classes)
    input:
    model = 模型
    loader = 一个loader的样本

    Output:
    '''
    model.eval()
    all_probs = [] #用于保存对应的概率
    all_indices = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, idxs) in enumerate(loader):#这里的targets对应的noisy_labels 因为__getitem__ return img, noisy_label, true_label, index
            if use_cuda:
                inputs = inputs.cuda()
            outputs = model(inputs) # shape = (batch_size, 10)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy() #把输出做softmax
            all_probs.append(probs)
            all_indices.append(idxs.numpy()) # 需要保存holdout样本的全局索引
    pred_probs = np.concatenate(all_probs, axis=0) # shape = (holdout_size, 10)
    holdout_idx = np.concatenate(all_indices, axis=0) # shape = (holdout_size,)
    return pred_probs, holdout_idx



def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')#学习率，默认为0.1
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint') #是否从检查点恢复，如果命令行中包含--resume, 则设置为True
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type (default: ResNet18)')#模型类型，原来默认模型为18，现在改为默认为Resnet50
    parser.add_argument('--name', default='0', type=str, help='name of run')#运行的名称，默认为“0”，用于日志和检查点的命名
    parser.add_argument('--seed', default=0, type=int, help='random seed')#随机种子，默认为128，保证结果可复现
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')#批量大小，默认为128
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')#总训练的轮数，默认为200
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')#是否使用数据增强，默认为True
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    

    # For CIFAR-N
    #parser from C:\Users\Qiuhao Liu\OneDrive\Desktop\Jiaheng论文及代码\code\cifar-10-100n-main\main.py
    parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='rand1') #在使用cifar100数据集的时候，noise_type只能从clean100和noise100中选择
    parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
    parser.add_argument('--is_human', action='store_true', default=False)#

    # ============ K 折相关参数 =============
    parser.add_argument('--cvn', type=int, default=5, help='number of folds, 0 means no fold')
    parser.add_argument('--cv', type=int, default=0, help='which fold index to run (0 ~ cvn-1)')
    parser.add_argument('--combine-folds', action='store_true', default=False,
                        help='just combine saved fold_{k}_probs.npy to make pyx.npy, then run find_label_issues')

    args = parser.parse_args()
    


    if args.seed != 0:
        torch.manual_seed(args.seed)#如果--seed参数不为0，则使用制定的种子值
        #如果--seed为0，则不设置种子，使用默认的随机状态

    # 如果要合并folds，直接执行combine逻辑后，退出
    if args.combine_folds:
        combine_folds(args)
        return
    # 否则正常跑K折train + holdout inference
    run_crossval_fold(args)

###############################################
# NEW: 合并fold
###############################################
def combine_folds(args):
    '''
    只负责读取fold_{k}_probs.npy 文件，并构建成一个(N, 10)的pyx.npy
    不做任何训练
    '''
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    args.noise_type = noise_type_map[args.noise_type]    
    #load dataset
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './mixup-cifar10-main/data/CIFAR-10_human.pt' #./代表源文件所在的文件夹，../代表源文件所在的文件夹的上一级文件夹
        elif args.dataset == 'cifar100':
            args.noise_path = './mixup-cifar10-main/data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')
            
    train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
    pyx = np.zeros((num_training_samples, num_classes), dtype=np.float32)

    print(f"Combining fold_{k}_probs.npy for k in [0..{args.cvn-1}] => pyx.npy")
    for k in range(args.cvn):
        fn = f"fold_{k}_probs.npy"
        if not Path(fn).exists():
            raise FileNotFoundError(f"Cannot find file {fn}. Please run with --cv={k} first.")
        data_k = np.load(fn, allow_pickle=True).item() # dict: {"probs":..., "holdout_idx":...}
        probs = data_k["probs"] # shape=(holdout_size, 10)
        holdout_idx = data_k["holdout_idx"] # shape = (holdout_size,)
        for i, idx in enumerate(holdout_idx):
            pyx[idx] = probs[i]


###############################################
# NEW: 核心 - 只训练 K-1 folds, holdout 1 fold => 推断 => fold_{cv}_probs.npy
###############################################    

def run_crossval_fold(args):

    use_cuda = torch.cuda.is_available()

    if args.cvn < 2:
        print("cvn < 2 => no cross-validation")
        return
    if args.cv is None:
        raise ValueError("Must specify which fold by --cv=0..cvn-1")
    
    #set_seed
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    
    # Data
    print('==> Preparing data..')
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), #随机剪裁
            transforms.RandomHorizontalFlip(), #随机翻转
            #以上是数据增强，增强数据的多样性，帮组模型更好地泛化
            transforms.ToTensor(),#将PIL图像或NumPy转换形状为（C,H,W)的torch.Tensor，并将像素值归一化到[0,1]
            transforms.Normalize((0.4914, 0.4822, 0.4465), #均值
                                 (0.2023, 0.1994, 0.2010)), #方差
                                 #加快训练收敛
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #现在我改为加载的是CIFARN，带人工标注的；而不是CIFAR
    #训练集，是来自cifar-10-100n-main\main.py
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    args.noise_type = noise_type_map[args.noise_type] 

    #load dataset
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './mixup-cifar10-main/data/CIFAR-10_human.pt' #./代表源文件所在的文件夹，../代表源文件所在的文件夹的上一级文件夹
        elif args.dataset == 'cifar100':
            args.noise_path = './mixup-cifar10-main/data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')
            
    train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
    '''
    train_dataset.train_dataset.shape = (50000, 32, 32, 3) = (Number of Trainning Image, W, H, Channel)
    train_dataset.train_labels = (50000, 1) : Ground true clean label，与args.noise_type对比，计算错误率的
    train_dataset.train_noisy_labels = (50000, 1) : 数据标签，根据args.noise_type加载对应档位的标签

    test_dataset.test_data.shape = (10000, 32, 32, 3)
    test_dataset.test_labels = (10000, 1)
    '''

    # 取出 train_dataset 里的labels 做K折
    y_full = train_dataset.train_noisy_labels # shape = (N,)

    from torch.utils.data import Subset
    skf = StratifiedKFold(n_splits=args.cvn, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.arange(len(y_full)),y_full))

    #如果fold的折数不符合规定
    fold_idx = args.cv
    if fold_idx < 0 or fold_idx >= args.cvn:
        raise ValueError(f"--cv={fold_idx} out of range [0..{args.cvn-1}]")
    
    train_idx, holdout_idx = folds[fold_idx]
    print(f"=====Crossval Fold {fold_idx}/{args.cvn}====")
    print(f"Train size:{len(train_idx)}, holdout size:{len(holdout_idx)}")

    #分别把训练集和验证集从train_dataset中拿出来
    train_subset = Subset(train_dataset, train_idx)
    holdout_subset = Subset(train_dataset, holdout_idx)


    # noise_prior = train_dataset.noise_prior
    # noise_or_not = train_dataset.noise_or_not
    train_loader = torch.utils.data.DataLoader(dataset = train_subset, # 只训练K-1fold
                                    batch_size = 32,
                                    num_workers=1,
                                    shuffle=True)
    holdout_loader = torch.utils.data.DataLoader(dataset = holdout_subset,
                                                 batch_size=32,
                                                 num_workers=args.num_workers,
                                                 shuffle=False
                                                 )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = 32,
                                    num_workers=args.num_workers,
                                    shuffle=False)    
        
    #训练集 这是本代码自带的
    # trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
    #                             transform=transform_train)
    # #root制定数据集的位置，位于data;  train制定的是训练集还是测试集; Download确定是否下载; transform接受的是图像预处理和增强操作
    # #trainset的shape也是 =  (50000, 32, 32, 3) = (Number of Trainning Image, W, H, Channel)
    # trainloader = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=args.batch_size,
    #                                           shuffle=True, num_workers=8)


    #测试集
    # testset = datasets.CIFAR10(root='~/data', train=False, download=False,
    #                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
    #                                          shuffle=False, num_workers=8)

    # Model
    if args.resume:
        # Load checkpoint.
        #是否从现有的checkpoint加载
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                                + str(args.seed))
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        #如果不恢复的话，重新建立模型
        print('==> Building model..')
        net = models.__dict__[args.model]()


    '''保存为csv文件，路径为results文件夹下方，命名方式为log_net_x'''
    if not os.path.isdir('results'):
        os.mkdir('results')
    current_date = datetime.now().strftime("%Y%m%d")
    logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
            + str(args.seed)+ '_' + args.noise_type + '_' + str(args.alpha) + '_' + current_date + '_'  + '.csv') #如果alpha是1.0就是完全的Mixup

    if use_cuda:
        net.cuda()#将模型移动到GPU
        net = torch.nn.DataParallel(net)#启用并行计算，如果可用的话
        print(torch.cuda.device_count())#打印设备号
        cudnn.benchmark = False #benchmark模式，自动寻找当前配置最高效的算法
        torch.backends.cudnn.deterministic = True

        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()#用交叉熵函数作为损失函数，适用于分类任务
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, 
                          weight_decay=args.decay)#使用随机梯度下降作为优化器

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    def train(epoch):
        '''返回训练损失，正则损失，以及训练正确率'''
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0.0
        reg_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (inputs, targets, _, idxs) in enumerate(train_loader): #inputs: torch.Size(batch_size, channel , W, H); targets: torch.Size(batch_size)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                        args.alpha, use_cuda)
            
            outputs = net(inputs) #inputs=(batch_size, channel , W, H)，混合后的图像，outputs = torch.Size([128, 10])= (batch_size, class_num)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam) #计算损失
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) #获取预测结果
            total += targets.size(0) #累加总样本数量
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()) #统计预测正确的结果数目，无论是targets_a还是targets_b都算，按照混合比例加权

            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播
            optimizer.step() #更新模型参数

            progress_bar(batch_idx, len(train_loader),
                        'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                            100.*correct/total, correct, total))
        return (train_loss/(batch_idx+1), reg_loss/(batch_idx+1), 100.*correct/total)

    def test(epoch):
        '''返回测试损失和测试准确率'''
        nonlocal best_acc
        net.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(test_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                progress_bar(batch_idx, len(test_loader),
                            'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total,
                                correct, total))
        acc = 100.*correct/total
        if epoch == start_epoch + args.epoch - 1 or acc > best_acc: #如果达到最后一个epochs，或者准确率超过最佳准确率，则保存checkpoint
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc
        print(f"Best accuracy is:{best_acc:.2f}")
        if epoch == start_epoch + args.epoch - 1:
            print(f"Best accuracy is:{best_acc:.2f}")#如果达到最后一个epochs打印最佳准确率
        return (test_loss/(batch_idx+1), 100.*correct/total)

    def checkpoint(acc, epoch):
        # Save checkpoint.
        '''保存检查点，创建字典，包括模型参数 (net)、当前准确率 (acc)、当前 epoch (epoch)、以及随机数生成器状态 (rng_state)'''
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './mixup-cifar10-main/checkpoint/ckpt.t7' + args.name + '_'
                + str(args.seed))

    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile: 
            logwriter = csv.writer(logfile, delimiter=',') #用于写入csv文件
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc']) #写入表头

    #Start Training
    for epoch in range(start_epoch, args.epoch):

        train(epoch)
        #test(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train_loss, reg_loss, train_acc = train(epoch) #训练
        # test_loss, test_acc = test(epoch) #测试
        # adjust_learning_rate(optimizer, epoch) #调整学习率
        # with open(logname, 'a') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',')
        #     logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
        #                         test_acc]) #记录对应数据

    print(f"Finished training fold = {fold_idx}, now do out-of-sample inference on holdout.")
    probs, holdout_idx =get_probs(net, holdout_loader, use_cuda=True)
    out = {"probs": probs, "holdout_idx":holdout_idx}
    np.save(f"fold_{fold_idx}_probs.npy",out, allow_pickle=True)
    print(f"Saved fold_{fold_idx}_probs.npy. Done.")

if __name__ == '__main__':
    main()
