#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# 这个文件用于跑Docta 纯Mixup和不Mixup的
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms #图像
import torchvision.datasets as datasets
import sys
import models
from utils import progress_bar

from data.datasets import input_dataset #这个输入的数据集是CIFARN，data文件夹是直接从C:\Users\Qiuhao Liu\OneDrive\Desktop\Jiaheng论文及代码\code\cifar-10-100n-main\data，粘贴过来的
from datetime import datetime


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
    # if alpha > 0:
    #     lam = np.random.beta(alpha, alpha) #混合比列lambda是从beta分布中随机抽取的值
    # else:
    #     lam = 1
    lam = 1 #这次跑的代码，不混合
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')#学习率，默认为0.1
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint') #是否从检查点恢复，如果命令行中包含--resume, 则设置为True
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type (default: ResNet18)')#模型类型，默认为Resnet18
    parser.add_argument('--name', default='0', type=str, help='name of run')#运行的名称，默认为“0”，用于日志和检查点的命名
    parser.add_argument('--seed', default=0, type=int, help='random seed')#随机种子，默认为128，保证结果可复现
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')#批量大小，默认为128
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')#总训练的轮数，默认为200
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')#是否使用数据增强，默认为True
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0., type=float,
                        help='mixup interpolation coefficient (default: 1)') #
    parser.add_argument('--use_docta',help='use Docta as ground true label', default=True) #是否加载Docta的cured_label覆盖原来的clean label

    #parser from C:\Users\Qiuhao Liu\OneDrive\Desktop\Jiaheng论文及代码\code\cifar-10-100n-main\main.py
    parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='rand1')
    parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--is_human', action='store_true', default=False)#
    

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed != 0:
        torch.manual_seed(args.seed)#如果--seed参数不为0，则使用制定的种子值
        #如果--seed为0，则不设置种子，使用默认的随机状态

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
    #train_dataset.train_dataset.shape = (50000, 32, 32, 3) = (Number of Trainning Image, W, H, Channel)

   
    
    if(args.use_docta):
        o_path = os.getcwd()
        print(o_path)
        sys.path.append(o_path) # set path so that modules from other foloders can be loaded
        cured_labels = torch.load('./docta-master/results/CIFAR_c10/cured_labels_CIFAR_c10.pt')#记载pt文件的方式，
        ground_true_labels = train_dataset.train_labels
        noisy_label = train_dataset.train_noisy_labels
        # 4) 计算 ground_true_labels 与 cured_labels 有多少是相同的
        same_count = 0
        for gt_label, cd_label in zip(ground_true_labels, cured_labels):
            if gt_label == cd_label:
                same_count += 1
        #计算rand1 与 ground true之间有多少不同
        same_count_with_rand1 = 0
        for gt_label, cd_label in zip(ground_true_labels, noisy_label):
            if gt_label == cd_label:
                same_count_with_rand1 += 1
        # 计算cured 与 rand1之间有多少不同
        same_count_between_cured_and_rand1 = 0
        for gt_label, cd_label in zip(cured_labels, noisy_label):
            if gt_label == cd_label:
                same_count_between_cured_and_rand1 += 1                
        print(f"Number of same labels between ground_true_labels and cured_labels: {same_count}")
        print(f"Total labels: {len(ground_true_labels)}")
        print(f"Total labels: {len(cured_labels)}")
        print(f"Same ratio between cured and ground true: {100.0 * same_count / len(ground_true_labels):.2f}%")
        print(f"Same ratio between rand1 and ground true: {100.0 * same_count_with_rand1 / len(ground_true_labels):.2f}%")
        print(f"Same ratio between rand1 and cured: {100.0 * same_count_between_cured_and_rand1 / len(ground_true_labels):.2f}%")
        # cured_labels int类型，len(50000)
        print("Cured CIFAR10 train set is ready.")      
        print(f"Sample train_labels before replacement: {train_dataset.train_labels[:10]}")   
        train_dataset.train_labels = cured_labels.tolist()  # 假设 train_labels 是列表
        print(f"Sample train_labels after replacement: {train_dataset.train_labels[:10]}")

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = 128,
                                    num_workers=8,
                                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = 64,
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
            + str(args.seed) + args.noise_type + '_' + str(args.alpha) + '.csv') #如果alpha是1.0就是完全的Mixup, alpha=0.0就是不做Mixup

    if use_cuda:
        net.cuda()#将模型移动到GPU
        net = torch.nn.DataParallel(net)#启用并行计算，如果可用的话
        print(torch.cuda.device_count())#打印设备号
        cudnn.benchmark = True#benchmark模式，自动寻找当前配置最高效的算法
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()#用交叉熵函数作为损失函数，适用于分类任务
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, 
                          weight_decay=args.decay)#使用随机梯度下降作为优化器

    def train(epoch):
        '''返回训练损失，正则损失，以及训练正确率'''
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0.0
        reg_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (inputs, targets, _) in enumerate(train_loader): #inputs: torch.Size(batch_size, channel , W, H); targets: torch.Size(batch_size)
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
        print("Best accuracy is: %.4f", best_acc)
        if epoch == start_epoch + args.epoch - 1:
            print(f"Best accuracy is: {best_acc:.2f}")#如果达到最后一个epochs打印最佳准确率
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

    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch) #训练
        test_loss, test_acc = test(epoch) #测试
        adjust_learning_rate(optimizer, epoch) #调整学习率
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc]) #记录对应数据

if __name__ == '__main__':
    main()
