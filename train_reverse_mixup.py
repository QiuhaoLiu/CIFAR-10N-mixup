'''代码修改自mixup-cifar10-main/train_selective_mixup_docta.py
在原有代码的基础上把
1. reverse selective mixup的方式
label: 1 + lambda : -lambda
data: lambda : 1 - lambda
2. mixup的timing
1-150 epochs 不做任何mixup
151-200 epochs 做reverse selective Mixup，并且学习率调整为1e-6

还是在Docta label各个档位上进行训练（rand1，worst)

'''
#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms #图像
import torchvision.datasets as datasets

import models
from utils import progress_bar

from data.datasets import input_dataset #这个输入的数据集是CIFARN，data文件夹是直接从C:\Users\Qiuhao Liu\OneDrive\Desktop\Jiaheng论文及代码\code\cifar-10-100n-main\data，粘贴过来的
from collections import defaultdict #字典

def get_mismatch_indices(noisy_labels, true_labels):
    """
    返回 noisy_labels 和 true_labels 中不相同元素的索引

    参数:
        noisy_labels (list): 带噪声的标签
        true_labels (list): 真实标签

    返回:
        list: 不相同元素的索引列表
    """
    #list转化为NumPy数组
    noisy_labels = np.array(noisy_labels)
    true_labels = np.array(true_labels)

    #使用NumPy比较，找到不相同的label所对应的索引
    mismatches = np.where(noisy_labels != true_labels)[0]
    mismatches = mismatches.tolist()
    return mismatches

def get_clean_samples(labels, clean_label_to_indices, clean_train_dataset, use_cuda):
    """
    根据给定的标签列表，从清洁数据集中批量获取样本。

    参数:
        labels (torch.Tensor): 真实标签，形状为 (num_mismatch,)

    返回:
        torch.Tensor: 清洁样本，形状为 (num_mismatch, C, H, W)
    """
    clean_samples = []
    for label in labels:
        label = label.item()
        clean_indices = clean_label_to_indices[label]
        if len(clean_indices) == 0:
            raise ValueError(f"No clean samples found for label {label}")
        random_idx = np.random.choice(clean_indices)
        clean_image, _, _, _ = clean_train_dataset[random_idx]
        if use_cuda:
            clean_image = clean_image.cuda()
        clean_samples.append(clean_image.unsqueeze(0))
    clean_samples = torch.cat(clean_samples, dim=0)
    return clean_samples

def apply_selective_mixup(inputs, noisy_labels, true_labels, indices, mismatch_set,
                          clean_label_to_indices, clean_train_dataset, use_cuda, alpha, do_mix):
    """
    当 do_mix=False 时，相当于不做 Mixup；否则执行“反向”Selective Mixup（数据同普通Mixup，标签配比反向）。
    
    对错标样本进行 Mixup 操作。

    输入参数:
        inputs (torch.Tensor): 当前批次的输入图像，形状为 (batch_size, C, H, W)。
        noisy_labels (torch.Tensor): 当前批次的带噪声标签，形状为 (batch_size,)。
        true_labels (torch.Tensor): 当前批次的真实标签，形状为 (batch_size,)。
        indices (list or torch.Tensor): 当前批次样本的索引，形状为 (batch_size,)。
        mismatch_set (set): 包含所有错标样本索引的集合。
        clean_label_to_indices (defaultdict): 映射每个标签到清洁样本索引的字典。
        clean_train_dataset (Dataset): 清洁训练数据集。
        use_cuda (bool): 是否使用 CUDA。
        alpha (float): Mixup 的 β 分布参数。

    返回:
        mixed_inputs (torch.Tensor): 混合后的输入图像，形状为 (batch_size, C, H, W)。
        y_a (torch.Tensor): 混合后的第一个标签，形状为 (batch_size,)。
        y_b (torch.Tensor): 混合后的第二个标签，形状为 (batch_size,)。
        lam (float): 混合比例 λ。
    """
    # 将索引转换为 list（如果是 Tensor）
    if isinstance(indices, torch.Tensor):
        batch_indices = indices.cpu().numpy().tolist()
    else:
        batch_indices = indices

    # 找出错标样本
    current_mismatch = [i for i in range(len(batch_indices)) if batch_indices[i] in mismatch_set]

    # 初始化
    y_a = noisy_labels.clone()
    y_b = noisy_labels.clone()
    lam = 1.0  # 不做Mixup时，等价于 lam=1

    # 如果 do_mix=True 且存在错标样本 => 执行 Mixup
    if do_mix and current_mismatch:
        # 1) 获取错标样本真实标签
        true_labels_mismatch = true_labels[current_mismatch]
        # 2) 从 clean_dataset 抽取对应样本
        clean_samples = get_clean_samples(true_labels_mismatch, clean_label_to_indices,
                                          clean_train_dataset, use_cuda)
        # 3) 采样 λ
        lam = np.random.beta(alpha, alpha)  # 不再做 max(lam, 1-lam)

        # 4) 数据Mix
        mixed_inputs = lam * inputs[current_mismatch] + (1 - lam) * clean_samples
        inputs[current_mismatch] = mixed_inputs

        # 5) 更新标签: y_b 替换为真实标签
        y_a[current_mismatch] = noisy_labels[current_mismatch]
        y_b[current_mismatch] = true_labels_mismatch

    return inputs, y_a, y_b, lam



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
        lam = 1 #不混合

    batch_size = x.size()[0] #128
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda() #index大小为batch_size #生成一个为从0到batch_size-1的随机张量，便于之后的随机混合
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
    """
    反向Selective Mixup: label => (1 + lam)*y_a + (- lam)*y_b
    data  => lam*x_a + (1-lam)*x_b（在 apply_selective_mixup 中完成）
    """
    loss_a = criterion(pred, y_a)  # CE(pred, y_a)
    loss_b = criterion(pred, y_b)  # CE(pred, y_b)
    # 反向配比: (1 + lam) * loss_a + (-lam) * loss_b
    return (1 + lam) * loss_a + (-lam) * loss_b
    #return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--use_docta',help='use Docta as ground true label', default=True) #是否用Docta的cured_label作为ground true label

    #parser from C:\Users\Qiuhao Liu\OneDrive\Desktop\Jiaheng论文及代码\code\cifar-10-100n-main\main.py
    parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='worst')
    parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--is_human', action='store_true', default=False)#
    
    #parser.add_argument('--true_path', type=, help= 'path of CIFAR-10_')
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
    #load dataset 确定用的是哪一个数据集，CIFAR10还是CIFAR100? 每个数据集里都包括了clean worst aggre rand1...
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './mixup-cifar10-main/data/CIFAR-10_human.pt' #./代表源文件所在的文件夹，../代表源文件所在的文件夹的上一级文件夹
        elif args.dataset == 'cifar100':
            args.noise_path = './mixup-cifar10-main/data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')
            


    # 加载训练和测试数据
    train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
    #train_dataset.train_dataset.shape = (50000, 32, 32, 3) = (Number of Trainning Image, W, H, Channel)
    
    # 加载clean训练数据集
    clean_train_dataset, _, _, _ =  input_dataset(args.dataset, 'clean_label', args.noise_path, args.is_human)

    if(args.use_docta):
        o_path = os.getcwd()
        print(o_path)
        sys.path.append(o_path) # set path so that modules from other foloders can be loaded
        cured_labels = torch.load('./docta-master/results/CIFAR_c10/cured_labels_CIFAR_c10.pt')#记载pt文件的方式，
        # cured_labels int类型，len(50000)
        print("Cured CIFAR10 train set is ready.")  
        
        # 加载cured_label
        #os.chdir('..')
        print(f"Sample train_labels before replacement: {train_dataset.train_labels[:10]}")
        print(f"Sample clean_train_dataset.train_labels before replacement: {clean_train_dataset.train_labels[:10]}")
        # 替换 train_dataset 和 clean_train_dataset 的真实标签为 cured_labels
        train_dataset.train_labels = cured_labels.tolist()  # 假设 train_labels 是列表
        clean_train_dataset.train_labels = cured_labels.tolist()  # 假设 train_labels 是列表
        print(f"Sample cured_labels: {cured_labels[:10]}")  # 打印前10个 cured_labels
        print(f"Sample train_labels after replacement: {train_dataset.train_labels[:10]}")
        print(f"Sample clean_train_dataset.train_labels after replacement: {clean_train_dataset.train_labels[:10]}")


    # 创建标签到索引的映射，比如说label = 6，所对应的图片0， 19， 22....。这个字典len=10，对应的是十个类别，clean_label_to_indices是字典索引
    clean_label_to_indices = defaultdict(list)
    # for idx, clean_label in enumerate (clean_train_dataset.train_labels):
    #     clean_label_to_indices[clean_label].append(idx)
    for idx, (_, clean_label, _, _) in enumerate (clean_train_dataset):
        clean_label_to_indices[clean_label].append(idx)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = 128,
                                    num_workers=8,
                                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = 64,
                                    num_workers=args.num_workers,
                                    shuffle=False)    
    #找到Random1与clean_label标注不一样的部分
    mismatch_indices = get_mismatch_indices(train_dataset.train_noisy_labels, train_dataset.train_labels)
    mismatch_set = set(mismatch_indices) # 转换为集合以便于加速查找 这是什么意思？
        
    
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
    # 如果不恢复checkpoint，则构建模型
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
    logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
            + str(args.seed) + '_' + str(args.noise_type) + '.csv')

    if use_cuda:
        net.cuda()#将模型移动到GPU
        net = torch.nn.DataParallel(net)#启用并行计算，如果可用的话
        print(torch.cuda.device_count())#打印设备号
        cudnn.benchmark = True#benchmark模式，自动寻找当前配置最高效的算法
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()#用交叉熵函数作为损失函数，适用于分类任务
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, 
                          weight_decay=args.decay)#使用随机梯度下降作为优化器

    def get_clean_sample(label):
        '''输入一个类别，随机返回一个该类别的图像'''
        indices = clean_label_to_indices[label]#从字典中查找，得到clean label所对应的indices
        if len(indices) == 0:
            raise ValueError(f"No clean samples found for label {label}")
        random_idx = np.random.choice(indices) #从该类别中随机挑出一个图片做mixup
        clean_image, _, _, _ = clean_train_dataset[random_idx] #在clean训练集中，找到该图像，并返回
        return clean_image.unsqueeze(0)

    def train(epoch):
        '''返回训练损失，正则损失，以及训练正确率'''
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0.0
        reg_loss = 0.0
        correct = 0.0
        total = 0.0
        # ### 修改点 2: 若 epoch < 151 => 不做 mixup; 若 epoch >= 151 => 反向 mixup
        do_mix = (epoch >= 151)  # 只有 >=151 才做 reverse selective mixup

        for batch_idx, (inputs, noisy_labels, true_labels, indices) in enumerate(train_loader): #inputs: img torch.Size(batch_size, channel , W, H); noisy_labels==true_labels: torch.Size(batch_size); ret_index是图片对应的名称索引，比如说img0001
            # true_labels代表的是该批次数据所对应的真实标签？
            if use_cuda:
                inputs= inputs.cuda()
                noisy_labels = noisy_labels.cuda()
                true_labels = true_labels.cuda()

            # selective mixup
            inputs, y_a, y_b, lam = apply_selective_mixup(
                inputs, noisy_labels, true_labels, indices, mismatch_set,
                clean_label_to_indices, clean_train_dataset, use_cuda, args.alpha, do_mix
            )

            # 计算输出和损失
            targets_a = y_a
            targets_b = y_b


            outputs = net(inputs) #inputs=(batch_size, channel , W, H)，混合后的图像，outputs = torch.Size([128, 10])= (batch_size, class_num)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam) #计算损失

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) #获取预测结果
            total += inputs.size(0) #累加总样本数量
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
            for batch_idx, (inputs, targets, _, _) in enumerate(test_loader):
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
            #lr /= 10
            lr = 0.000001
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
