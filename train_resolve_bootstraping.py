#!/usr/bin/env python3

import argparse
import csv
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type (default: ResNet18)')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    best_acc = 0.0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # 设置随机种子
    if args.seed != 0:
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    # 数据预处理
    print('==> Preparing data..')
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 这里可以把 download=True，自动下载 CIFAR10 数据集
    trainset = datasets.CIFAR10(
        root='~/data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8  # 或改成0
    )

    testset = datasets.CIFAR10(
        root='~/data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8  # 或改成0
    )

    # 创建模型
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_path = './checkpoint/ckpt.t7' + args.name + '_' + str(args.seed)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net = checkpoint['net']
        net = net.to(device)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        net = models.__dict__[args.model]()
        net = net.to(device)

    if use_cuda:
        net = nn.DataParallel(net)
        cudnn.benchmark = True
        print(f'Using {torch.cuda.device_count()} CUDA devices..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.decay)

    if not os.path.isdir('results'):
        os.mkdir('results')

    logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
               + str(args.seed) + '.csv')

    # mixup
    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # 训练函数
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0.0
        reg_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, args.alpha, use_cuda)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1),
                            reg_loss/(batch_idx+1),
                            100.*correct/total,
                            correct, total))
        return (train_loss/(batch_idx+1), reg_loss/(batch_idx+1), 100.*correct/total)

    # 测试函数
    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().float()

                progress_bar(batch_idx, len(testloader),
                             'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1),
                                100.*correct/total,
                                correct, total))
        acc = 100.*correct/total
        if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc
        return (test_loss/(batch_idx+1), acc)

    def checkpoint(acc, epoch):
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7' + args.name + '_' + str(args.seed))

    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 如果没有日志文件，就新建并写入表头
    if not os.path.exists(logname):
        with open(logname, 'w', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc'])

    # 开始训练
    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch,
                                train_loss,
                                reg_loss,
                                train_acc.item(),
                                test_loss,
                                test_acc.item()])


# ----------------------
# Windows 平台需加此判断
# ----------------------
if __name__ == '__main__':
    # 如果需要冻制（freeze），可调用 freeze_support()，一般情况不需要
    # freeze_support()
    main()
