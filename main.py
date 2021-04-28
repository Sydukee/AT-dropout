import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import argparse
import os


from dataset.load_data import create_test_dataset, create_train_dataset
from attack.pgd import IPGD
from attack.sm_pgd import sm_PGD
import torchattacks
from config import args
from utils import create_logger, load_checkpoint, save_checkpoint
from train import train_one_epoch, eval_one_epoch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

if args.net == 'resnet':
    from network.net import create_network
elif args.net == 'vgg':
    from network.net2 import create_network

# 指定 gpu
device = torch.device('cuda:{}'.format(args.gpu))
torch.backends.cudnn.benchmark = True  # True代表自动搜索合适卷积算法

# 创建网络
net = create_network()
net.to(device)

# loss 函数
criterion = torch.nn.CrossEntropyLoss().to(device)

# 优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# learningRate 衰减
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 100], gamma=0.1)

# 加载数据集
train_loader = create_train_dataset(args.batch_size)
val_loader = create_test_dataset(args.batch_size)

# 设置是否对抗训练
if args.at is True:
    train_attack = IPGD(net, eps=8 / 255.0, sigma=2 / 255.0, nb_iter=10, norm=np.inf, device=device,
                          drop_prob=args.drop_prob)
else:
    train_attack = None

# 设置攻击方法
if args.val_method == 'fgsm':
    print('use FGSM attack')
    val_attack = torchattacks.FGSM(net)
elif args.val_method == 'cw':
    print('use cw attack')
    val_attack = torchattacks.CW(net)
else:
    print('use pgd attack')
    val_attack = IPGD(net, eps=8 / 255.0, sigma=2 / 255.0, nb_iter=20, norm=np.inf, device=device, drop_prob=0)

# 创建终端logger
logger = create_logger('./log', 'train', 'info')

# 当前epoch
now_epoch = 0

# 断点
# if args.auto_continue:
#     args.resume = os.path.join(args.model_dir, 'last.checkpoint')
if args.resume is not None and os.path.isfile(args.resume):
    now_epoch = load_checkpoint(args.resume, net, optimizer, lr_scheduler)

# 初始化绘图数据
te_clean_list = []
te_adv_list = []
_te = 0

# 开始训练部分，达到指定epoch退出
while True:
    now_epoch = now_epoch + 1
    if now_epoch > args.epochs:
        break

    # logger输出
    descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(now_epoch, args.epochs, lr_scheduler.get_lr()[0])
    logger.info(f'now_epoch: {now_epoch:.1f}, lr: {lr_scheduler.get_lr()[0]:.2f}')

    # 调用训练函数
    train_one_epoch(net, train_loader, optimizer, criterion, device, descrip_str, train_attack, adv_coef=args.adv_coef,
                    logger=logger)

    # 训练一定epoch后进行eval
    if args.val_interval > 0 and now_epoch % args.val_interval == 0:
        te_clean, te_adv = eval_one_epoch(net, val_loader, device, val_attack, logger=logger)
        te_clean_list.append(te_clean)
        te_adv_list.append(te_adv)
        _te += 1

    # 更新学习率
    lr_scheduler.step()

    # 保存checkpoint
    save_checkpoint(now_epoch, net, optimizer, lr_scheduler,
                    file_name=os.path.join(args.model_dir, 'epoch-{}.checkpoint'.format(now_epoch)))

# 绘制training曲线图
x = range(0, _te)
plt.subplot(2, 1, 1)
plt.plot(x, te_clean_list)
plt.title('te_clean_fig')
plt.xlabel('epoch')
plt.ylabel('te_clean_acc')
plt.subplot(2, 1, 2)
plt.plot(x, te_adv_list)
plt.title('te_adv_fig')
plt.xlabel('epoch')
plt.ylabel('te_adv_acc')

plt.tight_layout()
plt.savefig("acc.jpg")
