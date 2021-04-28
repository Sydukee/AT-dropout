from dataset import load_data
import torchattacks
from train import eval_one_epoch
from utils import load_checkpoint
from config import args
from attack import pgd

import argparse
import torch
import numpy as np
import os

if args.net == 'resnet':
    from network.net import create_network
elif args.net == 'vgg':
    from network.net2 import create_network

device = torch.device('cuda:{}'.format(args.gpu))
torch.backends.cudnn.benchmark = True

net = create_network()
net.to(device)

ds_val = load_data.create_test_dataset(512)

if os.path.isfile(args.resume):
    load_checkpoint(args.resume, net)

if args.val_method == 'fgsm':
    print('use FGSM attack')
    val_attack = torchattacks.FGSM(net)
elif args.val_method == 'cw':
    print('use cw attack')
    val_attack = torchattacks.CW(net)
else:
    print('use pgd attack')
    val_attack = pgd.IPGD(net, eps=8 / 255.0, sigma=2 / 255.0, nb_iter=20, norm=np.inf, device=device, drop_prob=0)

print('Evaluating')
clean_acc, adv_acc = eval_one_epoch(net, ds_val, device, val_attack)
print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))
