import os
import logging
import torch
import math
from typing import List
from config import args


def create_logger(save_path='', file_type='', level='debug'):
    if level == 'debug':
        _level = logging.DEBUG
    else:
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


def save_checkpoint(now_epoch, net, optimizer, lr_scheduler, file_name):
    checkpoint = {'epoch': now_epoch,
                  'state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_scheduler_state_dict': lr_scheduler.state_dict()}
    # if not os.path.exists(file_name):
    #     os.mkdir(file_name)
    torch.save(checkpoint, file_name)
    # link_name = os.path.join(args.model_dir, 'last.checkpoint')
    # print(link_name)
    # make_symlink(source=file_name, link_name=link_name)


def load_checkpoint(file_name, net=None, optimizer=None, lr_scheduler=None):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        check_point = torch.load(file_name)
        if net is not None:
            print('Loading network state dict')
            net.load_state_dict(check_point['state_dict'])
        if optimizer is not None:
            print('Loading optimizer state dict')
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        if lr_scheduler is not None:
            print('Loading lr_scheduler state dict')
            lr_scheduler.load_state_dict(check_point['lr_scheduler_state_dict'])

        return check_point['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(file_name))


# def make_symlink(source, link_name):
#     if os.path.exists(link_name):
#         # print("Link name already exist! Removing '{}' and overwriting".format(link_name))
#         os.remove(link_name)
#     if os.path.exists(source):
#         os.symlink(source, link_name)
#         return
#     else:
#         print('Source path not exists')
#     # print('SymLink Wrong!')


def torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


class AvgMeter(object):
    name = 'No name'
    sum = 0
    mean = 0
    num = 0
    now = 0

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num
