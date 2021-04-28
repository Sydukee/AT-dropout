import os
import sys
from utils import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm

father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
# print(father_dir)
if not father_dir in sys.path:
    sys.path.append(father_dir)


# 训练一个epoch
def train_one_epoch(net, batch_generator, optimizer,
                    criterion, device=torch.device('cuda:0'),
                    descrip_str='Training', attack_method=None, adv_coef=1.0, logger=None):
    # 设置为训练模式
    net.train()

    # 初始化进度条，以及acc、loss
    pbar = tqdm(batch_generator)
    adv_acc = -1
    adv_loss = -1
    clean_acc = -1
    clean_loss = -1
    pbar.set_description(descrip_str)

    # 分批次取数据进行训练
    for i, (data, label) in enumerate(pbar):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        pbar_dic = OrderedDict()

        # 对当前batch生成对抗样本
        if attack_method is not None:
            adv_inp = attack_method.forward(data, label)
            optimizer.zero_grad()
            net.train()

            # 前传
            pred = net(adv_inp)
            loss = criterion(pred, label)
            acc = torch_accuracy(pred, label, (1,))
            adv_acc = acc[0].item()
            adv_loss = loss.item()

            # 反传
            (loss * adv_coef).backward()

        pred = net(data)

        loss = criterion(pred, label)
        # TotalLoss = TotalLoss + loss
        loss.backward()
        # TotalLoss.backward()
        # param = next(net.parameters())
        # grad_mean = torch.mean(param.grad)

        optimizer.step()
        acc = torch_accuracy(pred, label, (1,))
        clean_acc = acc[0].item()
        clean_loss = loss.item()
        pbar_dic['Acc'] = '{:.2f}'.format(clean_acc)
        pbar_dic['loss'] = '{:.2f}'.format(clean_loss)
        pbar_dic['Adv Acc'] = '{:.2f}'.format(adv_acc)
        pbar_dic['Adv loss'] = '{:.2f}'.format(adv_loss)
        pbar.set_postfix(pbar_dic)

        if logger is None:
            pass
        else:
            logger.info(f'standard loss: {clean_loss:.3f}, Adv loss: {adv_loss:.3f}')
            logger.info(f'standard acc: {clean_acc:.3f}%, robustness acc: {adv_acc:.3f}%')


def eval_one_epoch(net, batch_generator, device=torch.device('cuda:0'), val_attack=None, logger=None):
    # logger.info('test start')
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        if val_attack is not None:
            adv_inp = val_attack.forward(data, label)

            with torch.no_grad():
                pred = net(adv_inp)
                acc = torch_accuracy(pred, label, (1,))
                adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if val_attack is not None else 0

        if logger is None:
            pass
        else:
            logger.info(f'standard acc: {clean_accuracy.mean:.3f}%, robustness acc: {adv_accuracy.mean:.3f}%')
    return clean_accuracy.mean, adv_acc
