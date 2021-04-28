import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import heapq
from config import args


class sm_PGD():
    def __init__(self, net, eps=6 / 255.0, sigma=3 / 255.0, nb_iter=20, norm=np.inf, device=torch.device('cpu'),
                 mean=torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std=torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 random_start=True, drop_prob=0):
        self.eps = eps  # 单次扰动大小限制
        self.sigma = sigma  # 单次扰动学习步长
        self.nb_iter = nb_iter  # 攻击迭代次数
        self.norm = norm  # 攻击的范数限制
        self.criterion = torch.nn.CrossEntropyLoss().to(device)  # loss种类
        self.device = device  # 指定gpu
        self.random_start = random_start  # 最初是否使用随机扰动
        self.net = net
        self.drop_prob = drop_prob

        # 归一化参数
        self._mean = mean.to(device)
        self._std = std.to(device)

    # 单步攻击
    def single_attack(self, net, inp, label, eta):
        adv_inp = inp + eta
        # net.zero_grad()
        pred = net(adv_inp)
        total_loss = 0
        for i in range(pred.shape[0]):
            label_i = label[i].view(-1)
            pred_i = pred[i].view(-1, args.class_num)
            label_check = label[i].cpu().numpy()
            is_edge, top1, top2 = self.edge_check(pred[i], args.epsilon, label_check)  # edge check epsilon

            if is_edge:
                penal_label = torch.zeros(args.class_num)  # 应为类别数量

                penal_label[top1] = 0.5  # theta adjacent
                penal_label[top2] = 0.5

                penal_label_i = penal_label.view(-1, args.class_num)
                penal_loss = self.one_hot_ceLoss(1, pred_i, penal_label_i)

                loss_i = self.criterion(pred_i, label_i) - args.alpha * penal_loss

            else:
                loss_i = self.criterion(pred_i, label_i)
            total_loss += loss_i

        # 求loss
        loss = total_loss / args.batch_size
        # loss2 = self.criterion(pred, label)
        # print('loss is', loss)
        # print('loss2 is', loss2)
        # 求出梯度

        grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs=True, retain_graph=False)[0].sign()

        dropout = torch.nn.Dropout(p=self.drop_prob)
        grad_sign_d = dropout(grad_sign)

        adv_inp = adv_inp + grad_sign_d * (self.sigma / self._std)

        tmp_adv_inp = adv_inp * self._std + self._mean

        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)  ## clip into 0-1
        # tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = self.clip_eta(tmp_eta, norm=self.norm, eps=self.eps, device=self.device)

        eta = tmp_eta / self._std

        return eta

    def forward(self, inp, label):
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.device)
        eta = (eta - self._mean) / self._std
        self.net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attack(self.net, inp, label, eta)
            # print(i)
        # print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std

        return adv_inp

    def to(self, device):
        self.device = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

    def clip_eta(self, eta, norm, eps, device=torch.device('cuda:0')):
        assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

        with torch.no_grad():
            avoid_zero_div = torch.tensor(1e-12).to(device)
            eps = torch.tensor(eps).to(device)
            one = torch.tensor(1.0).to(device)

            if norm == np.inf:
                # l无穷 范数裁剪
                eta = torch.clamp(eta, -eps, eps)
            else:
                # l1、l2 范数裁剪
                normalize = torch.norm(eta.reshape(eta.size(0), -1), p=norm, dim=-1, keepdim=False)
                normalize = torch.max(normalize, avoid_zero_div)

                normalize.unsqueeze_(dim=-1)
                normalize.unsqueeze_(dim=-1)
                normalize.unsqueeze_(dim=-1)

                factor = torch.min(one, eps / normalize)
                eta = eta * factor
        return eta

    def edge_check(self, pred, epsilon, label_check):
        top = heapq.nlargest(2, range(len(pred)), pred.__getitem__)
        top1 = top[0]
        top2 = top[1]
        if (pred[top1] - pred[top2] < epsilon) and (top2 == label_check):
            return True, top1, top2
        else:
            return False, top1, top2

    def one_hot_ceLoss(self, n, inp, label):
        # print(inp.shape)
        log_prob = F.log_softmax(inp, dim=1)
        # print('log_softmax is', log_prob)
        # print('one hot label is', label)
        label = label.to(self.device)
        loss = -torch.sum(log_prob * label) / n
        return loss
