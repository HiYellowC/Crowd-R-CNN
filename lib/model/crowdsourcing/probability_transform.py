# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 下午10:42
# @File    : probability_transform.py

import torch
from model.utils.config import cfg

# confidence[i, j, k]
# 第i个标注者 把 j 类 标注为 k 类的概率
# 0 为 背景类

# update alpha_con in rpn
# 针对与 [i, 0, j]
# [3, 21, 21] [] [1, 20, 3]
def update_alpha_con_rpn(alpha_con, bg_index, crowdsourced_classes):
    for j in range(crowdsourced_classes.size(2)):
        # TODO
        if bg_index.size() != torch.Size([0]):
            l = crowdsourced_classes[bg_index.t()[0], bg_index.t()[1], j]
            # update 0->0
            alpha_con[j][0][0] += torch.nonzero(l == 0).size(0)
            # update 0->c
            for c in range(1, 21):
                alpha_con[j][0][c] += torch.nonzero(l == c).size(0)


# alpha_con -> alpha
def get_alpha(alpha_con):
    sum = alpha_con.sum(dim=-1).view(alpha_con.size(0), alpha_con.size(1), 1).expand_as(alpha_con).float()
    return alpha_con.float() / sum

# 把正类标为正类的概率
def get_sensitivity(alpha):
    return 1 - torch.mean(alpha[:, 1:, 0], dim=1)
    # return torch.mean(alpha[:, list(range(1, alpha.size(1))), list(range(1, alpha.size(1)))], dim=1)


# 把负类标为负类的概率
def get_specificity(alpha):
    return alpha[:, 0, 0]

def get_a(sensitivity, crowdsourced_label):
    a = sensitivity.repeat(crowdsourced_label.size(0), crowdsourced_label.size(1), 1)
    index_n = torch.nonzero(crowdsourced_label == 0)
    if index_n.size() != torch.Size([0]):
        index_n = index_n.t()
        a[index_n[0], index_n[1], index_n[2]] = 1 - a[index_n[0], index_n[1], index_n[2]]
    a = torch.prod(a, dim=-1)
    return a

def get_b(specificity, crowdsourced_label):
    b = specificity.repeat(crowdsourced_label.size(0), crowdsourced_label.size(1), 1)
    index_n = torch.nonzero(crowdsourced_label != 0)
    if index_n.size() != torch.Size([0]):
        index_n = index_n.t()
        b[index_n[0], index_n[1], index_n[2]] = 1 - b[index_n[0], index_n[1], index_n[2]]
    b = torch.prod(b, dim=-1)
    return b

def binary_get_mu(a, b, p):
    mu = a * p / (a * p + b * (1 - p))
    return mu

def get_d(alpha, crowdsourced_classes):
    class_num = alpha.size(1)
    result = torch.ones(class_num).type_as(alpha)
    for j in range(crowdsourced_classes.size(0)):
        result *= alpha[j, :, crowdsourced_classes[j]]
    return result

# crowdsourced_classes[i, gt_ix], score, alpha
def get_mu(crowdsourced_classes, score, alpha):

    d = get_d(alpha, crowdsourced_classes)
    dp = d * score
    return dp / dp.sum()