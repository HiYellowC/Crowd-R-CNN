# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 上午10:16
# @File    : rpn_aggregation_layer.py


from __future__ import absolute_import

import torch
import torch.nn as nn
from .probability_transform import *
import numpy as np
from model.rpn.bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch
from model.rpn.generate_anchors import generate_anchors
import numpy.random as npr

from model.utils.config import cfg

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _RPNAggregationLayer(nn.Module):

    def __init__(self, feat_stride, scales, ratios, classes, n_classes):
        super(_RPNAggregationLayer, self).__init__()
        self._feat_stride = feat_stride
        self.classes = classes
        self.n_classes = n_classes
        anchor_scales = scales
        self._anchors = torch.from_numpy(
            generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)


    def forward(self, input):
        # torch.Size([1, 18, 50, 37])
        # input (rpn_cls_prob.data, gt_boxes, num_boxes, crowdsourced_classes, alpha_con)
        rpn_cls_prob = input[0]
        gt_boxes = input[1]
        num_boxes = input[2]
        im_info = input[3]
        crowdsourced_classes = input[4]
        alpha_con = input[5]
        batch_size = gt_boxes.size(0)

        # 把每个anchor的坐标列出来
        feat_height, feat_width = rpn_cls_prob.size(2), rpn_cls_prob.size(3)
        # _feat_stride 16 图片到feature map的比例
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_prob).float()
        A = self._num_anchors
        K = shifts.size(0)
        self._anchors = self._anchors.type_as(gt_boxes)  # move to specific gpu.
        # all_anchors torch.Size([1850, 9, 4])
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)

        # 删除越界的anchors
        keep = ((all_anchors[:, 0] >= 0) &
                (all_anchors[:, 1] >= 0) &
                (all_anchors[:, 2] < long(im_info[0][1])) &
                (all_anchors[:, 3] < long(im_info[0][0])))

        # 保留的anchors 索引
        inds_inside = torch.nonzero(keep).view(-1)
        anchors = all_anchors[inds_inside, :]


        # 从rpn_cls_score中找到gt_box对应的标签
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        # arggt_max_overlaps size([1, 20])
        _, arggt_max_overlaps = torch.max(overlaps, 1)
        index = inds_inside[arggt_max_overlaps]

        reshape_rpn_cls_prob = rpn_cls_prob.view(batch_size, 2, -1)
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        # gt_boxes_cls <=> p
        gt_boxes_cls = torch.gather(reshape_rpn_cls_prob[:, 1], 1, index)

        alpha = get_alpha(alpha_con)

        sensitivity = get_sensitivity(alpha)
        specificity = get_specificity(alpha)
        if DEBUG:
            print('sensitivity: ', sensitivity)
            print('specificity: ', specificity)
        a = get_a(sensitivity, crowdsourced_classes)
        b = get_b(specificity, crowdsourced_classes)

        # print('a: ', a)
        # print('b: ', b)
        # print('p: ', gt_boxes_cls)

        # Size [1, 20]
        if DEBUG:
            print('a: ', a)
            print('b: ', b)
            print('gt_boxes_cls: ', gt_boxes_cls)
        mu = binary_get_mu(a, b, gt_boxes_cls)
        if DEBUG:
            print('mu: ', mu)

        bg_index = torch.nonzero(mu < 0.5)
        # 更新 alpha_con 针对与 [i, 0, j]
        update_alpha_con_rpn(alpha_con, bg_index, crowdsourced_classes)
        if bg_index.size() != torch.Size([0]):
            bg_index = bg_index.t()
            gt_boxes[bg_index[0], bg_index[1], :] = 0

        return gt_boxes