# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 下午10:53
# @File    : rcnn_aggregation_layer.py

import torch
import torch.nn as nn
from .probability_transform import *
from model.rpn.bbox_transform import bbox_overlaps_batch
from model.utils.config import cfg
import torch.nn.functional as F
import numpy as np

class _RCNNAggregationLayer(nn.Module):

    def __init__(self):
        super(_RCNNAggregationLayer, self).__init__()

    # cls_score, rois, gt_boxes, crowdsourced_classes, self.alpha_con, rois_label
    def forward(self, input):
        # [1, 256, 5]
        rois = input[1]
        batch_size = rois.size(0)
        # [1, 256, 21]
        cls_prob = input[0].view(batch_size, -1, input[0].size(1))
        # [1, 20, 5]
        gt_boxes = input[2]
        # [1, 20, 3]
        crowdsourced_classes = input[3]
        num_annotator = crowdsourced_classes.size(2)
        # [3, 21, 21]
        alpha_con = input[4]
        rois_label = input[5]

        overlaps = bbox_overlaps_batch(rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        alpha = get_alpha(alpha_con)

        # TODO batch_size 不为1 时可能有bug
        # 前景rois索引
        fg_rois_ix = torch.nonzero(rois_label != 0).view(-1)
        for i in range(batch_size):
            # 前景bbox 索引
            fg_gt_boxes_ix = torch.nonzero(gt_boxes[i, :, 4] != 0).view(-1)
            for ix in fg_gt_boxes_ix:
                # 搜集所有的与该bbox IoU最大的前景roi
                rois_ix = torch.nonzero(gt_assignment[i, fg_rois_ix] == ix).view(-1)

                # 留下的rois没有与该gt box匹配的
                if rois_ix.size(0) == 0:
                    continue

                # 映射全部rois下的索引
                rois_ix = fg_rois_ix[rois_ix]
                # 推理这个fg_gt_boxes的类别
                # p暂时用rois的类别平均概率 （后可以试试用IoU分配权重）size[21]
                mean_cls_prob = torch.mean(cls_prob[i, rois_ix, :], dim=0)
                tmp = torch.ones(mean_cls_prob.size()).type_as(mean_cls_prob)

                for c in range(tmp.size(0)):
                    tmp[c] *= mean_cls_prob[c]
                    for j in range(num_annotator):
                        tmp[c] *= alpha[j, c, crowdsourced_classes[i, ix, j]]

                tmp = tmp / tmp.sum()

                # 推理出的类别 不能为0
                _, mu = torch.max(tmp[1:], dim=0)
                mu += 1

                # 更改 rois_label
                rois_label[rois_ix] = mu
                # 更新 alpha_con
                for j in range(num_annotator):
                    alpha_con[j, mu, crowdsourced_classes[i, ix, j]] += 1

        return rois_label