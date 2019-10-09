# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  # 1
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  # 根据scales的数量，为每张图片生成一个scales的索引
  # [0]
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)

  # 断言，要求batchsize必须是图片数量的整数倍
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  # 调用函数，读取图片数据，返回矩阵数据和每张图片的尺寸缩放信息
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  # 把blobs字典的key：data赋值为im_blob，也就是图片的矩阵数据
  blobs = {'data': im_blob}
  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    # [0, 1, 2,...]
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  # 一张图片里有 len(gt_inds) 个 bbox
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)
  # 添加属性 crowdsourced_classes
  if 'crowdsourced_classes' in roidb[0].keys():
    blobs['crowdsourced_classes'] = roidb[0]['crowdsourced_classes']
  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  # resize
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
