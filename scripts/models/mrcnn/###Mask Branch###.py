###Mask Branch###

# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------


import math
import numpy as np
import tensorflow as tf
import tf.nn as nn
from torch.autograd import Variable
#from upsnet.operators.modules.fpn_roi_align import FPNRoIAlign
#from upsnet.operators.modules.roialign import RoIAlign
#from upsnet.operators.functions.roialign import RoIAlignFunction
from upsnet.operators.modules.view import View
from upsnet.config.config import config

class MaskBranch(nn.Module):

    def __init__(self, num_classes, dim_in=256, dim_hidden=256, with_norm='none'):
        super(MaskBranch, self).__init__()
        self.roi_pooling = FPNRoIAlign(config.network.mask_size // 2, config.network.mask_size // 2,
                                        [1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])
        conv = nn.Conv2d

        assert with_norm in ['batch_norm', 'group_norm', 'none']

        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        if with_norm != 'none':
            self.mask_conv1 = nn.Sequential(*[conv(dim_in, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_conv2 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_conv3 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_conv4 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_deconv1 = nn.Sequential(*[nn.ConvTranspose2d(dim_hidden, dim_hidden, 2, 2, 0), nn.ReLU(inplace=True)])
        else:
            self.mask_conv1 = nn.Sequential(*[conv(dim_in, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_conv2 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_conv3 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_conv4 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_deconv1 = nn.Sequential(*[nn.ConvTranspose2d(dim_hidden, dim_hidden, 2, 2, 0), nn.ReLU(inplace=True)])

        self.mask_score = nn.Conv2d(dim_hidden, num_classes, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, feat, rois):
        pool_feat = self.roi_pooling(feat, rois)
        mask_conv1 = self.mask_conv1(pool_feat)
        mask_conv2 = self.mask_conv2(mask_conv1)
        mask_conv3 = self.mask_conv3(mask_conv2)
        mask_conv4 = self.mask_conv4(mask_conv3)
        mask_deconv1 = self.mask_deconv1(mask_conv4)
        mask_score = self.mask_score(mask_deconv1)
        return mask_score
