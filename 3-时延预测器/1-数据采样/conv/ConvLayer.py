# -*- coding: utf-8 -*-
# @FunctionName: ConvLayer
# @Author: wanghongli
# @Time: 2022/2/28 16:23
import torch.nn as nn
import sys
import os

current_dir = os.getcwd()  # obtain work dir
sys.path.append(current_dir)  # add work dir to sys path
class ConvLayer(nn.Module):
    def __init__(self, cin, cout, kernelsize_t, stride_t):
        super(ConvLayer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=kernelsize_t, stride=stride_t)
        )

    def forward(self, x):
        x = self.features(x)
        return x