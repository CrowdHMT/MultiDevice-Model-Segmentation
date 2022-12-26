# -*- coding: utf-8 -*-
# @FunctionName: FCLayer
# @Author: wanghongli
# @Time: 2022/2/28 19:51

import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, cin, cout):
        super(FCLayer, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=cin, out_features=cout)
        )

    def forward(self, x):
        x = self.features(x)
        return x