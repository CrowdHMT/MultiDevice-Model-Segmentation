from collections import OrderedDict

import torch
import torch.nn as nn


import time
import xlwt
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')

# -------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
# -------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------#
#   卷积 + 上采样
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   最后获得yolov4的输出
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m



# ---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
# ---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat

# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        # self.backbone = darknet53_tiny(None)
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        # self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 3 // 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
            nn.Conv2d(32, 64, 3, 2, 3 // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # 104,104,64 -> 52,52,128
            Resblock_body(64, 64),
            # 52,52,128 -> 26,26,256
            Resblock_body(128, 128),
            # 26,26,256 -> 13,13,512
            Resblock_body(256, 256),
            # 13,13,512 -> 13,13,512
            # BasicConv(512, 512, kernel_size=3),
            nn.Conv2d(512, 512, 3, 1, 3 // 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # BasicConv(512, 256, 1),
            nn.Conv2d(512, 256, 1, 1, 1 // 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            yolo_head([512, num_anchors * (5 + num_classes)], 256),
            Upsample(256, 128),
            yolo_head([256, num_anchors * (5 + num_classes)], 384),
        )

    def forward(self, x, startLayer, endLayer,  isTrain):
        if isTrain:
            x = self.features[0](x)
            x = self.features[1](x)
            x = self.features[2](x)

            # x = self.conv2(x)
            x = self.features[3](x)
            x = self.features[4](x)
            x = self.features[5](x)

            # 104,104,64 -> 52,52,128
            x, _ = self.features[6](x)
            # 52,52,128 -> 26,26,256
            x, _ = self.features[7](x)
            # 26,26,256 -> x为13,13,512
            #           -> feat1为26,26,256
            x, _ = self.features[8](x)
            # 13,13,512 -> 13,13,512
            x = self.features[9](x)
            x = self.features[10](x)
            x = self.features[11](x)

            # 13,13,512 -> 13,13,256
            x = self.features[12](x)
            x = self.features[13](x)
            x = self.features[14](x)

            # 13,13,256 -> 13,13,512 -> 13,13,255
            x = self.features[15](x)

            # 13,13,256 -> 13,13,128 -> 26,26,128
            P5 = torch.randn(1, 256, 7, 7)
            x = self.features[16](P5)

            feat1 = torch.randn(1, 256, 14, 14)
            # 26,26,256 + 26,26,128 -> 26,26,384
            x = torch.cat([x, feat1], axis=1)

            # 26,26,384 -> 26,26,256 -> 26,26,255
            x = self.features[17](x)
        else:
            if startLayer == endLayer:
                if startLayer >= 0 and startLayer <= 5:
                    x = self.features[startLayer](x)
                if startLayer >= 6 and startLayer <= 8:
                    x, _ = self.features[startLayer](x)
                if startLayer >= 9 and startLayer <=15:
                    x = self.features[startLayer](x)
                if startLayer == 16:
                    P5 = torch.randn(1, 256, 7, 7)
                    x = self.features[16](P5)

                if startLayer == 17:
                    feat1 = torch.randn(1, 256, 14, 14)
                    # 26,26,256 + 26,26,128 -> 26,26,384
                    x = torch.cat([x, feat1], axis=1)
                    x = self.features[17](x)
        return x


def test():
    net = YoloBody(10, 1000)
    worksheet.write(2, 1, "第几层")  # 将时间写入列表
    worksheet.write(3, 1, "边缘端执行时间")  # 将时间写入列表
    for j in range(0, 10):
        x = torch.randn(1, 3, 224, 224)
        for layer in range(0, 18):
            if j == 0:
                worksheet.write(2, layer + 2, layer)  # 将层数写入列表
            start = time.time()
            for i in range(0, 10):
                y = net(x, layer, layer, isTrain=False)
            end = time.time()
            worksheet.write(j + 3, layer + 2, (end - start) / 10)  # 将时间写入列表
            x = y
            print(y.size())
    workbook.save('tinyyolo_imagenet_inferencelatency.xls')  # 保存文件

def testoutputdata():
    net = YoloBody(10, 1000)
    x = torch.randn(1, 3, 224, 224)
    for layer in range(0, 18):
        y = net(x, layer, layer, isTrain=False)
        #print(y.size())
        outer = y.detach().numpy()
        # 数据大小！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        print("%6d" % (outer.size * outer.itemsize))  # 一张图片的输出数据
        x = y

if __name__ == '__main__':
    test()
    #testoutputdata()