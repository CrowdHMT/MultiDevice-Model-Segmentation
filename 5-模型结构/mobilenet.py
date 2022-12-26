'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import xlwt
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    # ????
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    # 13个block
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(50176, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, startLayer, endLayer, isTrain):
        def switch_if(start, input):
            if start == 0:
                return self.conv1(input)
            elif start == 1:
                out = self.bn1(input)
                return F.relu(out)
            elif start >= 2 and start < 14:
                return self.layers[start-2](input)
            elif start == 14:
                out = self.layers[start-2](input)
                return F.avg_pool2d(out, 2)
        # 16层
        if isTrain:
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)  # out = F.relu(self.bn1(self.conv1(x)))
            out = self.layers(out)
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            if startLayer == endLayer:
                if startLayer == 15:
                    out = x.view(x.size(0), -1)
                    out = self.linear(out)
                elif startLayer < 15:
                    out = switch_if(startLayer, x)
                else:
                    out = self.linear(x)
            else:
                input = x
                for i in range(startLayer, endLayer+1):
                    if i < 15:
                        out = switch_if(i, input)
                    elif i == 15:
                        out = input.view(input.size(0), -1)
                        out = self.linear(out)
                    input = out

        return out


def test():
    # 16层
    net = MobileNet()
    worksheet.write(2, 1, "第几层")  # 将时间写入列表
    worksheet.write(3, 1, "边缘端执行时间")  # 将时间写入列表
    for j in range(0, 5):
        x = torch.randn(1, 3, 224, 224)
        for layer in range(0, 16):
            if j == 0:
                worksheet.write(2, layer + 2, layer)  # 将层数写入列表
            start = time.time()
            for i in range(0, 5):
                y = net(x, layer, layer, isTrain=False)
            end = time.time()
            worksheet.write(j + 3, layer + 2, (end - start) / 5)  # 将时间写入列表
            x = y
            print(y.size())
    workbook.save('mobilenet_imagenet_inferencelatency.xls')  # 保存文件


def testoutputdata():
    net = MobileNet()
    x = torch.randn(1, 3, 224, 224)
    for layer in range(0, 16):
        y = net(x, layer, layer, isTrain=False)
        outer = y.detach().numpy()
        # 数据大小！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        print("%6d" % (outer.size * outer.itemsize))  # 一张图片的输出数据
        x = y


if __name__ == '__main__':
    #test() # 27层
    testoutputdata()
