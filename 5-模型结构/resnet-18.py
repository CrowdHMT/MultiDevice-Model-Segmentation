'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import xlwt
import time

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, divided_by=1):
        super(ResNet, self).__init__()
        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//divided_by, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//divided_by, num_blocks[3], stride=2)
        self.linear = nn.Linear(25088, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    '''
    def forward(self, x):
        out = self.conv1(x)
        out=self.bn1(out)
        out = F.relu(out) # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    '''
    def forward(self, x, startLayer, endLayer, isTrain):
        def switch_if(start):
            if start == 0:
                return self.conv1(x)
            elif start == 1:
                out = self.bn1(x)
                return F.relu(out)
            elif start == 2:
                return self.layer1(x)
            elif start == 3:
                return self.layer2(x)
            elif start == 4:
                return self.layer3(x)
            elif start == 5:
                out = self.layer4(x)
                return F.avg_pool2d(out, 4)
        if isTrain:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        else:
            if startLayer == endLayer:
                if startLayer == 6:
                    x = x.view(x.size(0), -1)
                    x = self.linear(x)
                elif startLayer < 6:
                    x = switch_if(startLayer)
                else:
                    x = self.linear(x)
            else:
                for i in range(startLayer, endLayer+1):
                    if i < 6:
                        x = switch_if(i)
                        print(x.size())
                    elif i == 6:
                        x = x.view(x.size(0), -1)
                        x = self.linear(x)
                    else:
                        x = self.linear(x)
        return x




class ResNet_adv(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, w=1):
        super(ResNet_adv, self).__init__()
        self.in_planes = int(4*w)

        self.conv1 = nn.Conv2d(3, int(4*w), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(4*w))
        self.layer1 = self._make_layer(block, int(4*w), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(8*w), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(16*w), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(32*w), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(32*w)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_adv_wide(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, divided_by=1):
        super(ResNet_adv_wide, self).__init__()
        self.in_planes = 16//divided_by

        self.conv1 = nn.Conv2d(3, 16//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16//divided_by)
        self.layer1 = self._make_layer(block, 160//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 320//divided_by, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 640//divided_by, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512//divided_by, num_blocks[3], stride=2)
        self.linear = nn.Linear(640//divided_by*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet18_adv(w = 1):
    return ResNet_adv(BasicBlock,[2,2,2,2],w = w)

def ResNet18_adv_wide():
    return ResNet_adv_wide(BasicBlock, [4,3,3],divided_by=1)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNet18_1by16():
    return ResNet(BasicBlock, [2, 2, 2, 2], divided_by=16)

def test():
    net = ResNet18()
    worksheet.write(2, 1, "第几层")  # 将时间写入列表
    worksheet.write(3, 1, "边缘端执行时间")  # 将时间写入列表
    for j in range(0, 10):
        x = torch.randn(1, 3, 224, 224)
        for layer in range(0, 7):
            if j == 0:
                worksheet.write(2, layer + 2, layer)  # 将层数写入列表
            start = time.time()
            for i in range(0, 10):
                y = net(x, layer, layer, isTrain=False)
            end = time.time()
            worksheet.write(j + 3, layer + 2, (end - start) / 10)  # 将时间写入列表
            x = y
            print(y.size())
    workbook.save('resnet18_imagenet_inferencelatency.xls')  # 保存文件
# test()

def testoutputdata():
    net = ResNet18()
    x = torch.randn(1, 3, 224, 224)
    for layer in range(0, 7):
        y = net(x, layer, layer, isTrain=False)
        outer = y.detach().numpy()
        # 数据大小！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        print("%6d" % (outer.size * outer.itemsize))  # 一张图片的输出数据
        x = y


if __name__ == '__main__':
    test()
    #testoutputdata()

