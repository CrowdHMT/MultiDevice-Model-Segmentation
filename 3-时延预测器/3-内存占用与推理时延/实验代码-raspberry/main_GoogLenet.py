'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from memory_profiler import profile

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        # 12
        self.linear = nn.Linear(1024, 10)
    '''
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    '''

    def forward(self, x, startLayer, endLayer, isTrain):
        def switch_if(start):
            if start == 0:
                return self.pre_layers(x)
            elif start == 1:
                return self.a3(x)
            elif start == 2:
                return self.b3(x)
            elif start == 3:
                return self.maxpool(x)
            elif start == 4:
                return self.a4(x)
            elif start == 5:
                return self.b4(x)
            elif start == 6:
                return self.c4(x)
            elif start == 7:
                return self.d4(x)
            elif start == 8:
                return self.e4(x)
            elif start == 9:
                out = self.maxpool(x)
                return self.a5(out)
            elif start == 10:
                return self.b5(x)
            else:
                return self.avgpool(x)
        if isTrain:
            x = self.pre_layers(x)
            x = self.a3(x)
            x = self.b3(x)
            x = self.maxpool(x)
            x = self.a4(x)
            x = self.b4(x)
            x = self.c4(x)
            x = self.d4(x)
            x = self.e4(x)
            x = self.maxpool(x)
            x = self.a5(x)
            x = self.b5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        else:
            if startLayer == endLayer:
                if startLayer == 12:
                    x = x.view(x.size(0), -1)
                    x = self.linear(x)
                elif startLayer < 12:
                    x = switch_if(startLayer)
                else:
                    x = self.linear(x)
            else:
                for i in range(startLayer, endLayer+1):
                    if i < 12:
                        x = switch_if(i)
                    elif i == 12:
                        x = x.view(x.size(0), -1)
                        x = self.linear(x)
                    else:
                        x = self.linear(x)
        return x
#@profile
def test():
    net = GoogLeNet()
    x = torch.randn(1, 3, 32, 32)
    # y = net(x, 0, 15, False, 0, 0, 0)
    latency = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totalruntime = 0
    for i in range(0, 13):
        start = time.time()
        y = net(x, i, i, False)
        x = y
        end = time.time()
        runtime = end - start
        totalruntime = totalruntime + runtime
        latency[i] = runtime
    print(latency)
    print(totalruntime)
    # print(y)

# if __name__ == "__main__":
#    test()

if __name__ == '__main__':
    net = GoogLeNet()
    x = torch.randn(1, 3, 32, 32)
    start = time.time()
    for t in range(10):
        y = net(x, None, None, True)
    end = time.time()
    runtime = (end - start)/10
    print("runtime: ", runtime)
    print("test GoogLenet")
