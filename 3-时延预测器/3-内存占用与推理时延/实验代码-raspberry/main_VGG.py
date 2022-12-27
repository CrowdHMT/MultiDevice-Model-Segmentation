'''VGG11/13/16/19 in Pytorch.'''
import numpy as np
import torch
import torch.nn as nn
import time

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg16_1by8': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M'], #1/8
    'vgg16_1by16': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'], #1/16
    'vgg16_1by32': [2, 2, 'M', 4, 4, 'M', 8, 8, 8, 'M', 16, 16, 16, 'M', 16, 16, 16, 'M'] #1/32
}

cfg_now = {}
class VGG(nn.Module):
    def __init__(self, vgg_name, w=16):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], w)
        final_channels = None
        if vgg_name == 'vgg16':
            final_channels = int(512 * w / 16)
        elif vgg_name == 'vgg16_1by8':
            final_channels = 64
        elif vgg_name == 'vgg16_1by16':
            final_channels = 32
        elif vgg_name == 'vgg16_1by32':
            final_channels = 16
        self.classifier = nn.Linear(final_channels, 10)
        cfg_now = cfg[vgg_name]
    #def forward(self, x):
    #    out = self.features(x)
    #    out = out.view(out.size(0), -1)
    #    out = self.classifier(out)
    #   return out

    def forward(self, x, startLayer, endLayer, isTrain):
        ilayer = 44
        '''for x in cfg_now:
            if x == 'M':
                ilayer += 1
            else:
                ilayer += 3
        '''
        if isTrain:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            if startLayer == endLayer:
                if startLayer == ilayer:
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                elif startLayer < ilayer:
                    x = self.features[startLayer](x)
                else:
                    x = self.classifier[startLayer-ilayer](x)
            else:
                for i in range(startLayer, endLayer+1):
                    if i < ilayer:
                        x = self.features[i](x)
                    elif i == ilayer:
                        x = x.view(x.size(0), -1)
                        x = self.classifier(x)
                    else:
                        x = self.classifier(x)
        return x


    def _make_layers(self, cfg, w):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(w/16*x)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGG_adv(nn.Module):
    def __init__(self, vgg_name, w=1):
        super(VGG_adv, self).__init__()
        self.features = self._make_layers(cfg[vgg_name],w)

        final_channels = None
        self.base = 32
        #if vgg_name  == 'vgg16':
        final_channels = int(512*w/16)

        self.classifier = nn.Linear(final_channels, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg,w):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(w/16*x)
                print ('x is {}'.format(x))
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_ori_adv(nn.Module):
    def __init__(self, vgg_name, w=1):
        super(VGG_ori_adv, self).__init__()
        self.features = self._make_layers(cfg[vgg_name],w)

        final_channels = None


        final_channels = int(512*w/16)

        self.classifier = nn.Linear(final_channels, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg,w):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(w/16*x)
                print('x is {}'.format(x))
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),

                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = VGG('vgg16')
    x = torch.randn(2, 3, 32, 32)
    latency = [0 for _ in range(23)]
    start = time.time()
    y = net(x, 0, 0 , False)
    x = y
    end = time.time()
    runtime = end - start
    latency[0] = runtime
    count = 1
    for i in range(1, 44, 2):
        start = time.time()
        y = net(x, i, i+1, False)
        x = y
        end = time.time()
        runtime = end - start
        latency[count] = runtime
        count = count+1
    print(latency)

# test()
