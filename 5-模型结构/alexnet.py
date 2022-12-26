import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import time
import xlwt
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')
# you need to download the models to ~/.torch/models
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = 'alexnet-owt-4df8aa71.pth'

NUM_CLASSES = 1000


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 4096),

            nn.Linear(4096, 4096),

            nn.Linear(4096, NUM_CLASSES),
        )

    # def forward(self, x):
    #   x = self.features(x)
    #   x = x.view(x.size(0), 2*2*128)
    #   x = self.classifier(x)
    #   return x

    def forward(self, x, startLayer, endLayer, isTrain):
        if isTrain:
            x = self.features(x)
            x = x.view(x.size(0), 6*6*256)
            x = self.classifier(x)
        else:
            if startLayer == endLayer:
                if startLayer == 10:
                    x = x.view(x.size(0), 6*6*256)
                    x = self.classifier[startLayer-10](x)
                elif startLayer < 10:
                    x = self.features[startLayer](x)
                else:
                    x = self.classifier[startLayer-10](x)
            else:
                for i in range(startLayer, endLayer+1):
                    if i < 10:
                        x = self.features[i](x)
                    elif i == 10:
                        x = x.view(x.size(0), 6*6*256)
                        x = self.classifier[i-10](x)
                    else:
                        x = self.classifier[i-10](x)
        return x


def alexnet(pretrained=False, **kwargs):
    """
    AlexNet model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model


def test():
    net = AlexNet()
    worksheet.write(2, 1, "第几层")  # 将时间写入列表
    worksheet.write(3, 1, "边缘端执行时间")  # 将时间写入列表
    for j in range(0, 10):
        x = torch.randn(1, 3, 224, 224)
        for layer in range(0, 13):
            if j == 0:
                worksheet.write(2, layer+2, layer)  # 将层数写入列表
            start = time.time()
            for i in range(0, 10):
                y = net(x, layer, layer, isTrain=False)
            end = time.time()
            worksheet.write(j+3, layer+2, (end - start)/10)  # 将时间写入列表
            x = y
            print(y.size())
    workbook.save('alexnet_imagenet_inferencelatency.xls')  # 保存文件


def testoutputdata():
    net = AlexNet()
    x = torch.randn(1, 3, 224, 224)
    for layer in range(0, 13):
        y = net(x, layer, layer, isTrain=False)
        outer = y.detach().numpy()
        # 数据大小！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        print("%6d" % (outer.size * outer.itemsize))  # 一张图片的输出数据
        x = y

if __name__ == '__main__':
   test()
   #testoutputdata()
