
import torch
import torch.nn as nn
NUM_CLASSES = 10
import time


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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, NUM_CLASSES),
        )


    def forward(self, x, startLayer, endLayer, isTrain):
        if isTrain:
            x = self.features(x)
            x = x.view(x.size(0), 256)
            x = self.classifier(x)
        else:
            if startLayer == endLayer:
                if startLayer == 10:
                    x = x.view(x.size(0), 256)
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
                        x = x.view(x.size(0), 256)
                        x = self.classifier[i-10](x)
                    else:
                        x = self.classifier[i-10](x)
        return x

if __name__ == '__main__':
    net = AlexNet()
    x = torch.randn(128, 3, 32, 32)
    start = time.time()
    for t in range(10):
        y = net(x, None, None, True)
    end = time.time()
    runtime = (end - start)/10
    print("runtime: ", runtime)
    print("test AlexNet")

# if __name__ == '__main__':
#     net = AlexNet()
#     x = torch.randn(128, 3, 32, 32)
#     latency = [0 for _ in range(13)]
#     for i in range(0, 13):
#         start = time.time()
#         for t in range(10):
#             y = net(x, i, i, False)
#         end = time.time()
#         runtime = (end - start)/10
#         x = y
#         latency[i] = runtime
#     print(latency)


