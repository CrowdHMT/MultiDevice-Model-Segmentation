import torch
import torch.nn as nn
import torch.utils.data as Data
import time
from collections import OrderedDict
from models.alexnet import AlexNet
import torchvision
import torchvision.transforms as transforms
from java import jclass
import os
from os.path import dirname, join

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

# ALEXNET_MODEL_PATH = "src/models/alexnet_retrained_200epoch.pkl"
# ALEXNET_MODEL_PATH = "./models/alexnet_retrained_200epoch.pkl"
# path="alexnet_retrained_200epoch.pkl"

def run(model, inputData, startLayer, endLayer):
    print("云端运行%d到%d层" % (startLayer, endLayer))
    outputs = model(inputData, startLayer, endLayer, False)
    return outputs


# 测试模型精度
def test(model):
    criterion = nn.CrossEntropyLoss()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    rootPATH = join(dirname(__file__), "./datasets")
    testset = torchvision.datasets.CIFAR10(root=rootPATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("测试样例加载成功~")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            device = torch.device("cpu")
            inputs, targets = inputs.to(device), targets.to(device)
            # print("targets", targets)
            outputs = model(inputs, 0, 0, True)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            # print("predicted", predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print("模型分割前的精度为： ", 100. * correct/total, "%")
    return correct/total

# 测时间
def device_test(model, startlayer, endlayer):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    rootPATH = join(dirname(__file__), "./datasets")
    testset = torchvision.datasets.CIFAR10(root= rootPATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=0)
    model.eval()
    print("测试样例加载成功~")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            device = torch.device("cpu")
            inputs, targets = inputs.to(device), targets.to(device)
            print(inputs.shape)
            # shufflent 5  mobilenet 15 AlexNet13 GoogleNet  13 ResNet 7  VGG 44
            if(startlayer != 0):
                inputs= run(model, inputs, 0, startlayer-1)
            print("startlayer = ", startlayer)
            print("endlayer = ", endlayer)
            start = time.time()
            out = run(model, inputs, startlayer, endlayer)
            end = time.time()
            runtime = end - start
            print("PC端运行时间：%.6f" % runtime)  # 10张图片
            break
    return runtime
# startlayer 必须大于等于0小于等于11
def alexnet(startlayer, endlayer):
    state_dict = OrderedDict()
    model = AlexNet()
    x = torch.randn(10, 3, 32, 32)
    start = time.time()
    run(model, x, startlayer, endlayer)
    end = time.time()
    runtime = end - start
    # ALEXNET_MODEL_PATH = join(dirname(__file__), "Resnet38.pth.tar")
   # ALEXNET_MODEL_PATH = "alexnet_retrained_200epoch.pkl"
    #state_dict = torch.load(ALEXNET_MODEL_PATH, map_location=torch.device("cpu"))
   # new_state_dict = OrderedDict()
   # for k, v in state_dict.items():
   #     if k[0] == 'm' and k[1] == 'o':
    #        name = k[7:]  # remove `module.`
   #     else:
     #       name = k  # no change
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    #torch.set_num_threads(3)
    #print("模型加载成功")
    # 测量PC端的运行时间
    #runtime = device_test(model, startlayer, endlayer)
    # 测试模型精度
    # acc = test(model)
    print("模型的运行时延为：", runtime)   #10张图片
    return runtime

