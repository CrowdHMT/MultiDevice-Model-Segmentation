# -*- coding: utf-8 -*-
# @Author: wanghongli
# @Time: 2022/12/23 15:05
# @File: initCloud.py
# @Description: initCloud.py

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from data import get_data_set
import socket
import threading
import pickle
import io
import sys
import time
import os
import argparse
from models import *
import xlwt
from models.alexnet import AlexNet
from models.vgg import VGG
from models.googlenet import GoogLeNet
from models.resnet import ResNet18
from models.mobilenet import MobileNet
from models.shufflenet import ShuffleNetG2
import torchvision
import torchvision.transforms as transforms
_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

# ALEXNET_MODEL_PATH = "models/alexnetlayermodel_2epoch.pkl"
ALEXNET_MODEL_PATH = "models/alexnet_retrained_200epoch.pkl"
VGG16_MODEL_PATH = "models/vgg/vgg16_pretrained_300epoch_fei.pkl"
ResNet18_MODEL_PATH = "models/resnet18_pretrained_100epoch.pkl"
GoogLeNet_MODEL_PATH = "models/googlenet_pretrained_300epoch.pkl"
MobileNet_MODEL_PATH = "models/mobilenet_pretrained_400epoch.pkl"
ShuffleNet_MODEL_PATH = "models/shufflenet_pretrained_400epoch.pkl"
#ALEXNET_MODEL_PATH="models/alexnet_retrained_200epoch.pkl"
#VGG16_MODEL_PATH = "models/vgg/vgg16_retrained_300epoch_ir16.pkl"
#ResNet18_MODEL_PATH = "models/resnet18_retrained_300epoch-85.pkl"
#GoogLeNet_MODEL_PATH = "models/googlenet_retrained_200epoch.pkl"
#MobileNet_MODEL_PATH = "models/mobilenet/mobilenet_retrained_300epoch_compressreate_0.75.pkl"
#ShuffleNet_MODEL_PATH = "models/shufflenet_retrained_300epoch_comlressrate_0.75.pkl"
IP = "192.168.43.57"
PORT = 8081


class Data(object):
    def __init__(self, inputData, startLayer, endLayer):
        self.inputData = inputData
        self.startLayer =startLayer
        self.endLayer = endLayer

'''
update_time:
function:
input:
output:
user:
'''
def test(outputs, test_x, test_y):
    correct_classified = 0
    total = 0
    print("outputs", outputs)
    prediction = torch.max(outputs.data, 1)
    print("prediction", prediction)
    print("test_y", test_y)
    correct_classified += np.sum(prediction[1].numpy() == test_y.numpy())
    acc = (correct_classified/len(test_x))*100

    correct = 0
    _, predicted = outputs.max(1)
    print("predicted", predicted)
    total += test_y.size(0)
    correct += predicted.eq(test_y).sum().item()
    print(' Acc: %.3f%% (%d/%d)', 100. * correct / total, correct, total)
    return acc


def get_parameter_number(net):
    total_nonzeros = 0
    total_zeros = 0
    for name,W in model.named_parameters():
        W = W.cpu().detach().numpy()
        total_zeros += np.sum(W == 0)
        total_nonzeros += np.sum(W != 0)
    print("模型0参数量:", total_zeros, "\n")
    print("模型非0参数量:", total_nonzeros)

# 运行 开始层到结束层
def run(model, inputData, startLayer, endLayer):
    # print("云端运行alexnet-pretrained模型%d到%d层" % (startLayer, endLayer))
    # print("云端运行"+sys.argv[1].lower()+"-pretrained模型%d到%d层" % (startLayer, endLayer))
    print("云端运行" + sys.argv[1].lower() + "-retrained模型")
    outputs = model(inputData, startLayer, endLayer, False)
    return outputs


# 向移动端传送数据
def sendData(server, inputData, startLayer, endLayer):
    data = Data(inputData, startLayer, endLayer)
    str = pickle.dumps(data)
    # print(sys.getsizeof(str))
    server.send(len(str).to_bytes(length=6, byteorder='big'))
    server.send(str)


# 接收移动端数据
def receiveData(server, model):
    while True:
        conn, addr = server.accept()
        while True:
            lengthData = conn.recv(6)
            length = int.from_bytes(lengthData, byteorder='big')
            b = bytes()
            if length == 0:
                continue
            count = 0
            while True:
                value = conn.recv(length)
                b = b+value
                count += len(value)
                if count >= length:
                    break
            data = pickle.loads(b)
            outputs = run(model, data.inputData, data.startLayer, data.endLayer)
            sendData(conn, outputs, data.endLayer+1, 1)

def test(model):
    transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
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

def device_test(mode):
    '''
    :param mode:
    :return:
    '''
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.shape)
            # start = time.time()
            # shufflent 5  mobilenet 15 AlexNet 12 GoogleNet  12 ResNet 6  VGG 44
            if sys.argv[1] == "AlexNet":
                layers = 12
            elif sys.argv[1] == "VGG16":
                layers = 44
            elif sys.argv[1] == "ResNet18":
                layers = 6
            elif sys.argv[1] == "GoogLeNet":
                layers = 12
            elif sys.argv[1] == "MobileNet":
                layers = 15
            elif sys.argv[1] == "ShuffleNet":
                layers = 5
            for layer in range(0, layers+1):
                start = time.time()
                for i in range(10):
                    out = run(model, inputs, layer, layer)
                end = time.time()
                runtime = (end - start)/10
                print("PC端第 %d 层的运行时间：%.6f s" %(layer, runtime))
            # end = time.time()
            # runtime = end - start
            # print("PC端运行时间：%.6f s" % runtime)
            break
# sys.argv[1]
if __name__=="__main__":
    from collections import OrderedDict
    state_dict = OrderedDict()
    if sys.argv[1] == "AlexNet":
        model = AlexNet()
        state_dict = torch.load(ALEXNET_MODEL_PATH, map_location=torch.device("cpu"))
    elif sys.argv[1] == "VGG16":
        model = VGG('vgg16', w=16)
        state_dict = torch.load(VGG16_MODEL_PATH, map_location=torch.device("cpu"))
    elif sys.argv[1] == "ResNet18":
        model = ResNet18()
        state_dict = torch.load(ResNet18_MODEL_PATH, map_location=torch.device("cpu"))
    elif sys.argv[1] == "GoogLeNet":
        model = GoogLeNet()
        state_dict = torch.load(GoogLeNet_MODEL_PATH, map_location=torch.device("cpu"))
    elif sys.argv[1] == "MobileNet":
        model = MobileNet()
        state_dict = torch.load(MobileNet_MODEL_PATH, map_location=torch.device("cpu"))
    elif sys.argv[1] == "ShuffleNet":
        model = ShuffleNetG2()
        state_dict = torch.load(ShuffleNet_MODEL_PATH, map_location=torch.device("cpu"))
    criterion = nn.CrossEntropyLoss()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0] == 'm' and k[1] == 'o':
            name = k[7:]  # remove `module.`
        else:
            name = k  # no change
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model = models.alexnet(pretrained=True)
    #if isinstance(model, torch.nn.DataParallel):
    #	model = model.module
    device = torch.device("cpu")
    torch.set_num_threads(3)
    test_x, test_y, test_l = get_data_set("test")
    # test_x = torch.from_numpy(test_x[0:2000]).float()
    # test_y = torch.from_numpy(test_y[0:2000]).long()
    print("模型加载成功")
    # 得到模型的参数量
    # get_parameter_number(model)
    # 测量PC端的运行时间
    #device_test(model)
    # 测试模型精度
    # test(model)
    # sever 端开启
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(1)
    server.bind((IP, PORT))
    print("云端启动，准备接受任务")

    server.listen(1)
    receiveData(server, model)





