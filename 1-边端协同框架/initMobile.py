import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
from data import get_data_set
from nodegraph import readnodedata
import socket
import threading
import pickle
import io
import sys
import time
from models import *
from models.alexnet import AlexNet
from models.vgg import VGG
from models.googlenet_row import GoogLeNet
from models.resnet import ResNet18
from models.mobilenet import MobileNet
from models.shufflenet import ShuffleNetG2
import torchvision
from main import validate
from memory_profiler import profile

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="models/alexnet_retrained_200epoch.pkl"
VGG16_MODEL_PATH = "models/vgg/vgg16_retrained_300epoch_ir16.pkl"
ResNet18_MODEL_PATH = "models/resnet18_retrained_300epoch-85.pkl"
GoogLeNet_MODEL_PATH = "models/googlenet_retrained_200epoch.pkl"
MobileNet_MODEL_PATH = "models/mobilenet/mobilenet_retrained_300epoch_compressrate_0.75.pkl"
ShuffleNet_MODEL_PATH = "models/shufflenet_retrained_300epoch_comlressrate_0.75.pkl"

IP = "192.168.1.106"
PORT = 8081

class Data(object):

    def __init__(self, inputData, startLayer, endLayer):
        self.inputData = inputData
        self.startLayer = startLayer
        self.endLayer=endLayer

def run(model, inputData, startLayer, endLayer):
    print("移动端运行%d到%d层" % (startLayer, endLayer))
    if sys.argv[1] == "GoogLeNet":
        outputs = model(inputData, startLayer, endLayer, False)
    else:
        outputs = model(inputData, startLayer, endLayer, False)
    return outputs

def test(outputs):
    # model.eval()
    prediction = torch.max(outputs.data, 1)
    correct_classified = np.sum(prediction[1] == test_y)
    acc = (correct_classified / len(test_x)) * 100
    return acc

def sendData(client, inputData, startLayer, endLayer):
    data = Data(inputData, startLayer, endLayer)
    str=pickle.dumps(data)
    print("data size to trasmit" , sys.getsizeof(str))
    client.send(len(str).to_bytes(length=6, byteorder='big'))
    client.send(str)


def receiveData(client, model, x, test_x, test_y):
    while True:
        lengthData=client.recv(6)
        length=int.from_bytes(lengthData, byteorder='big')
        if length==0:
            continue
        b=bytes()
        count=0
        while True:
            value=client.recv(length)
            b=b+value
            count+=len(value)
            if count>=length:
                break
        data=pickle.loads(b)
        if data.startLayer>=len(x):
            end=time.time()
            acc=test(data.inputData)
            runtime=end-start
            print("计算任务运行完成，响应时间为：%f，准确率为：%f" % (runtime, acc))
            print(time.time())
            #client.close()
            break
        else:
            count=0
            for i in range(data.startLayer, len(x)):
                if x[i]==1:
                    break
                count=i
            outputs=run(model, data.inputData, data.startLayer, count)
            if count==len(x)-1:
                end=time.time()
                acc=test(outputs)
                runtime=end-start
                print("计算任务运行完成，响应时间为：%f，准确率为：%f" % (runtime, acc))
                print(time.time())
                #client.close()
                break
            else:
                endLayer=0
                for i in range(count+1, len(x)):
                    if x[i]==0:
                        break
                    endLayer=i
                sendData(client, outputs, count+1, endLayer)


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
    
    device = torch.device("cpu")
    torch.set_num_threads(3)
    # test_x, test_y, test_l = get_data_set("test")
    # test_x=torch.from_numpy(test_x[4000:5000]).float()
    # test_y=torch.from_numpy(test_y[4000:5000]).long()
    '''transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)'''
    test_x = torch.randn(10, 3, 32, 32)
    test_y = [1]

    #print('test_x')
    # print(len(test_x))
    print("模型加载成功")
    # outputs = run(model, test_x, 0, 12)
    # acc = test(outputs)
    # print("模型分割前准确率为：%f" % acc)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sendData(client, outputs, -1, 0)
    client.connect((IP, PORT))
    print("start to construct graph!!")
    print(time.time())

    '''# 模拟构建图  CAS
    from nodegraph import readnodedata
    num = 26
    node = readnodedata(num, "alexnet.xlsx")
    # 模拟寻找最佳卸载方案
    print(len(node))'''


    print("任务已提交，进行卸载决策!!!!")
    print(time.time())
    # validate(testloader, criterion, model)
    # x = [1,1,1,1,1,1,1,1,1,1,1,1,1]  # alexnet13  googlenet 16  mobilenet 16  VGG45  resnet7 shufflenet 6
    x = [0 for i in range(13)]
    # 决定哪几层在云端执行
    for i in range(10, 13):
        x[i] = 1
    #x = [0, 0, 0, 0, 0, 0, 0] #Resnet18
    #x=[1,1,1,1,1,1,1,1]
    #x=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    #x = [0, 0, 0, 0, 0, 0, 0] #132.49ms
    #x = [0 for i in range(45)] # vgg16

    print("start to compute!!")
    print(time.time())
    model.eval()
    '''list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]'''
    list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
    item = 0
    while item < 8:
        '''# 模拟构建图  DADS
        num = 15
        node = readnodedata(num, "alexnet.xlsx")
        # 模拟寻找最佳卸载方案
        while item in node:
            m = item.m'''
        # CAS
        num = 5
        node = readnodedata(num, "alexnet.xlsx")
        # 模拟寻找最佳卸载方案

        '''# 模拟nerosurgeon
        from nodegraph import readnodedata
        num = 26
        node = readnodedata(num, "alexnet.xlsx")
        tm = 0
        # 模拟寻找最佳卸载方案
        for no in node:
            t = no.tee + no.tran * (1 / 2)  # 带宽是缩小的
            if tm > t:
                tm = t'''

        x = list[item]
        start = time.time()
        if x[0] == 1:
            count = 0
            for i in range(1, len(x)):
                if x[i] == 0:
                    break
                count = count + 1
            sendData(client, test_x, 0, count)
            t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
            t.start()
            t.join()
        else:
            count = 0
            # 在移动端执行哪几层
            for i in range(1, len(x)):
                if x[i] == 1:
                    break
                count = i
            outputs = run(model, test_x, 0, count)
            if count == len(x) - 1:
                end = time.time()
                acc = test(outputs)
                runtime = end - start
                print("计算任务运行完成，响应时间为：%.6f，准确率为：%f" % (runtime, acc))
                print(time.time())
                #client.close()
            else:
                endLayer = 0
                for i in range(count + 1, len(x)):
                    if x[i] == 0:
                        break
                    endLayer = i
                sendData(client, outputs, count + 1, endLayer)
                t = threading.Thread(target=receiveData, name='receiveData',
                                     args=(client, model, x, test_x, test_y))
                t.start()
                t.join()
        item = item + 1
    client.close()








