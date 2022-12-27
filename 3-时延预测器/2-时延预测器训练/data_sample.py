# -*- coding: utf-8 -*-
# @FunctionName: data_sample
# @Author: wanghongli
# @Time: 2022/3/4 17:01
import os
import sys
import csv
import numpy as np
from fvcore.nn import FlopCountAnalysis

sys.path.append('../')

from conv.ConvLayer import ConvLayer
import torch
import time
from thop import profile

# 采样空间中的数据并转存为csv格式 以训练随机森林模型
# f = open('conv_example_computer_adaptivesample.csv', 'w', encoding='utf-8')
f = open('conv_example_flops.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
arr_hw = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
#arr_cin = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
arr_cout = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
arr_cin = [384]
#arr_cout = [384]
arr_k_s = [1, 3, 5, 7]
arr_stride = [1, 3, 5, 7]

def save_sample_data():
    # 模型类型/conv、fc；特征图长宽；输入大小cin；输出大小cout；卷积核大小kernel；步长stride
    headers = ['type','index', 'hw', 'cin', 'cout', 'kernel', 'stride', 'latency']
    csv_writer.writerow(headers)
    count = 0
    for hw in arr_hw:
        for cin in arr_cin:
            for cout in arr_cout:
                for k_s in arr_k_s:
                    for stride in arr_stride:
                        # print(hw, cin, cout, k_s, stride)
                        # kernelsize不能大于cin
                        if hw < k_s:
                            continue
                        else:
                            sample(hw, cin, cout, k_s, stride, count)
                            print("第"+str(count)+"次采样")
                            count = count + 1
    f.close()


def sample_flops_data():
    # 模型的flops latency  根据这些去拟合
    headers = ['flops', 'latency']
    csv_writer.writerow(headers)
    count = 0
    for hw in arr_hw:
        for cin in arr_cin:
            for cout in arr_cout:
                for k_s in arr_k_s:
                    for stride in arr_stride:
                        # print(hw, cin, cout, k_s, stride)
                        # kernelsize不能大于cin
                        if hw < k_s:
                            continue
                        else:
                            sample_flops(hw, cin, cout, k_s, stride, count)
                            print("第" + str(count) + "次采样")
                            count = count + 1
    f.close()

# 需要写成循环嵌套 形式
def sample_flops(hw, cin, cout, k_s, stride, index):
    x = torch.randn(1, cin, hw, hw)
    #模型类型
    net = ConvLayer(cin, cout, k_s, stride)
    starttime = time.time()
    for i in range(100):
        out = net(x)
    endtime = time.time()
    execution_t = (endtime - starttime)
    # print(execution_t)
    flops, params = profile(net, inputs=(x,))
    print("FlopsL", flops)
    rows = [flops, execution_t]
    csv_writer.writerow(rows)


# 需要写成循环嵌套 形式
def sample(hw, cin, cout, k_s, stride, index):
    x = torch.randn(1, cin, hw, hw)
    #模型类型
    net = ConvLayer(cin, cout, k_s, stride)
    starttime = time.time()
    for i in range(10):
        out = net(x)
    endtime = time.time()
    execution_t = (endtime - starttime)/10
    # print(execution_t)
    rows = ["conv", index, hw, cin, cout, k_s, stride, execution_t]
    csv_writer.writerow(rows)

if __name__ == "__main__":
    # save_sample_data()
    sample_flops_data()