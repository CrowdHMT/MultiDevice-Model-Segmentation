# -*- coding: utf-8 -*-
# @FunctionName: sample_cin
# @Author: wanghongli
# @Time: 2022/2/28 17:11

import torch
import time
import xlwt
from ConvLayer import *
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')


def sample_cin_random():
    worksheet.write(2, 1, "in_channels")  # 将时间写入列表
    worksheet.write(3, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    arr = np.random.randint(1, 60, 7)  #随机采样 取7次采样
    arr.sort()
    # print(arr)
    for cin in arr:
        x = torch.randn(1, cin, 224, 224)
        net = ConvLayer(cin, 64, 3, 1)
        starttime = time.time()
        for i in range(10):
            out = net(x)
        endtime = time.time()
        execution_t = endtime - starttime
        worksheet.write(2, count, cin.item())  # 将cin写入列表
        worksheet.write(3, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    workbook.save('conv_sample_cin.xls')  # 保存文件
    print("随机采样完成")

def sample_cin_all():
    worksheet.write(4, 1, "in_channels")  # 将时间写入列表
    worksheet.write(5, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    for cin in range(1, 60):
        x = torch.randn(1, cin, 224, 224)
        net = ConvLayer(cin, 64, 3, 1)
        starttime = time.time()
        for i in range(10):
            out = net(x)
        endtime = time.time()
        execution_t = endtime - starttime
        worksheet.write(4, count, cin)  # 将cin写入列表
        worksheet.write(5, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    workbook.save('conv_sample_cin.xls')  # 保存文件
    print("全采样完成")


if __name__ == '__main__':
  sample_cin_random()
  sample_cin_all()
  # workbook.save('conv_sample_cin.xls')  # 保存文件