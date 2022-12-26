# -*- coding: utf-8 -*-
# @FunctionName: sample_cin
# @Author: wanghongli
# @Time: 2022/2/28 19:50

import torch
import time
import xlwt
from FCLayer import *
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')


def sample_cin_random():
    worksheet.write(2, 1, "in_channels")  # 将时间写入列表
    worksheet.write(3, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    arr = np.random.randint(1, 4000, 50)  #随机采样 取100次采样
    arr.sort()
    # print(arr)
    for cin in arr:
        x = torch.randn(1, cin)
        net = FCLayer(cin, 4096)
        starttime = time.time()
        for i in range(100):
            out = net(x)
        endtime = time.time()
        execution_t = (endtime - starttime)/10
        print(str(count - 1) + "sample")
        worksheet.write(2, count, cin.item())  # 将cin写入列表
        worksheet.write(3, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    print("随机采样完成")
    workbook.save('fc_sample_cin.xls')  # 保存文件

def sample_cin_all():
    worksheet.write(4, 1, "in_channels")  # 将时间写入列表
    worksheet.write(5, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    for cin in range(2400, 2450):
        x = torch.randn(1, cin)
        net = FCLayer(cin, 4096)
        starttime = time.time()
        for i in range(100):
            out = net(x)
        endtime = time.time()
        execution_t = (endtime - starttime)/10
        print(str(count - 1) + "sample")
        worksheet.write(4, count, cin)  # 将cin写入列表
        worksheet.write(5, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    print("全采样完成")
    workbook.save('fc_sample_cin.xls')  # 保存文件


if __name__ == '__main__':
  sample_cin_random()
  sample_cin_all()
  # workbook.save('conv_sample_cin.xls')  # 保存文件