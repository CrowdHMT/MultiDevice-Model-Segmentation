# -*- coding: utf-8 -*-
# @FunctionName: sample_cout
# @Author: wanghongli
# @Time: 2022/2/28 19:20

import torch
import time
import xlwt
from ConvLayer import *
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')


def sample_cout_random():
    worksheet.write(2, 1, "out_channels")  # 将时间写入列表
    worksheet.write(3, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    arr = np.random.randint(1, 60, 7)  #随机采样 取7次采样
    arr.sort()
    # print(arr)
    for cout in arr:
        x = torch.randn(1, 3, 224, 224)
        net = ConvLayer(3, cout, 3, 1)
        starttime = time.time()
        for i in range(10):
            out = net(x)
        endtime = time.time()
        execution_t = endtime - starttime
        worksheet.write(2, count, cout.item())  # 将cin写入列表
        worksheet.write(3, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    workbook.save('conv_sample_cout.xls')  # 保存文件
    print("随机采样完成")

def sample_cout_all():
    worksheet.write(4, 1, "out_channels")  # 将时间写入列表
    worksheet.write(5, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    for cout in range(1, 60):
        x = torch.randn(1, 3, 224, 224)
        net = ConvLayer(3, cout, 3, 1)
        starttime = time.time()
        for i in range(10):
            out = net(x)
        endtime = time.time()
        execution_t = endtime - starttime
        worksheet.write(4, count, cout)  # 将cin写入列表
        worksheet.write(5, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    workbook.save('conv_sample_cout.xls')  # 保存文件
    print("全采样完成")


if __name__ == '__main__':
  sample_cout_random()
  sample_cout_all()
  # workbook.save('conv_sample_cin.xls')  # 保存文件