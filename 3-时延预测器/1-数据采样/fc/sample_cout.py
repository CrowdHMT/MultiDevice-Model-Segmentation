# -*- coding: utf-8 -*-
# @FunctionName: sample_cout
# @Author: wanghongli
# @Time: 2022/2/28 20:30
import torch
import time
import xlwt
from FCLayer import *
import numpy as np

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')


def sample_cout_random():
    worksheet.write(2, 1, "out_channels")  # 将时间写入列表
    worksheet.write(3, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    arr = np.random.randint(1, 4097, 50)  #随机采样 取100次采样
    arr.sort()
    # print(arr)
    for cout in arr:
        x = torch.randn(1, 1024)
        net = FCLayer(1024, cout)
        starttime = time.time()
        for i in range(100):
            out = net(x)
        endtime = time.time()
        execution_t = (endtime - starttime)/10
        print(str(count - 1) + "sample")
        worksheet.write(2, count, cout.item())  # 将cin写入列表
        worksheet.write(3, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    print("随机采样完成")
    workbook.save('fc_sample_cout.xls')  # 保存文件

def sample_cout_all():
    worksheet.write(4, 1, "out_channels")  # 将时间写入列表
    worksheet.write(5, 1, "10次执行时间/s")  # 将时间写入列表
    count = 2
    for cout in range(2000, 2060):
        x = torch.randn(1, 1024)
        net = FCLayer(1024, cout)
        starttime = time.time()
        for i in range(100):
            out = net(x)
        endtime = time.time()
        execution_t = (endtime - starttime)/10
        print(str(count - 1) + "sample")
        worksheet.write(4, count, cout)  # 将cin写入列表
        worksheet.write(5, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    print("全采样完成")
    workbook.save('fc_sample_cout.xls')  # 保存文件


if __name__ == '__main__':
  sample_cout_random()
  sample_cout_all()
