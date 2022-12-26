# -*- coding: utf-8 -*-
# @FunctionName: sample_kernelsize
# @Author: wanghongli
# @Time: 2022/2/28 16:23

import torch
import time
import xlwt
from ConvLayer import *

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')


def sample_kernelsize():
    worksheet.write(2, 1, "kernel_size")  # 将时间写入列表
    worksheet.write(3, 1, "10次执行时间/s")  # 将时间写入列表
    x = torch.randn(1, 3, 224, 224)
    count = 2
    for kernelsize in range(1, 16, 2):
        net = ConvLayer(3, 64, kernelsize, 1)
        starttime = time.time()
        for i in range(10):
            out = net(x)
        endtime = time.time()
        execution_t = endtime - starttime
        worksheet.write(2, count, kernelsize)  # 将时间写入列表
        worksheet.write(3, count, execution_t)  # 将时间写入列表
        count = count + 1
        # print(execution_t)
    workbook.save('conv_sample_kernelsize.xls')  # 保存文件


if __name__ == '__main__':
  sample_kernelsize()