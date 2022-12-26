# -*- coding: utf-8 -*-
# @FunctionName: sample_hw
# @Author: wanghongli
# @Time: 2022/2/28 16:16

import torch
import time
import xlwt
from ConvLayer import *

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')


def sample_hw():
    worksheet.write(2, 1, "hw")  # 将时间写入列表
    worksheet.write(3, 1, "10次执行时间/s")  # 将时间写入列表
    net = ConvLayer(3, 64, 3, 1)
    count = 2
    hw = 4
    while hw < 300:
        x = torch.randn(1, 3, hw, hw)
        starttime = time.time()
        for i in range(10):
            out = net(x)
        endtime = time.time()
        execution_t = endtime - starttime
        worksheet.write(2, count, hw)  # 将时间写入列表
        worksheet.write(3, count, execution_t)  # 将时间写入列表
        count = count + 1
        hw = hw * 2
        # print(execution_t)
    workbook.save('conv_sample_hw.xls')  # 保存文件


if __name__ == '__main__':
  sample_hw()



