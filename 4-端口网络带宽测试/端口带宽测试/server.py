from __future__ import print_function

import datetime
import socket
import xlwt

HOST = "192.168.43.57"
PORT = 8081
BUFFER = 4096

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(0)
print('listening at %s:%s\n\r' %(HOST, PORT))
# 创建一个可以写入网络带宽的excel表
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
worksheet.write(0, 0, '时间/(min)')  # 将时间写入列表
worksheet.write(1, 0, '带宽/(MB/s)')  # 将时间写入列表
connectnum = 0 #连接次数
connectime = 0 #当前时间
while True:
    worksheet.write(0, connectnum+1, connectime)  # 将时间写入列表
    client_sock, client_addr = sock.accept()
    # 开始时间
    starttime = datetime.datetime.now()
    print(starttime, end="")
    print('%s:%s connected\n\r' % client_addr)
    count = 0
    while True:
        data = client_sock.recv(BUFFER)
        if data:
            count += len(data)
            del data
            continue
        client_sock.close()

        endtime = datetime.datetime.now()
        print(endtime)
        print('%s:%s disconnected\n\r' % client_addr)
        print('bytes transferred: %d' % count)
        delta = endtime - starttime
        delta = delta.seconds + delta.microseconds / 1000000.0
        print('time used (seconds): %f' % delta)
        speed = count / 1024 / 1024 / delta
        print('averaged speed (MB/s): %f\n\r' % (count / 1024 / 1024 / delta))
        worksheet.write(1, connectnum+1, speed)  # 将网速写入列表
        break
    connectnum = connectnum + 1  # 每次连接次数加一
    connectime = connectime + 0.5 #每次连接时间之间相差30s即半分钟
    workbook.save('bandwith.xls')  # 保存文件
sock.close()