# CPU、内存、网速实时监控

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import psutil as p
import win32gui,win32con

def show():#增加窗口置顶功能
  global hwnd
  #win32gui.SetForegroundWindow (hwnd)#这句好象不行
  win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0,0,500,400, \
                        win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE| \
                        win32con.SWP_NOOWNERZORDER|win32con.SWP_SHOWWINDOW)
POINTS = 300
fig, ax = plt.subplots(2,facecolor='#FFDAB9')#设置上下两个图、面板色。#,edgecolor='#7FFF00'好象无效
fig.canvas.set_window_title(u'电脑性能(cpu、内存、网速）实时监测')#设置窗口标题
fig.canvas.toolbar.destroy()#取消工具栏
fig.canvas.callbacks.callbacks.clear()#清除回调信息

#ax为列表，分别设置
ax[0].set_ylim([0, 100])
ax[0].set_xlim([0, POINTS])
ax[0].set_autoscale_on(False)
ax[0].set_xticks([])
ax[0].set_yticks(range(0, 101, 20))
#ax[0].set_facecolor('#A9A9A9')#可设置底色，支持'black')#matplotlib颜色分层设置
ax[0].grid(True)

ax[1].set_ylim([-5, 50])#设置0线上浮
ax[1].set_xlim([0, POINTS])
ax[1].set_autoscale_on(False)
ax[1].set_xticks([])
ax[1].set_yticks(range(0, 51, 10))
ax[1].grid(True)

# 设置CPU、内存横坐标数据位
cpu = [None] * POINTS
mem=[None] * POINTS
# 设置接收字节(下载）横坐标数据位
down = [None] * POINTS
#设置第一个图
cpu_l, = ax[0].plot(range(POINTS), cpu, label='Cpu%')
mem_l, = ax[0].plot(range(POINTS), mem, label='Mem%')
ax[0].legend(loc='upper center', ncol=4, prop=font_manager.FontProperties(size=10))#打标
#设置第二个图
down_l, = ax[1].plot(range(POINTS), down, label='Down(M)')
ax[1].legend(loc='upper center', ncol=4, prop=font_manager.FontProperties(size=10))

before =p.net_io_counters().bytes_recv#获取网络字节数
#把查找窗口句柄放在这里，不在SHOW()里，
hwnd=win32gui.FindWindow(None,u'电脑性能(cpu、内存、网速）实时监测')#查找本窗口句柄
print(hwnd)#调试用
#把win32gui.SetWindowPos(放在这里不行列。

def get_delta():#获取下载变化值
  global before
  now = p.net_io_counters().bytes_recv
  delta = (now-before)/102400#变成K再除100，大致相当于多少M宽带。
  before = now
  return  delta #返回改变量

def OnTimer(ax):
  global cpu, mem, down
  show()
  tmp = get_delta()#得到下载字节数的变化值
  cpu_p=p.cpu_percent()#读取CPU使用百分比
  cpu = cpu[1:] + [cpu_p]#加入到数据末尾
  mem_p=p.virtual_memory().percent#读取内存使用百分比
  mem =mem[1:] + [mem_p]#加入到数据末尾
  cpu_l.set_ydata(cpu)#设置新数据
  mem_l.set_ydata(mem)#设置新数据   
  down = down[1:] + [tmp]
  down_l.set_ydata(down)#设置新数据
###下面这部分可以忽略
##    while True:
##        try:
##            ax.draw_artist(cpu_l)
##            ax.draw_artist(mem_l)
##            ax.draw_artist(down_l)
##            break
##        except:
##            pass
  ax.figure.canvas.draw()#刷新画布

def start_monitor():
  timer = fig.canvas.new_timer(interval=1000)#1秒刷新一次
  timer.add_callback(OnTimer, ax[1])#只加一个即可
  #timer.add_callback(OnTimer, ax[0])
  timer.start()
  plt.show()

if __name__ == '__main__':
  start_monitor()
