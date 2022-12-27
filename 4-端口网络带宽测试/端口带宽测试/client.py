import datetime
import socket
import time

HOST = "192.168.43.57"
PORT = 8081
BUFFER = 4096

testdata = b'x' * BUFFER * 4
while True:
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect((HOST, PORT))
  for i in range(1, 1000):
      sock.send(testdata)
  # 30s
  sock.close()
  time.sleep(60)