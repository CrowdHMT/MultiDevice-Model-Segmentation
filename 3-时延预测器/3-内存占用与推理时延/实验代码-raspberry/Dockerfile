FROM python:3.7
COPY PyTorch /usr/src/app/PyTorch
WORKDIR /usr/src/app/PyTorch
RUN mv /etc/apt/sources.list /etc/apt/sources.list_bk
COPY sources.list /etc/apt/sources.list

#RUN python -m pip install --upgrade pip
#RUN pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/  
#RUN apt-get update
#RUN apt-get install libopenblas-dev -y
#WORKDIR home/pi/tmp

#COPY PyTorch /usr/src/app/PyTorch
#WORKDIR /usr/src/app/PyTorch
RUN pip install future==0.18.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install Pillow==9.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install memory_profiler -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN gpg --keyserver keyserver.ubuntu.com --recv 9165938D90FDDD2E \
&& gpg --keyserver keyserver.ubuntu.com --recv 82B129927FA3303E \
&& gpg --export --armor 9165938D90FDDD2E | apt-key add - \
&& gpg --export --armor 82B129927FA3303E | apt-key add - \
&& apt-get update \
&& apt-get install libopenblas-dev -y \
&& pip3 install -i https://mirrors.aliyun.com/pypi/simple numpy pyyaml \
&& pip install torch-1.6.0a0+b31f58d-cp37-cp37m-linux_armv7l.whl \
&& pip install torchvision-0.8.0a0+10d5a55-cp37-cp37m-linux_armv7l.whl \
&& pip install thop

#RUN pip3 install libgfortran5 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
#RUN pip3 install libopenblas-dev -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
#RUN apt-get install sudo
#RUN  install libgfortran5
#RUN dpkg -i libopenblas0-pthread_0.3.13+ds-3_armhf.deb
#RUN dpkg -i libopenblas-pthread-dev_0.3.13+ds-3_armhf.deb 


#RUN pip install torch-1.6.0a0+b31f58d-cp37-cp37m-linux_armv7l.whl
#RUN pip3 install pillow==9.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
#RUN pip install torchvision-0.8.0a0+10d5a55-cp37-cp37m-linux_armv7l.whl
WORKDIR /usr/src/app
RUN rm -rf Pytorch
#RUN pip3 install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN mkdir -p /workfolder
# COPY ./main_AlexNet.py /workfolder/
COPY ./main_GoogLenet.py /workfolder/
#COPY ./torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl /workfolder/
#RUN pip3 install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl --user
# CMD [ "python", "/workfolder/main_AlexNet.py" ]
CMD [ "python", "/workfolder/main_GoogLenet.py" ]

