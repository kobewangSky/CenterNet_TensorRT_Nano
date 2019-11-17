FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update\
&& apt-get install -y openssh-server\
&& apt-get install -y net-tools\
&& apt-get install -y vim

RUN mkdir /var/run/sshd\
&& echo 'root:2012620126' | chpasswd\
&& sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
&& sed -i 's/\/usr\/lib\/openssh\/sftp-server/internal-sftp/' /etc/ssh/sshd_config\
&& sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd\
&& echo "export VISIBLE=now" >> /etc/profile

WORKDIR /root/
RUN wget https://www.dropbox.com/s/7cuxj5munuto6bw/nv-tensorrt-repo-ubuntu1804-cuda10.1-trt6.0.1.5-ga-20190913_1-1_amd64.deb?dl=0\
&& dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt6.0.1.5-ga-20190913_1-1_amd64.deb?dl=0\
&& apt-key add /var/nv-tensorrt-repo-cuda10.1-trt6.0.1.5-ga-20190913/7fa2af80.pub\
&& apt-get update\
&& apt-get install -y tensorrt\
&& apt-get install -y python3-libnvinfer-dev\
&& apt-get install -y python3-libnvinfer

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y python3-pip\
&& apt-get install -y git\
&& apt-get install -y python3-opencv

RUN pip3 install torch torchvision\
&& pip3 install onnx

RUN apt-get install -y libprotobuf* protobuf-compiler ninja-build

WORKDIR /root/
RUN git clone https://github.com/kobewangSky/CenterNet_edge.git
WORKDIR /root/CenterNet_edge/
RUN python3 setup.py install --plugins




#docker run --shm-size 24G --gpus all -it -p 6667:22  --name tensorrt_pytorch_amd  bluce54088/tensorrt_pytorch_amd
#TensorRT
#