#https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson
#https://medium.com/@jackycsie/jetson-nano-9d89cbf2fc18

#1.use JetPack 4.2.2 image

FROM nvcr.io/nvidia/l4t-base:r32.2

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends make g++
# cp -r /usr/local/cuda/samples/ ./ 使用cuda的sample
COPY ./samples /tmp/samples
#libnvToolsExt.so.1
WORKDIR /tmp/samples/1_Utilities/deviceQuery
RUN make clean && make

#https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/

WORKDIR /root/
RUN sudo apt install -y python3-pip --reinstall\
&& wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl\
&& pip3 install numpy torch-1.3.0-cp36-cp36m-linux_aarch64.whl\
&& apt install -y git\
&& git clone https://github.com/kobewangSky/CenterNet_edge.git\
&& apt install -y python3-opencv

RUN apt-get install -y openssh-server\
&& mkdir /var/run/sshd\
&& echo 'root:2012620126' | chpasswd\
&& sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
&& sed -i 's/\/usr\/lib\/openssh\/sftp-server/internal-sftp/' /etc/ssh/sshd_config\
&& sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd\
&& echo "export VISIBLE=now" >> /etc/profile

RUN apt-get update\
&& apt-get install libjpeg-dev zlib1g-dev\
&& pip3 install pillow\
&& pip3 install torchvision


#docker run -it --runtime nvidia devicequery
#docker run --runtime nvidia --network host -it --device /dev/video0 -e DISPLAY=$DISPLAY bluce54088/nano_cuda

#Tensorrt
#docker run -it --rm --net=host --runtime nvidia  -v /usr/lib/python3.6/dist-packages/tensorrt:/usr/lib/python3.6/dist-packages/tensorrt penolove/jetson_nano_tensorrt_yolov3

