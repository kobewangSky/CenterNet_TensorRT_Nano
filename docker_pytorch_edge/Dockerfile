From pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
RUN pip install --upgrade pip\
&& pip install Cython\
&& pip install torchvision\
&& pip install opencv-python\
&& pip install numba\
&& pip install progress\
&& pip install matplotlib\
&& pip install easydict\
&& pip install scipy\
&& pip install wandb
#
RUN apt-get update\
&& apt-get install -y openssh-server\
&& apt-get install net-tools\
&& apt-get install -y libglib2.0-0\
&& apt-get install -y libsm6 libxext6\
&& apt-get install -y libxrender-dev
#
#
RUN mkdir /var/run/sshd\
&& echo 'root:2012620126' | chpasswd\
&& sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
&& sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd\
&& echo "export VISIBLE=now" >> /etc/profile
#
EXPOSE 22
#
#CMD ["/usr/sbin/sshd", "-D"]

RUN mkdir CenterNet_Easy_Pytorch
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR ./cocoapi/PythonAPI
RUN make
RUN python setup.py install --user
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh"\
&& echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc


CMD ["/usr/sbin/sshd", "-D"]

#docker run --shm-size 24G --gpus all -p 6667:22 -v centernet_edge:/workspace/CenterNet_Easy_Pytorch --name centernet_edge bluce54088/centernet_edge
#