From pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
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

RUN apt-get update\
&& apt-get install -y openssh-server\
&& apt-get install net-tools\
&& apt-get install -y libglib2.0-0\
&& apt-get install -y libsm6 libxext6\
&& apt-get install -y libxrender-dev


RUN mkdir /var/run/sshd\
&& echo 'root:2012620126' | chpasswd\
&& sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
&& sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd\
&& echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22

#CMD ["/usr/sbin/sshd", "-D"]

RUN service ssh restart

RUN mkdir CenterNet_Easy_Pythorch
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR ./cocoapi/PythonAPI
RUN make
RUN python setup.py install --user


CMD ["/usr/sbin/sshd", "-D"]