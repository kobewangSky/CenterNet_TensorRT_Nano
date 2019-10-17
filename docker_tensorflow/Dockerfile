From tensorflow/tensorflow:2.0.0-gpu-py3
RUN pip install --upgrade pip\
&& pip install opencv-python\
&& pip install Cython\
&& pip install wandb\
&& apt-get update\
&& apt-get install -y openssh-server\
&& apt-get install net-tools\
&& apt-get install -y git
RUN mkdir /var/run/sshd\
&& echo 'root:2012620126' | chpasswd\
&& sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
&& sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd\
&& echo "export VISIBLE=now" >> /etc/profile
#
EXPOSE 22
#

RUN mkdir CenterNet_Easy_Pytorch\
&& git clone https://github.com/cocodataset/cocoapi.gitf
WORKDIR ./cocoapi/PythonAPI
RUN make\
&& python setup.py install --user
#RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh"\
#&& echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc


#CMD ["/usr/sbin/sshd", "-D"]

#docker run --shm-size 24G --gpus all -p 6668:22 -v CenterNet_Easy_Pytorch:~/workspace/CenterNet_Easy_Pytorch --name centernet_tensorflow bluce54088/centernet_tensorflow
#