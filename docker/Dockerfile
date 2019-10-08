From pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
RUN pip install --upgrade pip\ 
&& pip install Cython\
&& pip install torchvision\
&& pip install opencv-python\
&& pip install numba\
&& pip install progress\
&& pip install matplotlib\
&& pip install easydict\
&& pip install scipy

RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR ./cocoapi/PythonAPI
RUN make
RUN python setup.py install --user