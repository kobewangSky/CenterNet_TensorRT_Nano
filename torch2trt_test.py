import torch
from torch2trt.torch2trt import torch2trt
from torchvision.models.resnet import resnet50
import time
import numpy as np
# create some regular pytorch model...
model = resnet50(pretrained=True).eval().cuda()
# create example data
x = torch.ones((1, 3, 512, 512)).cuda()

#get model parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# convert to TensorRT feeding sample data as input
print('x = torch.ones((1, 3, 512, 512)).cuda()')
model_trt = torch2trt(model, [x])

timelist = []
print('y = model(x)')
for i in range(101):

    start = time.time()
    y = model(x)
    end = time.time()
    print(end - start)
    timelist.append(end - start)
temp = np.array(timelist)
print(sum(timelist))
print('mean = {}'.format(temp.mean()))

torch.save(model_trt.state_dict(), 'Resnet_50.pth')


from torch2trt import TRTModule
model_trt_load = TRTModule()

model_trt_load.load_state_dict(torch.load('Resnet_50.pth'))
timelist.clear()
timelist = []
for i in range(101):
    start = time.time()
    y_trt = model_trt_load(x)
    end = time.time()
    print(end - start)
    timelist.append(end - start)
temp = np.array(timelist)
print(sum(timelist))
print('mean = {}'.format(temp.mean()))
print(1)
