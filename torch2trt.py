import torch
from torch2trt import torch2trt
from torchvision.models.resnet import resnet50
import time
import numpy as np
# create some regular pytorch model...
model = resnet50(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 512, 512)).cuda()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# convert to TensorRT feeding sample data as input
print('x = torch.ones((1, 3, 512, 512)).cuda()')
#model_trt = torch2trt(model, [x])
# model_trt_16 = torch2trt(model, [x], fp16_mode=True)
# model_trt_8 = torch2trt(model, [x], int8_mode=True)
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

# print('y_trt = model_trt(x)')
# start = time.time()
# y_trt = model_trt(x)
# end = time.time()
# print(end - start)
#
# start = time.time()
# y_trt = model_trt_16(x)
# end = time.time()
# print(end - start)

# start = time.time()
# y_trt = model_trt_8(x)
# end = time.time()
# print(end - start)

#torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

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
