import torch
from torch2trt import torch2trt
from torchvision.models.resnet import resnet152
import time

# create some regular pytorch model...
model = resnet152(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 512, 512)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
model_trt_16 = torch2trt(model, [x], fp16_mode=True)
model_trt_8 = torch2trt(model, [x], int8_mode=True)

start = time.time()
y = model(x)
end = time.time()
print(end - start)


start = time.time()
y_trt = model_trt(x)
end = time.time()
print(end - start)

start = time.time()
y_trt = model_trt_16(x)
end = time.time()
print(end - start)

start = time.time()
y_trt = model_trt_8(x)
end = time.time()
print(end - start)

# torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
#
# from torch2trt import TRTModule
#
# model_trt = TRTModule()
#
# model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
#
# start = time.time()
# y_trt = model_trt(x)
# end = time.time()
# print(end - start)
#
# print(1)
