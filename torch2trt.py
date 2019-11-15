import torch
from torch2trt import torch2trt
from torchvision.models.resnet import resnet18
import time

# create some regular pytorch model...
model = resnet18(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

start = time.time()
y = model(x)
end = time.time()
print(end - start)


start = time.time()
y_trt = model_trt(x)
end = time.time()
print(end - start)

torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))

start = time.time()
y_trt = model_trt(x)
end = time.time()
print(end - start)

print(1)
