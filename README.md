# CenterNet_TensorRT_Nano
Centernet use TensorRT speed up on Nano

# TODO

- [x] x86 x64 Dockerfile
- [x] Nano Dockerfile
- [x] Resnet50 to Tensorrt
- [x] Centernet backbone to Tensorrt
- [x] Centernet inference on nano camera
- [ ] upsample for Tensorrt 
- [ ] CI/CD

# Environment
I am use the Docker to build Amd(x64/x86) and Arm(Nano) environment
so use docker or follow my dockerfile to build the environment

## Dockerfile x64_x86
Dockerfile  :  CenterNet_TensorRT_Nano -> docker_pytorch_x86_x64 -> Dockerfile
DockeImage : bluce54088/tensorrt_pytorch_x86_x64:v0

1. Run docker 
```
    docker run --shm-size 24G --gpus all -it -p 6667:22  --name tensorrt_pytorch  bluce54088/tensorrt_pytorch_x86_x64:v0
```
2. check environment
```
    python3
    import tensorrt
```

## Dockerfile Nano
Dockerfile  :  CenterNet_TensorRT_Nano -> docker_tensorrt_python_nano_arm -> Dockerfile
DockeImage : bluce54088/nano_cuda_pytorch:v0

1. Run docker 
```
    docker run -it --net=host --runtime nvidia --device /dev/video0 -e DISPLAY=$DISPLAY -v /usr/lib/python3.6/dist-packages/tensorrt:/usr/lib/python3.6/dist-packages/tensorrt bluce54088/nano_cuda_pytorch:v1
```
2. check environment
```
    python3
    import tensorrt
```

# Quick start test tensorrt 
1.Pull CenterNet_TensorRT_Nano
```
    cd /root/CenterNet_edge/
    git pull
```
2. Run Tesorrt Resnet50 test 
```
    python3 torch2trt_test.py
```

Model           | Device  | without TensorRT | with TensorRT
--------------|:-----:|-----:| --------------------------
Resnet50    | 1080ti |  0.123ms |    0.051ms 
Resnet50    | Nano |  0.438ms |    0.200ms 
  


# Inference Centernet For CoCo Sampledata
```
    python3 inference.py ctdet --exp_id coco_res18 --backbone res_18 --batch_size 1 --load_model ./exp/ctdet/coco_res18/model_best.pth --fix_res --tensorrt
```
Result sample

<img src="https://github.com/kobewangSky/CenterNet_TensorRT_Nano/blob/master/result/0.jpg" width="200" height="200" alt="图片描述文字"/> <img src="https://github.com/kobewangSky/CenterNet_TensorRT_Nano/blob/master/result/1.jpg" width="200" height="200" alt="图片描述文字"/> <img src="https://github.com/kobewangSky/CenterNet_TensorRT_Nano/blob/master/result/2.jpg" width="200" height="200" alt="图片描述文字"/> <img src="https://github.com/kobewangSky/CenterNet_TensorRT_Nano/blob/master/result/3.jpg" width="200" height="200" alt="图片描述文字"/>


# Inference Centernet For Webcam

```
ctdet --exp_id coco_res18 --backbone res_18 --batch_size 1 --load_model ./exp/ctdet/coco_res18/model_best.pth --fix_res --tensorrt --demo Webcam
```

# Reference

[CenterNet](https://github.com/xingyizhou/CenterNet)

[Tensorrt](https://developer.nvidia.com/tensorrt)

[Torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
