# CenterNet_TensorRT_Nano
Centernet use TensorRT speed up on Nano

# TODO

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
    docker run -it --net=host --runtime nvidia  -v /usr/lib/python3.6/dist-packages/tensorrt:/usr/lib/python3.6/dist-packages/tensorrt bluce54088/nano_cuda_pytorch
```
2. check environment
```
    python3
    import tensorrt
```

# Quick start
1.Pull CenterNet_TensorRT_Nano
```
    cd /root/CenterNet_edge/
    
```


