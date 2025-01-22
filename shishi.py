import torch
print(torch.cuda.is_available())  # 是否支持 GPU
print(torch.version.cuda)         # PyTorch 使用的 CUDA 版本
print(torch.cuda.get_device_name(0))  # GPU 设备名称
