import torch
print("CUDA 是否可用:", torch.cuda.is_available())
print("可用的 GPU 数量:", torch.cuda.device_count())
print("当前设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
