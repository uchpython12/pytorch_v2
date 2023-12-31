import torch
from d2l import torch as d2l
# 检查GPU是否可用
gpu_count = torch.device("mps")
print(gpu_count)
d2l.try_gpu()
if torch.cuda.is_available():
    # 获取GPU数量
    # gpu_count = torch.cuda.device_count()
    gpu_count = torch.device("mps")
    print(f"发现 {gpu_count} 个可用的GPU")

    # 遍历每个GPU并打印信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("未找到可用的GPU")


