import numpy
import pandas
import torch

flag = torch.cuda.is_available()
print("CUDA available:", flag)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device:", device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 4).cuda())

cuda_version = torch.version.cuda
print("CUDA version:", cuda_version)

cudnn_version = torch.backends.cudnn.version()
print("cuDNN version:", cudnn_version)