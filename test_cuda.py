import torch

x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()
print("cuda: ",torch.cuda.is_available())

print(torch.__version__)
