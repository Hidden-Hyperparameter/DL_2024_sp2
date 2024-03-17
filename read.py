import torch
with open('read_result.txt','w') as f:
    f.write(torch.load('./models/cifar10_4x_03171020_acc53.json'))