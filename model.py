import torch.nn as nn
import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self,device) -> None:
#         super().__init__()
        
#     def forward(self,x):
#         return nn.ReLU(self.net(x)+x)
# class BigResBlock(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.net = [ResBlock(),ResBlock(),ResBlock()]
#     def forward(self,x):
#         for lyer in self.net:
#             x=lyer(x)
#         return x
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.resBlock1 = self.resBlock2 = self.resBlock3 = nn.Sequential(
            nn.Conv2d(16,16,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(16,16,5,padding=2),
        )
        self.manyResBlock=[self.resBlock1,self.resBlock2,self.resBlock3]
        # for _ in range(3):
        #     self.manyResBlock.append(self.resBlock)
        # [self.resBlock,self.resBlock,self.resBlock]
        self.first = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # self.blocks = []
        # for _ in range(3):
        #     self.blocks.append(self.manyResBlock)
        #     self.blocks.append(nn.MaxPool2d(2,2))
        self.final = nn.Sequential(
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
        # self.conv1 = nn.Conv2d(3, 16, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 16, 5)
        # self.conv3 = nn.Conv2d(16, 16, 5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
    def ResBlock(self,x,i):
        return nn.ReLU()(x+self.manyResBlock[i](x))
    def ManyResBlock(self,x):
        for i in range(3):
            x = self.ResBlock(x,i)
            x = nn.MaxPool2d(2,2)(x)
        return x
    def forward(self, x):
        bsize = x.shape[0]
        x = self.pre_process(x)
        # print('original',x.shape)
        x = self.first(x)
        # print('after first',x.shape)
        # for i in range(self.blocksNum):
        #     x = self.BigRes(self,x,i)
        # for block in self.blocks:
        #     x = block(x)
            # print(x.shape)
        x = self.ManyResBlock(x)
        # x = self.block(x)
        # # print('after block 1',x.shape)
        # x = self.block(x)
        # # print('after block 2',x.shape)
        # x = self.block(x)
        # print('after block 3',x.shape)
        # print('still alive:')
        x = self.final(x.reshape(bsize,-1))
        # print('final output',x.shape)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 16 * 4 * 4)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def pre_process(self, x):
        return x.float()