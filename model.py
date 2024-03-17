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
    def createManyResBlock(self,channels=64,BlockNum=3,kernel_size=3):
        self.cnt+=1
        manyResBlock = []
        for i in range(BlockNum):
            x = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size,padding=(kernel_size-1)//2),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(channels,channels,kernel_size,padding=(kernel_size-1)//2),
            )
            self.add_module(f'{self.cnt}_{i}',x)
            manyResBlock.append(x)
        return manyResBlock
    def PassThrough(self,manyResBlock:list,x):
        for i in range(len(manyResBlock)):
            x = F.relu(x+manyResBlock[i](x))
            # if i%2:
            #     x = nn.MaxPool2d(2)(x)
        return x

    def __init__(self):
        super(Net,self).__init__()
        self.cnt=0
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.manyResBlock1 = self.createManyResBlock()
        self.conv2 =  nn.Sequential(
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.manyResBlock2 = self.createManyResBlock(channels=128
                                                    #  ,BlockNum=4
                                                     )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
            )
        self.manyResBlock3 = self.createManyResBlock(channels=256
                                                    #  ,BlockNum=5
                                                     )
        self.final = nn.Sequential(
            nn.Linear(256,128),
            # nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(128,64),
            # nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(64,10),
        )
    def forward(self, x):
        bsize = x.shape[0]
        x = self.pre_process(x)
        x = self.conv1(x)
        # print('after conv1',x.shape)
        x = self.PassThrough(self.manyResBlock1,x)
        # print('after block 1',x.shape)
        x = self.conv2(x)
        # print('after conv2',x.shape)
        x = self.PassThrough(self.manyResBlock2,x)
        # print('after block 2',x.shape)
        x = self.conv3(x)
        # print('after conv3',x.shape)
        x = self.PassThrough(self.manyResBlock3,x)
        # print('after block 3',x.shape)
        x = nn.AvgPool2d(4)(x)
        y = x.reshape(bsize,-1)
        x = self.final(y)
        return x
    def to(self,device):
        self.final.to(device)
        lst1 = [
            self.conv1,
                self.conv2,
                self.conv3
                ]
        for i in lst1:
            i.to(device)
        lst = [
            self.manyResBlock1,
            self.manyResBlock2,
            self.manyResBlock3
            ]
        for i in lst:
            for j in i:
                j.to(device)
    def pre_process(self, x):
        return x.float()