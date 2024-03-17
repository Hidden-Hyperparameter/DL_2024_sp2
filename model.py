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
            if i%2:
                x = nn.MaxPool2d(2)(x)
        return x

    def __init__(self):
        super(Net,self).__init__()
        self.cnt=0
        self.first = nn.Sequential(
            nn.Conv2d(3,128,5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64,128,7,stride=2,padding=3),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(2)
        )
        self.manyResBlock1 = self.createManyResBlock(channels=128)
        # print('------------1-------------')
        self.conv2 =  nn.Sequential(
            nn.Conv2d(128,128,5,stride=2,padding=2),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(2)
        )
        self.manyResBlock2 = self.createManyResBlock(channels=128,BlockNum=5)
        # print('------------2-------------')
        # self.conv3 =  nn.Sequential(
        #     nn.Conv2d(128,256,3,stride=2,padding=1),
        #     nn.MaxPool2d(2,2)
        # )
        # self.manyResBlock3 = self.createManyResBlock(channels=256,BlockNum=5)
        # print('------------3-------------')
        # self.conv4 =  nn.Sequential(
        #     nn.Conv2d(256,512,3,stride=2,padding=1),
        #     nn.MaxPool2d(2,2)
        # )
        # self.manyResBlock4 = self.createManyResBlock(channels=512)
        
        # for _ in range(3):
        #     self.manyResBlock.append(self.resBlock)
        # [self.resBlock,self.resBlock,self.resBlock]
        
        # self.blocks = []
        # for _ in range(3):
        #     self.blocks.append(self.manyResBlock)
        #     self.blocks.append(nn.MaxPool2d(2,2))
        self.final = nn.Sequential(
            nn.Linear(128,256),
            # nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(256,128),
            # nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(128,10),
        )
        # print(self.manyResBlock1[0])
        # self.conv1 = nn.Conv2d(3, 16, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 16, 5)
        # self.conv3 = nn.Conv2d(16, 16, 5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
    # def ManyResBlock(self,x):
    #     for i in range(self.blockNum):
    #         x = self.ResBlock(x,i)
    #         if i%2:
    #             x = nn.MaxPool2d(2,2)(x)
    #     return x
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
        # print('before self.conv1')
        # x = self.conv1(x)
        # print(self.manyResBlock1[0])
        self.manyResBlock1[0](x)
        # print('----------------')
        x = self.PassThrough(self.manyResBlock1,x)
        # print('before self.conv2')
        x = self.conv2(x)
        x = self.PassThrough(self.manyResBlock2,x)
        # print('finish')
        # x = self.conv3(x)
        # x = self.PassThrough(self.manyResBlock3,x)
        # x = self.conv4(x)
        # x = self.PassThrough(self.manyResBlock4,x)
        # print('after blocks',x.shape)
        # print('still alive:')
        x = nn.MaxPool2d(4,4)(x)
        y = x.reshape(bsize,-1)
        # print(y.shape)
        x = self.final(y)
        # print('final output',x.shape)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 16 * 4 * 4)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
    def to(self,device):
        self.first.to(device)
        self.final.to(device)
        lst1 = [
            self.conv1,
                self.conv2,
                # self.conv3
                ]
        for i in lst1:
            i.to(device)
        lst = [
            self.manyResBlock1,
            self.manyResBlock2,
            # self.manyResBlock3
            ]
        for i in lst:
            for j in i:
                j.to(device)
    def pre_process(self, x):
        return x.float()