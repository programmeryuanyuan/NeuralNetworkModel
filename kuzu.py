from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1=nn.Linear(28*28, 10)
        # INSERT CODE HERE

    def forward(self, x):
        # print(x.size())
        # exit(0)
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)
        x = F.log_softmax(x,dim=1)
        return x

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()

        self.fc1=nn.Linear(28*28, 28*28)
        self.fc2=nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1=nn.Conv2d(1,32,5)
        self.conv2=nn.Conv2d(32,64,5)
        self.fc1=nn.Linear(64*16, 64*16)
        self.fc2=nn.Linear(64*16, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = self.conv1(x)
        # print(x.size())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)#pooling
        # print(x.size())

        x = self.conv2(x)
        # print(x.size())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        # print(x.size())
        # exit(0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
