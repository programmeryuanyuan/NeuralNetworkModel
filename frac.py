import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.fc1=nn.Linear(2, hid)
        self.fc2=nn.Linear(hid, hid)
        self.fc3=nn.Linear(hid, 1)

    def forward(self, x):
        x=self.fc1(x)
        x = torch.tanh(x)
        self.hid1 = x

        x=self.fc2(x)
        x = torch.tanh(x)
        self.hid2 = x

        x=self.fc3(x)
        x=torch.sigmoid(x)

        return x
        # return 0*x[:,0]

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()

        self.fc1=nn.Linear(2, hid)
        self.fc2=nn.Linear(hid, hid)
        self.fc3=nn.Linear(hid, hid)
        self.fc4=nn.Linear(hid, 1)
    def forward(self, x):
        x=self.fc1(x)
        x = torch.tanh(x)
        self.hid1 = x

        x=self.fc2(x)
        x = torch.tanh(x)
        self.hid2 = x

        x=self.fc3(x)
        x = torch.tanh(x)
        self.hid3 = x

        x=self.fc4(x)
        x=torch.sigmoid(x)

        return x

class DenseNet(torch.nn.Module):
    def __init__(self, hid):
        super(DenseNet, self).__init__()
        self.to2=nn.Linear(2,hid)
        self.to3=nn.Linear(2+hid,hid)
        self.to4=nn.Linear(2+2*hid,1)
    def forward(self, x):

        x1=x
        x2=self.to2(x1)
        x2 = torch.tanh(x2)
        x3=self.to3(torch.cat((x1,x2),1))
        x3 = torch.tanh(x3)
        x4=self.to4(torch.cat((x1,x2,x3),1))
        x4=torch.sigmoid(x4)
        # x3=torch.cat(self.)
        # print(x1.size(),x2.size(),torch.cat((x1,x2),1).size())
        # exit(0)
        self.hid1 = x2
        self.hid2 = x3
        return x4
        # return 0*x[:,0]
