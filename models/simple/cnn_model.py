import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as func
from torch import Tensor


class EmbryoModel(nn.Module):

    def __init__(self):
        super(EmbryoModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=3)
        self.reg1 = nn.Linear(in_features=35840, out_features=512)
        self.reg2 = nn.Linear(in_features=512, out_features=50)
        self.cls1 = nn.Linear(in_features=35840, out_features=512)
        self.cls2 = nn.Linear(in_features=512, out_features=25 * 5)
        self.hole1 = nn.Linear(in_features=35840, out_features=100)
        self.hole2 = nn.Linear(in_features=100, out_features=1)
        self.hole3 = nn.Sigmoid()

    def forward(self, X):
        X = self.pooling(func.relu(self.conv1(X)))
        X = self.pooling(func.relu(self.conv2(X)))
        X = self.pooling(func.relu(self.conv3(X)))

        X = X.view(X.size(0), -1)

        reg = func.relu(self.reg1(X))
        reg = self.reg2(reg)
        reg = reg.view(-1, 25, 2)

        cls = func.relu(self.cls1(X))
        cls = self.cls2(cls)
        cls = cls.view(-1, 25, 5)

        hole = func.relu(self.hole1(X))
        hole = self.hole2(hole)
        hole = self.hole3(hole)
        return reg, cls, hole
