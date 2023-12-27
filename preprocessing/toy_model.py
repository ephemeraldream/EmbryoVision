import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as func
from torch import Tensor


class EmbryoModel(nn.Module):

    def __init__(self):
        super(EmbryoModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reg1 = nn.Linear(in_features=256 * 16 * 16, out_features=512)
        self.reg2 = nn.Linear(in_features=512, out_features=50)
        self.cls1 = nn.Linear(in_features=256 * 16 * 16, out_features=512)
        self.cls2 = nn.Linear(in_features=512, out_features=25 * 3)

    def forward(self, X):
        X = self.pool(func.relu(self.conv1(X)))
        X = self.pool(func.relu(self.conv2(X)))
        X = self.pool(func.relu(self.conv3(X)))

        X = X.view(X.size(0), -1)

        reg = func.relu(self.fc1_reg(X))
        reg = self.fc2_reg(reg)

        cls = func.relu(self.fc1_cls(X))
        cls = self.fc2_cls(cls)
        cls = cls.view(-1, 25, 3)
        cls = func.softmax(cls, dim=2)

        return reg, cls
