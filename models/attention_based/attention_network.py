import torch
from torch import nn
from EmbryoVision.models.attention_based.se_and_att import SqueezeExcitation, AttentionBlock
import torch.nn.functional as F


class EmbryoModelWithAttention(nn.Module):
    def __init__(self):
        super(EmbryoModelWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
        self.se1 = SqueezeExcitation(channel=64)
        self.se2 = SqueezeExcitation(channel=128)
        self.se3 = SqueezeExcitation(channel=256)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=3)
        self.attention = AttentionBlock(in_features=35840, out_features=256)
        self.reg1 = nn.Linear(in_features=256, out_features=512)
        self.reg2 = nn.Linear(in_features=512, out_features=50)
        self.cls1 = nn.Linear(in_features=256, out_features=512)
        self.cls2 = nn.Linear(in_features=512, out_features=25 * 5)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = self.se1(X)
        X = self.pooling(X)
        X = F.relu(self.conv2(X))
        X = self.se2(X)
        X = self.pooling(X)
        X = F.relu(self.conv3(X))
        X = self.se3(X)
        X = self.pooling(X)

        X = X.view(X.size(0), -1)
        X = self.attention(X)

        reg = F.relu(self.reg1(X))
        reg = self.reg2(reg)
        reg = reg.view(-1, 25, 2)

        cls = F.relu(self.cls1(X))
        cls = self.cls2(cls)
        cls = cls.view(-1, 25, 5)
        return reg, cls
