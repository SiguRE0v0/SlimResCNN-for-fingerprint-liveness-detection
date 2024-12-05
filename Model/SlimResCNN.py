import torch
import torch.nn as nn
from Model.ResBlock import ResBlock

class SlimResCNN(nn.Module):
    def __init__(self, in_channels):
        super(SlimResCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            ResBlock(in_channels=32, out_channels=32, stride=1),
            ResBlock(in_channels=32, out_channels=32, stride=1),
            ResBlock(in_channels=32, out_channels=32, stride=1)
        )
        self.conv3_1 = ResBlock(in_channels=32, out_channels=64, stride=2)
        self.conv3_2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, stride=1),
            ResBlock(in_channels=64, out_channels=64, stride=1)
        )
        self.conv4_1 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.conv4_2 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=128, stride=1),
            ResBlock(in_channels=128, out_channels=128, stride=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return torch.squeeze(logits)