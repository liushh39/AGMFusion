from collections import OrderedDict
import torch.nn as nn


class residualBlock1(nn.Module):
    def __init__(self, in_channels=32, k=3, n=32, s=1, g=32):
        super(residualBlock1, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, n, 1, stride=s, padding=0)
        m['bn1'] = nn.BatchNorm2d(n)
        m['ReLU1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(n, n, k, stride=s, padding=1, groups=g)
        m['bn2'] = nn.BatchNorm2d(n)
        m['ReLU2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(n, in_channels, 1, stride=s, padding=0)
        m['bn3'] = nn.BatchNorm2d(in_channels)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.group1(x) + x
        out = self.relu(out)
        return out


class AGMFusion(nn.Module):
    def __init__(self, n_residual_blocks):
        super(AGMFusion, self).__init__()
        self.n_residual_blocks = n_residual_blocks

        self.conv1 = nn.Conv2d(2, 16, 7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock1(32, n=16, g=16))

        self.conv5 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = (self.relu1(self.bn1(self.conv1(x))))
        y = x.clone()
        x = (self.relu2(self.bn2(self.conv2(x))))

        for i in range(self.n_residual_blocks):
            x = self.__getattr__('residual_block' + str(i + 1))(x)

        x = (self.relu5(self.bn5(self.conv5(x)))) + y

        return self.sigmoid(self.conv6(x))
