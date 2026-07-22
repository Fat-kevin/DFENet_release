import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log


class EdgeFeatureModulation(nn.Module):
    def __init__(self, channel):
        super(EdgeFeatureModulation, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode="bilinear", align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = (
            self.conv1d(wei.squeeze(-1).transpose(-1, -2))
            .transpose(-1, -2)
            .unsqueeze(-1)
        )
        wei = self.sigmoid(wei)
        x = x * wei
        return x


class ConvBNR(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False
    ):
        super(ConvBNR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
