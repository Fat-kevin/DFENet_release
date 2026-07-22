import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return attention


class QKVAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(QKVAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.up_conv = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, query_rgb, key_value_depth):
        query = self.query(query_rgb)
        key = self.key(key_value_depth)
        value = self.value(key_value_depth)
        attention_map = self.softmax(torch.matmul(query, key.transpose(-2, -1)))
        enhanced_feature = torch.matmul(attention_map, value)
        enhanced_feature = self.up_conv(enhanced_feature)
        return enhanced_feature
