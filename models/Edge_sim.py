import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.size(2) * x.size(3) - 1
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mean).pow(2)
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class ECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c).unsqueeze(1)
        y = self.sigmoid(self.conv(y)).squeeze(1).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dwconv(x)
        out = self.pwconv(out)
        out = self.bn1(out)
        return self.relu(out + residual)


class EdgeAttentionModule(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channel_attention = ECAModule(channels)
        self.spatial_attention = SimAM()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EdgeAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_convs = nn.ModuleList(
            [
                DepthwiseResidualBlock(64, 64),
                DepthwiseResidualBlock(64, 64),
                DepthwiseResidualBlock(64, 64),
                DepthwiseResidualBlock(64, 64),
            ]
        )
        self.edge_attention = EdgeAttentionModule(64)
        self.edge_pred_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x96, x48, x24, x12):
        features = [
            self.feature_convs[0](x96),
            self.feature_convs[1](x48),
            self.feature_convs[2](x24),
            self.feature_convs[3](x12),
        ]
        for i in range(len(features) - 1, 0, -1):
            upsampled = F.interpolate(
                features[i],
                size=features[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            features[i - 1] = features[i - 1] + upsampled
        final_feature = features[0]
        final_feature = self.edge_attention(final_feature)
        edge = self.edge_pred_conv(final_feature)
        return edge
