import torch.nn as nn
from torch import Tensor

try:
    from .DEConv import DEConv
    from .Dysample import DySample_UP
    from .RCM import RCM
    from .SCSA import SCSA
except ImportError:
    from DEConv import DEConv
    from Dysample import DySample_UP
    from RCM import RCM
    from SCSA import SCSA


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
        nn.BatchNorm2d(out_planes),
        nn.GELU(),
    )


class ContextualAttentionBlock(nn.Module):
    def __init__(self, dim, head_num=4, drop_path=0.0):
        super().__init__()
        self.shared_dw = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.shared_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.scsa = SCSA(dim=dim, head_num=head_num)
        self.rcm = RCM(dim=dim, drop_path=drop_path)
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        identity = x
        shared = self.shared_pw(self.shared_dw(x))
        x1 = self.scsa(shared) + shared
        x2 = self.rcm(shared) + shared
        out = self.output_proj(x1 + x2)
        return out + identity


class ChannelExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelExcitation, self).__init__()
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
        return x * y.expand_as(x)


class EdgeGuidanceUnit(nn.Module):
    def __init__(
        self, dim, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )
        self.spatial_mixing = DEConv(dim)
        self.channel_attention = ChannelExcitation(dim, reduction=4)
        self.up = DySample_UP(in_channels=dim, scale=2, style="lp")

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.channel_attention(x)
        x = shortcut + self.mlp(x)
        x = self.up(x)
        return x
