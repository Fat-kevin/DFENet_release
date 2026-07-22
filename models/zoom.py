import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
import torch.nn.functional as F
from einops import rearrange


def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation
        )
        self.conv3x3_2 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation
        )
        self.conv3x3_3 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_lm = ConvBNReLU(2 * in_dim, 2 * in_dim, 1)
        self.initial_merge = ConvBNReLU(2 * in_dim, 2 * in_dim, 1)
        self.num_groups = num_groups
        self.trans = nn.Sequential(
            ConvBNReLU(2 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, l, m):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l(l)
        m = self.conv_m(m)
        lm = torch.cat([l, m], dim=1)
        attn = self.conv_lm(lm)
        attn = rearrange(
            attn, "b (nb ng d) h w -> (b ng) (nb d) h w", nb=2, ng=self.num_groups
        )
        attn = self.trans(attn)
        attn = attn.unsqueeze(dim=2)
        x = self.initial_merge(lm)
        x = rearrange(x, "b (nb ng d) h w -> (b ng) nb d h w", nb=2, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(b ng) d h w -> b (ng d) h w", ng=self.num_groups)
        return x
