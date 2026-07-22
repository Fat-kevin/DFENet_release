import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dsf_activations import resolve_activation
except ImportError:
    from dsf_activations import resolve_activation


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
        norm_type="bn",
        num_groups=8,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if is_transposed else nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        if norm_type == "bn":
            self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        elif norm_type == "gn":
            self.add_module(name="gn", module=nn.GroupNorm(num_groups, out_planes))
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        if act_name is not None:
            self.add_module(name=act_name, module=resolve_activation(name=act_name))


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1, norm_type="bn")
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1, norm_type="gn", num_groups=8)
        self.conv3x3_1 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation, norm_type="bn"
        )
        self.conv3x3_2 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation, norm_type="bn"
        )
        self.conv3x3_3 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation, norm_type="bn"
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1, norm_type="bn"),
            ConvBNReLU(out_dim, out_dim, 3, padding=1, norm_type="bn"),
        )

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = F.interpolate(y0, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))
