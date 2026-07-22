import torch.nn as nn

try:
    from .dsf_attention import ChannelAttention, QKVAttention, SpatialAttention
    from .dsf_layers import SimpleASPP
    from .dsf_runtime import flow
except ImportError:
    from dsf_attention import ChannelAttention, QKVAttention, SpatialAttention
    from dsf_layers import SimpleASPP
    from dsf_runtime import flow


class ModalityAdaptiveGatedFusion(nn.Module):
    def __init__(self, in_channels=64, reduction=8):
        super(ModalityAdaptiveGatedFusion, self).__init__()
        self.rgb_align = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.depth_align = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gate_conv1 = nn.Conv2d(in_channels * 2 + 1, 32, kernel_size=3, padding=1)
        self.gate_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.gate_conv3 = nn.Conv2d(16, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.cross_attention = QKVAttention(in_channels, reduction)
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.rgb_pre_channel = ChannelAttention(in_channels, reduction)
        self.depth_pre_spatial = SpatialAttention(kernel_size=7)
        self.aspp = SimpleASPP(in_channels, in_channels)

    def forward(self, rgb_feature, depth_feature):
        return flow(self, rgb_feature, depth_feature)
