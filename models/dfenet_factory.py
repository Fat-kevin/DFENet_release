import torch.nn as nn

try:
    from .dfenet_blocks import ContextualAttentionBlock, EdgeGuidanceUnit, conv3x3_bn_relu
    from .EAM import EdgeFeatureModulation
    from .Edge_sim import EdgeAwareModule
    from .RGBDFusion2 import ModalityAdaptiveGatedFusion
    from .smt import smt_t
    from .zoom import ConvBNReLU, MHSIU, SimpleASPP
except ImportError:
    from dfenet_blocks import ContextualAttentionBlock, EdgeGuidanceUnit, conv3x3_bn_relu
    from EAM import EdgeFeatureModulation
    from Edge_sim import EdgeAwareModule
    from RGBDFusion2 import ModalityAdaptiveGatedFusion
    from smt import smt_t
    from zoom import ConvBNReLU, MHSIU, SimpleASPP


def _bind(o, n, v):
    setattr(o, n, v)
    return o


def _conv(n, c, k=1, p=0):
    return n, conv3x3_bn_relu(c, 64, k=k, s=1, p=p)


def _proj(n, src, dst=64):
    return n, nn.Conv2d(in_channels=src, out_channels=dst, kernel_size=1, bias=False)


def _tap(n, src, dst=64):
    return n, nn.Conv2d(src, dst, kernel_size=1)


def _pred():
    return nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
    )


def _items():
    return (
        ("smt", smt_t()),
        ("smt1", smt_t()),
        ("tra512", SimpleASPP(512, 64)),
        ("tra256", ConvBNReLU(256, 64, 3, 1, 1)),
        ("tra128", ConvBNReLU(128, 64, 3, 1, 1)),
        ("tra64", ConvBNReLU(64, 64, 3, 1, 1)),
        ("trD512", SimpleASPP(512, 64)),
        ("trD256", ConvBNReLU(256, 64, 3, 1, 1)),
        ("trD128", ConvBNReLU(128, 64, 3, 1, 1)),
        ("trD64", ConvBNReLU(64, 64, 3, 1, 1)),
        ("fuse64", MHSIU(64, 4)),
        ("fuse128", MHSIU(64, 4)),
        ("fuse256", MHSIU(64, 4)),
        ("fuse512", MHSIU(64, 4)),
        ("fusion_module1", ModalityAdaptiveGatedFusion(in_channels=64)),
        ("fusion_module2", ModalityAdaptiveGatedFusion(in_channels=64)),
        ("fusion_module3", ModalityAdaptiveGatedFusion(in_channels=64)),
        ("fusion_module4", ModalityAdaptiveGatedFusion(in_channels=64)),
        ("EDGE", EdgeAwareModule()),
        ("efm1", EdgeFeatureModulation(64)),
        ("efm2", EdgeFeatureModulation(64)),
        ("efm3", EdgeFeatureModulation(64)),
        ("efm4", EdgeFeatureModulation(64)),
        ("MAM_1", ContextualAttentionBlock(dim=64, head_num=4)),
        ("MAM_2", ContextualAttentionBlock(dim=128, head_num=8)),
        ("MAM_3", ContextualAttentionBlock(dim=128, head_num=8)),
        ("MAM_4", ContextualAttentionBlock(dim=64, head_num=4)),
        ("PCM1", EdgeGuidanceUnit(dim=64)),
        ("PCM2", EdgeGuidanceUnit(dim=64)),
        ("PCM3", EdgeGuidanceUnit(dim=64)),
        ("PCM4", EdgeGuidanceUnit(dim=64)),
        _proj("deconv_layer_2", 128),
        _proj("deconv_layer_3", 128),
        _proj("deconv_layer_4", 128),
        _conv("dwc1", 64, k=1, p=0),
        _conv("dwc2", 128, k=1, p=0),
        _conv("dwc3", 128, k=1, p=0),
        _conv("dwcon_2", 128, k=3, p=1),
        _conv("dwcon_3", 128, k=3, p=1),
        _conv("dwcon_4", 64, k=3, p=1),
        _tap("xf_22", 128),
        _tap("xf_23", 128),
        _tap("xf_24", 128),
        _tap("xf_33", 128),
        _tap("xf_34", 128),
        ("predict_layer_1", _pred()),
        ("predtrans2", nn.Conv2d(64, 1, kernel_size=3, padding=1)),
        ("predtrans3", nn.Conv2d(64, 1, kernel_size=3, padding=1)),
        ("predtrans4", nn.Conv2d(64, 1, kernel_size=3, padding=1)),
    )


def wire(o):
    for n, v in _items():
        _bind(o, n, v)
    return o
