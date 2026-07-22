import torch
import torch.nn.functional as F


def modal_cosine(rgb_feature, depth_feature):
    return F.cosine_similarity(rgb_feature, depth_feature, dim=1, eps=1e-6).unsqueeze(1)


def gate_context(rgb_feature, depth_feature):
    return torch.cat(
        [rgb_feature, depth_feature, modal_cosine(rgb_feature, depth_feature)], dim=1
    )


def residual_depth(depth_feature, gate):
    return depth_feature * gate


def spatially_filter(feature, attention):
    return feature * attention(feature)
