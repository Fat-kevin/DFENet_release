class K:
    A0 = "rgb_pre_channel"
    A1 = "depth_pre_spatial"
    B0 = "rgb_align"
    B1 = "depth_align"
    G0 = "gate_conv1"
    G1 = "gate_conv2"
    G2 = "gate_conv3"
    R = "relu"
    S = "sigmoid"
    X = "cross_attention"
    C = "channel_attention"
    P = "spatial_attention"
    Z = "aspp"


CHAIN_G = (K.G0, K.R, K.G1, K.R, K.G2, K.S)
