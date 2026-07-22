import torch.nn as nn


def resolve_activation(name="relu"):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise NotImplementedError(f"Activation {name} not supported")
