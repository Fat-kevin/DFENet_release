import torch
import torch.nn as nn

try:
    from .dfenet_factory import wire
    from .dfenet_runtime import flow
except ImportError:
    from dfenet_factory import wire
    from dfenet_runtime import flow


class DFENet(nn.Module):
    def __init__(self):
        super(DFENet, self).__init__()
        wire(self)

    def forward(self, x, depth):
        return flow(self, x, depth)

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)["model"], strict=False)
        print(f"Loaded RGB backbone weights from {pre_model}")

    def load_pre2(self, pre_model):
        self.smt1.load_state_dict(torch.load(pre_model)["model"], strict=False)
        print(f"Loaded depth backbone weights from {pre_model}")
