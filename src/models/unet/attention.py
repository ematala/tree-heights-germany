from torch import Tensor
from torch.nn import Module


class SelfAwareAttention(Module):
    def __init__(self):
        super(SelfAwareAttention, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
