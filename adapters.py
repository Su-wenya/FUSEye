import torch
from torch import nn

class GeometricAdapterV2(nn.Module):
    """Zero-initialized adapter with 3x3 depthwise conv for fisheye spatial modeling.
    When zero-initialized, acts as perfect identity: forward(x) = x."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        inner = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, inner, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner, affine=False)
        self.act = nn.SiLU()
        self.dw_conv = nn.Conv2d(inner, inner, 3, padding=1, groups=inner, bias=False)
        self.bn2 = nn.BatchNorm2d(inner, affine=False)
        self.conv2 = nn.Conv2d(inner, channels, 1, bias=False)
        # Zero-initialize only the output projection: A(x)=0 at init,
        # while conv2 receives non-zero input so gradients can flow.
        nn.init.zeros_(self.conv2.weight)
    def forward(self, x):
        return x + self.conv2(self.act(self.bn2(self.dw_conv(self.act(self.bn1(self.conv1(x)))))))


class AdapterWrapper(nn.Module):
    def __init__(self, original, adapter):
        super().__init__()
        self.original = original
        self.adapter = adapter
        self.f = getattr(original, "f", -1)
        self.i = getattr(original, "i", None)
        self.np = getattr(original, "np", 1)
    def forward(self, x):
        return self.adapter(self.original(x))
