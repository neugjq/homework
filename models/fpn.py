import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return self.conv(x)

class BiFPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align = nn.ModuleList([
            Conv1x1(c, out_channels) if c != out_channels else nn.Identity()
            for c in in_channels
        ])
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    def forward(self, features):
        # features: [p5, p4, p3]
        p5, p4, p3 = [align(f) for align, f in zip(self.align, features)]
        w1 = F.relu(self.w1); w1 = w1 / (w1.sum() + self.eps)
        p4_td = w1[0] * p4 + w1[1] * F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.conv(p4_td)
        w2 = F.relu(self.w2); w2 = w2 / (w2.sum() + self.eps)
        p3_out = w2[0] * p3 + w2[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest') + w2[2] * F.interpolate(p5, size=p3.shape[-2:], mode='nearest')
        p3_out = self.conv(p3_out)
        return [p5, p4_td, p3_out]

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(in_channels if i == 0 else [out_channels]*3, out_channels)
            for i in range(num_layers)
        ])
    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return features 