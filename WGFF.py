import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class DFBIA(nn.Module):
    def __init__(self, in_channels, wave='haar'):
        super(DFBIA, self).__init__()
        self.in_channels = in_channels
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        se_reduction = 16
        self.se_fc1 = nn.Conv2d(6 * in_channels, (6 * in_channels) // se_reduction, kernel_size=1)
        self.se_fc2 = nn.Conv2d((6 * in_channels) // se_reduction, 6 * in_channels, kernel_size=1)
        mid_channels = max(in_channels // 4, 1)
        self.diff_gate = nn.Sequential(
            nn.Conv2d(3 * in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(8 * in_channels, 2*in_channels, kernel_size=1)

    def wavelet_decompose(self, x):
        ll, yh = self.dwt(x)
        if yh[0].shape[-1] == 3:
            h = yh[0].permute(0, 4, 1, 2, 3)
        else:
            h = yh[0].permute(0, 2, 1, 3, 4)
        h = h.reshape(h.size(0), 3 * self.in_channels, h.size(-2), h.size(-1))
        ll = F.interpolate(ll, scale_factor=2, mode='bilinear')
        h = F.interpolate(h, scale_factor=2, mode='bilinear')
        return ll, h

    def forward(self, x1, x2):
        ll1, hh1 = self.wavelet_decompose(x1)
        ll2, hh2 = self.wavelet_decompose(x2)
        low_cat = torch.cat([ll1, ll2], dim=1)
        high_cat = torch.cat([hh1, hh2], dim=1)
        se = F.adaptive_avg_pool2d(high_cat, 1)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        attn_out = high_cat * se

        diff = torch.abs(hh1 - hh2)
        gate = self.diff_gate(diff)
        fused_high = gate * attn_out + (1 - gate) * high_cat
        fused = torch.cat([low_cat, fused_high], dim=1)
        return self.final_conv(fused)



