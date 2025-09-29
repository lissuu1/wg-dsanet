import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=(k - 1) // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MultiScalePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.pool5 = nn.AvgPool2d(5, stride=1, padding=2)
        self.conv_fuse = Conv(in_ch * 2, out_ch, k=1)

    def forward(self, x):
        x3 = self.pool3(x)
        x5 = self.pool5(x)
        return self.conv_fuse(torch.cat([x3, x5], dim=1))


class DSAN(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, stride=4):
        super().__init__()
        self.out_ch = out_ch
        self.stride = stride
        self.softmax = nn.Softmax(dim=-1)

        att_stride = [2, 2] if stride == 4 else [2, 1] if stride == 2 else [1, 1]

        self.attention = nn.Sequential(
            MultiScalePool(in_ch, out_ch),
            Conv(out_ch, out_ch * 2, k=3, s=att_stride[0]),
            nn.GELU(),
            Conv(out_ch * 2, out_ch, k=3, s=att_stride[1])
        )

        self.ds_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3,
                      stride=2 if stride >= 2 else 1,
                      padding=1, groups=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=1),
            nn.BatchNorm2d(in_ch * 4),
            nn.ReLU(),
            nn.Conv2d(in_ch * 4, in_ch * 4, kernel_size=3,
                      stride=2 if stride >= 4 else 1,
                      padding=1, groups=1, bias=False),
            nn.BatchNorm2d(in_ch * 4),
            nn.Conv2d(in_ch * 4, out_ch * 4, kernel_size=1),
            nn.BatchNorm2d(out_ch * 4),
            nn.ReLU()
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch * 4)
        )

        self.group_channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(out_ch, max(out_ch // 4, 4), 1),
            nn.ReLU(),
            Conv(max(out_ch // 4, 4), out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.attention(x)

        B, C, H, W = att.shape
        s1, s2 = 2, 2
        h, w = H // s1, W // s2

        att = rearrange(att, 'b c (h s1) (w s2) -> b c h w (s1 s2)', s1=s1, s2=s2)
        att = self.softmax(att)

        x_ds = self.ds_conv(x)
        x_res = self.residual(x)
        x_ds = x_ds + x_res

        x_grouped = rearrange(x_ds, 'b (s c) h w -> b s c h w', s=4)

        B, S, C, H, W = x_grouped.shape
        group_merged = x_grouped.view(B * S, C, H, W)
        channel_weights = self.group_channel_att(group_merged)
        channel_weights = channel_weights.view(B, S, C, 1, 1)
        x_grouped = x_grouped * channel_weights

        x_ds = rearrange(x_grouped, 'b s c h w -> b (s c) h w')

        x = rearrange(x_ds, 'b (s c) h w -> b c h w s', s=4)

        att = rearrange(att, 'b c h w s -> b (c s) h w')
        att = F.interpolate(att, size=(H, W), mode='bilinear', align_corners=False)
        att = rearrange(att, 'b (c s) h w -> b c h w s', c=self.out_ch, s=4)

        x = torch.sum(x * att, dim=-1)
        return x