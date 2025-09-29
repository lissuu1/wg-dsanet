import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image, to_tensor
import os
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )
    def forward(self, x):
        return self.conv(x)



class ECAM_64(nn.Module):
    def __init__(self):
        super(ECAM_64, self).__init__()
        self.down_path = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            DSConv3x3(12, 24, stride=2),
        )
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(24, 24, 1),
            nn.BatchNorm2d(24),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.down_path(x)
        edge = x - self.avg_pool(x)
        weight = self.edge_conv(edge)
        return weight * x + x

class ECAM_16(nn.Module):
    def __init__(self):
        super(ECAM_16, self).__init__()
        self.down_path = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DSConv3x3(16, 32, stride=2),
            DSConv3x3(32, 64, stride=2),
            DSConv3x3(64, 160, stride=2),
        )
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(160, 160, 1),
            nn.BatchNorm2d(160),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down_path(x)
        edge = x - self.avg_pool(x)
        weight = self.edge_conv(edge)
        return weight * x + x