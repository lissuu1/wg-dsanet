import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.deconv import build_deconv
from modeling.backbone import build_backbone
from modeling.ECAM import ECAM_64
from modeling.ECAM import ECAM_16
from modeling.DSAN import DSAN
from modeling.kanconv import KANCONV
from torchvision.transforms.functional import to_pil_image, to_tensor
from modeling.WTcat import DFBIA
from thop import profile
def basic_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
class CDNet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(CDNet, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.deconv = build_deconv(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn
        if backbone == 'resnet':
            self.lowf_conv = basic_block(256 * 2, 256 * 1)
            self.highf_conv = basic_block(2048 * 2, 2048 * 1)
        if backbone == 'xception':
            self.lowf_conv = basic_block(128 * 2, 128 * 1)
            self.highf_conv = basic_block(2048 * 2, 2048 * 1)
        if backbone == 'mobilenet':
            self.lowf_conv = basic_block(24 * 4, 24 * 1)
            self.highf_conv = basic_block(160 * 4, 160)
        self.block0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    )
        self.block1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(32),
                                    nn.ReLU(),
                                    )
        self.block4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(32),
                                    nn.ReLU(),
                                    )
        self.conv = nn.Conv2d(32, num_classes, kernel_size=1, stride=1)
        self.wfusion1 = DFBIA(in_channels=48)
        self.wfusion2 = DFBIA(in_channels=320)
        self.wfusion3 = DFBIA(in_channels=64)
        self.wfusion4 = DFBIA(in_channels=16)
        self.wfusion5 = DFBIA(in_channels=32)
        self.wfusion6 = DFBIA(in_channels=64)
        self.up1 = nn.ConvTranspose2d(160, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.DSAN1 = DSAN(in_ch=3, out_ch=16, stride=2)
        self.DSAN2 = DSAN(in_ch=16, out_ch=32, stride=2)
        self.DSAN3 = DSAN(in_ch=32, out_ch=64, stride=2)
        self.ECAM16=ECAM_16()
        self.ECAM64 = ECAM_64()
    def forward(self, input1, input2):
        FH1, FL1 = self.backbone(input1)
        FH2, FL2 = self.backbone(input2)
        self.feat_FL2 = FL2
        FE1_64=self.ECAM64(input1)
        FE2_64=self.ECAM64(input2)
        FE1_16 = self.ECAM16(input1)
        FE2_16 = self.ECAM16(input2)
        FH1=torch.cat((FH1,FE1_16),dim=1)
        FH2 = torch.cat((FH2, FE2_16),dim=1)
        FL1 = torch.cat((FL1, FE1_64), dim=1)
        FL2 = torch.cat((FL2, FE2_64), dim=1)
        Fdsan1t1=self.DSAN1(input1)
        Fdsan1t2 = self.DSAN1(input2)
        Fdsan1=self.wfusion4(Fdsan1t1,Fdsan1t2)
        Fdsan2t1 = self.DSAN2(Fdsan1t1)
        Fdsan2t2 = self.DSAN2(Fdsan1t2)
        Fdsan2 = self.wfusion5(Fdsan2t1, Fdsan2t2)
        Fdsan3t1 = self.DSAN3(Fdsan2t1)
        Fdsan3t2 = self.DSAN3(Fdsan2t2)
        Fdsan3 = self.wfusion6(Fdsan3t1, Fdsan3t2)
        FLC = self.wfusion1(FL1, FL2)
        FLC1 = self.lowf_conv(FLC)
        FHC = self.wfusion2(FH1, FH2)
        FHC1 = self.highf_conv(FHC)
        FHC2 = self.up1(FHC1)
        FHC2=self.block0(FHC2)
        FLC1=self.block1(FLC1)
        FMC=self.wfusion3(FLC1,FHC2)
        FD1 = self.deconv(FMC)
        FD1=FD1+Fdsan3
        F1 = self.up3(FD1)
        F2 = self.block2(F1)
        F2=F2+Fdsan2
        F3 = self.up4(F2)
        F4 = self.block3(F3)
        F4=F4+Fdsan1
        self.feat_F4 = F4
        F5 = self.up5(F4)
        F6 = self.block4(F5)
        F = self.conv(F6)
        return F
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
    def get_10x_lr_params(self):
        modules = [self.deconv, self.lowf_conv, self.highf_conv, self.up1,self.up2,self.up3,self.up4, self.up5,self.block0,self.block1, self.block2,self.block3,self.block4,
                   self.DSAN1,self.DSAN2,self.DSAN3,self.ECAM16,self.ECAM64,self.wfusion1,self.wfusion2,self.wfusion3,self.wfusion4,self.wfusion5,self.wfusion6,
                   self.conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
