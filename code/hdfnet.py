import torch
import torch.nn as nn

from BaseBlocks import BasicConv2d
import torch.nn.functional as F
from ResNet import Backbone_ResNet50_in1, Backbone_ResNet50_in3, Backbone_WResNet50_in1, Backbone_WResNet50_in3, Backbone_ResNet101_in3, Backbone_ResNet101_in1
# from backbone.VGG import (
#     Backbone_VGG19_in1,
#     Backbone_VGG19_in3,
#     Backbone_VGG_in1,
#     Backbone_VGG_in3,
# )
from MyModules import (
    DDPM,
    DenseTransLayer,
)

def cus_sample(feat, **kwargs):
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=True)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y

class HDFNet_VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_VGG16, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.depth_trans16 = DenseTransLayer(512, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)
        self.depth_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d, in_data_1_d

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)

        in_data_4_aux = self.depth_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_4_d, in_data_8_d, in_data_16_d

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        del out_data_4, in_data_4_aux, in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        del out_data_2, in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()


class HDFNet_VGG19(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_VGG19, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG19_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG19_in1(pretrained=pretrained)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.depth_trans16 = DenseTransLayer(512, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)
        self.depth_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d, in_data_1_d

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)

        in_data_4_aux = self.depth_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_4_d, in_data_8_d, in_data_16_d

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        del out_data_4, in_data_4_aux, in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        del out_data_2, in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()


class HDFNet_Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_Res50, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_ResNet50_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32,
        ) = Backbone_ResNet50_in1(pretrained=pretrained)

        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)

        self.depth_trans32 = DenseTransLayer(2048, 64)
        self.depth_trans16 = DenseTransLayer(1024, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 3, 1)
        self.act = nn.ReLU()

    def forward(self, in_data, in_depth):
        in_data_2 = self.encoder2(in_data)
        # del in_data
        in_data_2_d = self.depth_encoder2(in_depth)
        del in_depth
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        del in_data_4_d
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)
        in_data_32 = self.encoder32(in_data_16)
        in_data_32_d = self.depth_encoder32(in_data_16_d)

        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        del in_data_8_d
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_16_d
        in_data_32_aux = self.depth_trans32(in_data_32, in_data_32_d)
        del in_data_32_d

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)  # 1024
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1
        return self.act(in_data+out_data)


class HDFNet_WRes50(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_WRes50, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_WResNet50_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32,
        ) = Backbone_WResNet50_in1(pretrained=pretrained)

        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)

        self.depth_trans32 = DenseTransLayer(2048, 64)
        self.depth_trans16 = DenseTransLayer(1024, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 3, 1)
        self.act = nn.ReLU()

    def forward(self, in_data, in_depth):
        in_data_2 = self.encoder2(in_data)
        del in_data
        in_data_2_d = self.depth_encoder2(in_depth)
        del in_depth
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        del in_data_4_d
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)
        in_data_32 = self.encoder32(in_data_16)
        in_data_32_d = self.depth_encoder32(in_data_16_d)

        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        del in_data_8_d
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_16_d
        in_data_32_aux = self.depth_trans32(in_data_32, in_data_32_d)
        del in_data_32_d

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)  # 1024
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1
        return self.act(out_data)


class HDFNet_Res101(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_Res101, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_ResNet101_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32,
        ) = Backbone_ResNet101_in1(pretrained=pretrained)

        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)

        self.depth_trans32 = DenseTransLayer(2048, 64)
        self.depth_trans16 = DenseTransLayer(1024, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 3, 1)
        self.act = nn.ReLU()

    def forward(self, in_data, in_depth):
        in_data_2 = self.encoder2(in_data)
        in_data
        in_data_2_d = self.depth_encoder2(in_depth)
        del in_depth
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        del in_data_4_d
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)
        in_data_32 = self.encoder32(in_data_16)
        in_data_32_d = self.depth_encoder32(in_data_16_d)

        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        del in_data_8_d
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_16_d
        in_data_32_aux = self.depth_trans32(in_data_32, in_data_32_d)
        del in_data_32_d

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)  # 1024
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1
        return self.act(out_data+in_data)

class HDFNet_Res75(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_Res75, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_ResNet101_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32,
        ) = Backbone_ResNet50_in1(pretrained=pretrained)

        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)

        self.depth_trans32 = DenseTransLayer(2048, 64)
        self.depth_trans16 = DenseTransLayer(1024, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 3, 1)
        self.act = nn.ReLU()

    def forward(self, in_data, in_depth):
        in_data_2 = self.encoder2(in_data)
        del in_data
        in_data_2_d = self.depth_encoder2(in_depth)
        del in_depth
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        del in_data_4_d
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)
        in_data_32 = self.encoder32(in_data_16)
        in_data_32_d = self.depth_encoder32(in_data_16_d)

        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        del in_data_8_d
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_16_d
        in_data_32_aux = self.depth_trans32(in_data_32, in_data_32_d)
        del in_data_32_d

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)  # 1024
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1
        return self.act(out_data)

class HDFNet_Res50res(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_Res50res, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_ResNet50_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32,
        ) = Backbone_ResNet50_in1(pretrained=pretrained)

        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)

        self.depth_trans32 = DenseTransLayer(2048, 64)
        self.depth_trans16 = DenseTransLayer(1024, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 3, 1)
        self.act = nn.ReLU()

    def forward(self, in_data, in_depth):
        in_data_2 = self.encoder2(in_data)
        in_data_2_d = self.depth_encoder2(in_depth)
        del in_depth
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        del in_data_4_d
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)
        in_data_32 = self.encoder32(in_data_16)
        in_data_32_d = self.depth_encoder32(in_data_16_d)

        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        del in_data_8_d
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_16_d
        in_data_32_aux = self.depth_trans32(in_data_32, in_data_32_d)
        del in_data_32_d

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)  # 1024
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1
        return self.act(out_data+in_data)



if __name__ == "__main__":
    import time
    model =  HDFNet_Res50(False)
    model.eval()
    a = torch.rand(1,3,1024,1024)
    b = torch.rand(1,1,1024,1024)
    t0 = time.time()
    for i in range(10):
        c = model(a,b)
    t1 = time.time()
    print((t1-t0)/10)
