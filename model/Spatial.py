import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_aspp import ResNet_ASPP


class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)

    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)


class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(
            *[nn.Conv2d(infilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, outfilter, 3, padding=1, bias=False), nn.BatchNorm2d(outfilter),
              nn.ReLU(inplace=True)])

    def forward(self, x):
        return self.layer(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Spatial(nn.Module):
    def __init__(self, input_channel, mode):
        super(Spatial, self).__init__()
        self.rgb_bkbone = ResNet_ASPP(input_channel, 1, 16, 'resnet34')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.squeeze5 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # ------------decoder----------------#
        self.decoder5 = decoder_stage(64, 128, 64)  #
        self.decoder4 = decoder_stage(128, 128, 64)  #
        self.decoder3 = decoder_stage(128, 128, 64)  #
        self.decoder2 = decoder_stage(128, 128, 64)  #
        self.decoder1 = decoder_stage(128, 128, 64)  #

        self.out5 = out_block(64)
        self.out4 = out_block(64)
        self.out3 = out_block(64)
        self.out2 = out_block(64)
        self.out1 = out_block(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if mode == 'pretrain_rgb':
            self.rgb_bkbone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')

    def forward(self, image):
        out4, out1, img_conv1_feat, out2, out3, out5_aspp, course_img = self.rgb_bkbone(
            image)

        out1, out2, out3, out4, out5 = self.squeeze1(out1), \
                                       self.squeeze2(out2), \
                                       self.squeeze3(out3), \
                                       self.squeeze4(out4), \
                                       self.squeeze5(out5_aspp)

        feature5 = self.decoder5(out5)
        feature4 = self.decoder4(torch.cat([feature5, out4], 1))
        B, C, H, W = out3.size()
        feature3 = self.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), out3), 1))
        B, C, H, W = out2.size()
        feature2 = self.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), out2), 1))
        B, C, H, W = out1.size()
        feature1 = self.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), out1), 1))

        decoder_out5_rgb = self.out5(feature5, H * 4, W * 4)
        decoder_out4_rgb = self.out4(feature4, H * 4, W * 4)
        decoder_out3_rgb = self.out3(feature3, H * 4, W * 4)
        decoder_out2_rgb = self.out2(feature2, H * 4, W * 4)
        decoder_out1_rgb = self.out1(feature1, H * 4, W * 4)

        return decoder_out1_rgb, decoder_out2_rgb, decoder_out3_rgb, decoder_out4_rgb, decoder_out5_rgb
