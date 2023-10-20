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


class Depth(nn.Module):
    def __init__(self, input_channel, mode):
        super(Depth, self).__init__()
        self.depth_bkbone = ResNet_ASPP(input_channel, 1, 16, 'resnet34')

        self.squeeze5_depth = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4_depth = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3_depth = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2_depth = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1_depth = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # ------------decoder----------------#
        self.decoder5_depth = decoder_stage(64, 128, 64)  #
        self.decoder4_depth = decoder_stage(128, 128, 64)  #
        self.decoder3_depth = decoder_stage(128, 128, 64)  #
        self.decoder2_depth = decoder_stage(128, 128, 64)  #
        self.decoder1_depth = decoder_stage(128, 128, 64)  #

        self.out5_depth = out_block(64)
        self.out4_depth = out_block(64)
        self.out3_depth = out_block(64)
        self.out2_depth = out_block(64)
        self.out1_depth = out_block(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if mode == 'pretrain_depth':
            self.depth_bkbone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')

    def forward(self, depth):

        depth_out4, depth_out1, depth_conv1_feat, depth_out2, depth_out3, depth_out5_aspp, course_dep = self.depth_bkbone(
            depth)

        depth_out1, depth_out2, depth_out3, depth_out4, depth_out5_aspp = self.squeeze1_depth(depth_out1), \
                                                                          self.squeeze2_depth(depth_out2), \
                                                                          self.squeeze3_depth(depth_out3), \
                                                                          self.squeeze4_depth(depth_out4), \
                                                                          self.squeeze5_depth(depth_out5_aspp)

        feature5 = self.decoder5_depth(depth_out5_aspp)
        feature4 = self.decoder4_depth(torch.cat([feature5, depth_out4], 1))
        B, C, H, W = depth_out3.size()
        feature3 = self.decoder3_depth(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), depth_out3), 1))
        B, C, H, W = depth_out2.size()
        feature2 = self.decoder2_depth(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), depth_out2), 1))
        B, C, H, W = depth_out1.size()
        feature1 = self.decoder1_depth(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), depth_out1), 1))

        decoder_out5_depth = self.out5_depth(feature5, H * 4, W * 4)
        decoder_out4_depth = self.out4_depth(feature4, H * 4, W * 4)
        decoder_out3_depth = self.out3_depth(feature3, H * 4, W * 4)
        decoder_out2_depth = self.out2_depth(feature2, H * 4, W * 4)
        decoder_out1_depth = self.out1_depth(feature1, H * 4, W * 4)

        return decoder_out1_depth, decoder_out2_depth, decoder_out3_depth, decoder_out4_depth, decoder_out5_depth
