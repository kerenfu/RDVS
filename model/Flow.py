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


class Flow(nn.Module):
    def __init__(self, input_channel, mode):
        super(Flow, self).__init__()
        self.flow_bkbone = ResNet_ASPP(input_channel, 1, 16, 'resnet34')

        self.squeeze5_flow = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4_flow = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3_flow = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2_flow = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1_flow = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # ------------decoder----------------#
        self.decoder5_flow = decoder_stage(64, 128, 64)  #
        self.decoder4_flow = decoder_stage(128, 128, 64)  #
        self.decoder3_flow = decoder_stage(128, 128, 64)  #
        self.decoder2_flow = decoder_stage(128, 128, 64)  #
        self.decoder1_flow = decoder_stage(128, 128, 64)  #

        self.out5_flow = out_block(64)
        self.out4_flow = out_block(64)
        self.out3_flow = out_block(64)
        self.out2_flow = out_block(64)
        self.out1_flow = out_block(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if mode == 'pretrain_flow':
            self.flow_bkbone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')

    def forward(self, flow):

        flow_out4, flow_out1, flow_conv1_feat, flow_out2, flow_out3, flow_out5_aspp, course_flo = self.flow_bkbone(
            flow)

        flow_out1, flow_out2, flow_out3, flow_out4, flow_out5_aspp = self.squeeze1_flow(flow_out1), \
                                                                     self.squeeze2_flow(flow_out2), \
                                                                     self.squeeze3_flow(flow_out3), \
                                                                     self.squeeze4_flow(flow_out4), \
                                                                     self.squeeze5_flow(flow_out5_aspp)

        feature5 = self.decoder5_flow(flow_out5_aspp)
        feature4 = self.decoder4_flow(torch.cat([feature5, flow_out4], 1))
        B, C, H, W = flow_out3.size()
        feature3 = self.decoder3_flow(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), flow_out3), 1))
        B, C, H, W = flow_out2.size()
        feature2 = self.decoder2_flow(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), flow_out2), 1))
        B, C, H, W = flow_out1.size()
        feature1 = self.decoder1_flow(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), flow_out1), 1))

        decoder_out5_flow = self.out5_flow(feature5, H * 4, W * 4)
        decoder_out4_flow = self.out4_flow(feature4, H * 4, W * 4)
        decoder_out3_flow = self.out3_flow(feature3, H * 4, W * 4)
        decoder_out2_flow = self.out2_flow(feature2, H * 4, W * 4)
        decoder_out1_flow = self.out1_flow(feature1, H * 4, W * 4)

        return decoder_out1_flow, decoder_out2_flow, decoder_out3_flow, decoder_out4_flow, decoder_out5_flow
