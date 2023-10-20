import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Spatial import Spatial
from model.Flow import Flow
from model.Depth import Depth


# from model.resnet_aspp import ResNet_ASPP


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)

    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)


# U-Net
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


class CIM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CIM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


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


class UIM(nn.Module):
    def __init__(self, in_planes):
        super(UIM, self).__init__()
        self.catconv1 = BasicConv2d(in_planes=in_planes * 2, out_planes=in_planes, kernel_size=3,
                                    padding=1, stride=1)
        self.catconv2 = BasicConv2d(in_planes=in_planes * 4, out_planes=in_planes, kernel_size=3,
                                    padding=1, stride=1)

    def forward(self, input1, input2):
        cat_out = self.catconv1(torch.cat([input1, input2], dim=1))
        mul_out = input1 * input2
        sub_out = input1 - input2
        max_put = torch.maximum(input1, input2)
        interactive_out = self.catconv2(torch.cat([cat_out, mul_out, sub_out, max_put], dim=1))

        return interactive_out


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=8):
#         super(ChannelAttention, self).__init__()
#         # 最大池化
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = max_out
#         return self.sigmoid(out)

# CoA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # out = identity * a_w * a_h
        out = a_w * a_h
        return out


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)


# 先CA后SA
class RFMV(nn.Module):
    def __init__(self):
        super(RFMV, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.ca_rgb = CoordinateAttention(64, 64)
        self.ca_depth = CoordinateAttention(64, 64)
        self.ca_flow = CoordinateAttention(64, 64)

        self.sa_rgb = SpatialAttention()
        self.sa_depth = SpatialAttention()
        self.sa_flow = SpatialAttention()

        self.cauim_rd = UIM(64)
        self.cauim_rf = UIM(64)
        self.sauim_rd = UIM(1)
        self.sauim_rf = UIM(1)

        self.catconv1 = BasicConv2d(2 * 64, 64, kernel_size=3, stride=1, padding=1)
        self.catconv2 = BasicConv2d(2 * 64, 64, kernel_size=3, stride=1, padding=1)

        self.fuse_auxiliary = UIM(64)
        self.fuse = UIM(64)

    def forward(self, rgb, depth, flow):
        rgb_aligned = F.relu(self.bn1(self.conv1(rgb)), inplace=True)
        depth_aligned = F.relu(self.bn2(self.conv2(depth)), inplace=True)
        flow_aligned = F.relu(self.bn3(self.conv3(flow)), inplace=True)

        # 通道注意力
        caatt_rgb = self.ca_rgb(rgb_aligned)
        caatt_depth = self.ca_depth(depth_aligned)
        caatt_flow = self.ca_flow(flow_aligned)

        # 两个模态通道注意力矩阵交叉
        uimca_rd = self.cauim_rd(caatt_rgb, caatt_depth)
        uimca_rf = self.cauim_rf(caatt_rgb, caatt_flow)

        depth_afterca = depth_aligned * uimca_rd + depth_aligned * caatt_depth + depth_aligned
        flow_afterca = flow_aligned * uimca_rf + flow_aligned * caatt_flow + flow_aligned
        rgb_afterca_rd = rgb_aligned * uimca_rd + rgb_aligned * caatt_rgb
        rgb_afterca_rf = rgb_aligned * uimca_rf + rgb_aligned * caatt_rgb
        rgb_afterca = self.catconv1(torch.cat([rgb_afterca_rd, rgb_afterca_rf], dim=1)) + rgb_aligned

        # 空间注意力
        saatt_rgb = self.sa_rgb(rgb_afterca)
        saatt_depth = self.sa_depth(depth_afterca)
        saatt_flow = self.sa_flow(flow_afterca)

        # 两个模态空间注意力矩阵交叉
        uimsa_rd = self.sauim_rd(saatt_rgb, saatt_depth)
        uimsa_rf = self.sauim_rf(saatt_rgb, saatt_flow)

        depth_aftersa = depth_afterca * uimsa_rd + depth_afterca * saatt_depth
        flow_aftersa = flow_afterca * uimsa_rf + flow_afterca * saatt_flow
        rgb_aftersa_rd = rgb_afterca * uimsa_rd + rgb_afterca * saatt_rgb
        rgb_aftersa_rf = rgb_afterca * uimsa_rf + rgb_afterca * saatt_rgb
        rgb_aftersa = self.catconv2(torch.cat([rgb_aftersa_rd, rgb_aftersa_rf], dim=1))

        depth_out = depth_aftersa + depth_aligned
        flow_out = flow_aftersa + flow_aligned
        rgb_out = rgb_aftersa + rgb_aligned

        # 渐进式融合
        auxiliary = self.fuse_auxiliary(depth_out, flow_out)
        fusion = self.fuse(rgb_out, auxiliary)

        return fusion



class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv2d
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.catconv = BasicConv2d(in_planes=self.in_channels * 2, out_planes=self.in_channels, kernel_size=3,
                                   padding=1, stride=1)

        self.main_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.auxiliary_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, main_fea, auxiliary_fea):
        mainNonLocal_fea = self.main_bnRelu(main_fea)
        auxiliaryNonLocal_fea = self.auxiliary_bnRelu(auxiliary_fea)

        batch_size = mainNonLocal_fea.size(0)

        g_x = self.R_g(mainNonLocal_fea).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        l_x = self.F_g(auxiliaryNonLocal_fea).view(batch_size, self.inter_channels, -1)
        l_x = l_x.permute(0, 2, 1)

        catNonLocal_fea = self.catconv(torch.cat([mainNonLocal_fea, auxiliaryNonLocal_fea], dim=1))

        theta_x = self.F_theta(catNonLocal_fea).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.F_phi(catNonLocal_fea).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # add self_f and mutual f
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *mainNonLocal_fea.size()[2:])
        W_y = self.R_W(y)
        z = W_y + main_fea

        m = torch.matmul(f_div_C, l_x)
        m = m.permute(0, 2, 1).contiguous()
        m = m.view(batch_size, self.inter_channels, *auxiliaryNonLocal_fea.size()[2:])
        W_m = self.F_W(m)
        p = W_m + auxiliary_fea

        return z, p


class MAM(nn.Module):
    def __init__(self, inchannels):
        super(MAM, self).__init__()
        self.Nonlocal_RGB_Flow = NonLocalBlock(inchannels)
        self.Nonlocal_RGB_Depth = NonLocalBlock(inchannels)
        self.catconv = BasicConv2d(2 * inchannels, inchannels, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, flow, depth):
        rgb_f, flow = self.Nonlocal_RGB_Flow(rgb, flow)
        rgb_d, depth = self.Nonlocal_RGB_Depth(rgb, depth)
        rgb_final = self.catconv(torch.cat([rgb_f, rgb_d], dim=1))
        return rgb_final, flow, depth


class Model(nn.Module):
    def __init__(self, inchannels, mode, spatial_ckpt=None, flow_ckpt=None, depth_ckpt=None):
        # def __init__(self, inchannels, mode, model_path):
        super(Model, self).__init__()
        self.spatial_net = Spatial(inchannels, mode)
        self.flow_net = Flow(inchannels, mode)
        self.depth_net = Depth(inchannels, mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if spatial_ckpt is not None:
            self.spatial_net.load_state_dict(torch.load(spatial_ckpt, map_location='cpu'))
            print("Successfully load spatial:{}".format(spatial_ckpt))
        if flow_ckpt is not None:
            self.flow_net.load_state_dict(torch.load(flow_ckpt, map_location='cpu'))
            print("Successfully load flow:{}".format(flow_ckpt))
        if depth_ckpt is not None:
            self.depth_net.load_state_dict(torch.load(depth_ckpt, map_location='cpu'))
            print("Successfully load depth:{}".format(depth_ckpt))
        # if mode == 'train':
        #     self.ImageBone.backbone_features._load_pretrained_model(model_path)
        #     self.FlowBone.backbone_features._load_pretrained_model(model_path)
        #     self.DepthBone.backbone_features._load_pretrained_model(model_path)
        #     print("Successfully load")

        self.rfm1 = RFMV()
        self.rfm2 = RFMV()
        self.rfm3 = RFMV()
        self.rfm4 = RFMV()
        self.rfm5 = RFMV()

        self.mam3 = MAM(64)
        self.mam4 = MAM(64)
        self.mam5 = MAM(64)

        self.fuse_last_auxiliary = UIM(1)
        self.fuse_last = UIM(1)

        self.cim5 = CIM(64, 64)
        self.cim4 = CIM(64, 64)
        self.cim3 = CIM(64, 64)
        self.cim2 = CIM(64, 64)
        self.cim1 = CIM(64, 64)

        # Components of PTM module
        self.inplanes = 32
        self.deconv1 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.inplanes = 16
        self.deconv2 = self._make_transpose(TransBasicBlock, 16, 3, stride=2)
        self.agant1 = self._make_agant_layer(64, 32)
        self.agant2 = self._make_agant_layer(32, 16)
        self.outconv2 = nn.Conv2d(16 * 1, 1, kernel_size=1, stride=1, bias=True)

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, image, flow, depth):
        out4, out1, img_conv1_feat, out2, out3, out5_aspp, out_last = self.spatial_net.rgb_bkbone(image)
        coarse_rgb = F.interpolate(out_last, image.shape[2:], mode='bilinear', align_corners=True)
        # out4, out1, img_conv1_feat, out2, out3, out5_aspp, course_img = self.ImageBone(image)

        flow_out4, flow_out1, flow_conv1_feat, flow_out2, flow_out3, flow_out5_aspp, flow_out_last = self.flow_net.flow_bkbone(
            flow)
        coarse_flow = F.interpolate(flow_out_last, flow.shape[2:], mode='bilinear', align_corners=True)
        # flow_out4, flow_out1, flow_conv1_feat, flow_out2, flow_out3, flow_out5_aspp, course_flo = self.FlowBone(flow)

        depth_out4, depth_out1, depth_conv1_feat, depth_out2, depth_out3, depth_out5_aspp, depth_out_last = self.depth_net.depth_bkbone(
            depth)
        coarse_depth = F.interpolate(depth_out_last, depth.shape[2:], mode='bilinear', align_corners=True)
        # depth_out4, depth_out1, depth_conv1_feat, depth_out2, depth_out3, depth_out5_aspp, course_dep = self.DepthBone(
        #     depth)

        out1, out2, out3, out4, out5 = self.spatial_net.squeeze1(out1), \
                                       self.spatial_net.squeeze2(out2), \
                                       self.spatial_net.squeeze3(out3), \
                                       self.spatial_net.squeeze4(out4), \
                                       self.spatial_net.squeeze5(out5_aspp)

        flow_out1, flow_out2, flow_out3, flow_out4, flow_out5 = self.flow_net.squeeze1_flow(flow_out1), \
                                                                self.flow_net.squeeze2_flow(flow_out2), \
                                                                self.flow_net.squeeze3_flow(flow_out3), \
                                                                self.flow_net.squeeze4_flow(flow_out4), \
                                                                self.flow_net.squeeze5_flow(flow_out5_aspp)

        depth_out1, depth_out2, depth_out3, depth_out4, depth_out5 = self.depth_net.squeeze1_depth(depth_out1), \
                                                                     self.depth_net.squeeze2_depth(depth_out2), \
                                                                     self.depth_net.squeeze3_depth(depth_out3), \
                                                                     self.depth_net.squeeze4_depth(depth_out4), \
                                                                     self.depth_net.squeeze5_depth(depth_out5_aspp)

        sc_auxiliary = self.fuse_last_auxiliary(flow_out_last, depth_out_last)
        sc_feature = self.fuse_last(sc_auxiliary, out_last)

        out3, flow_out3, depth_out3 = self.mam3(out3, flow_out3, depth_out3)
        out4, flow_out4, depth_out4 = self.mam4(out4, flow_out4, depth_out4)
        out5, flow_out5, depth_out5 = self.mam5(out5, flow_out5, depth_out5)

        # Sc细化
        B, C, H, W = out3.size()
        sc_feature3 = F.interpolate(sc_feature, (H, W), mode='bilinear', align_corners=True)
        out3 = torch.mul(sc_feature3, out3) + out3
        flow_out3 = torch.mul(sc_feature3, flow_out3) + flow_out3
        depth_out3 = torch.mul(sc_feature3, depth_out3) + depth_out3
        B, C, H, W = out2.size()
        sc_feature2 = F.interpolate(sc_feature, (H, W), mode='bilinear', align_corners=True)
        out2 = torch.mul(sc_feature2, out2) + out2
        flow_out2 = torch.mul(sc_feature2, flow_out2) + flow_out2
        depth_out2 = torch.mul(sc_feature2, depth_out2) + depth_out2
        B, C, H, W = out1.size()
        sc_feature1 = F.interpolate(sc_feature, (H, W), mode='bilinear', align_corners=True)
        out1 = torch.mul(sc_feature1, out1) + out1
        flow_out1 = torch.mul(sc_feature1, flow_out1) + flow_out1
        depth_out1 = torch.mul(sc_feature1, depth_out1) + depth_out1

        fusion1 = self.rfm1(out1, depth_out1, flow_out1)
        fusion2 = self.rfm2(out2, depth_out2, flow_out2)
        fusion3 = self.rfm3(out3, depth_out3, flow_out3)
        fusion4 = self.rfm4(out4, depth_out4, flow_out4)
        fusion5 = self.rfm5(out5, depth_out5, flow_out5)


        feature5 = self.spatial_net.decoder5(fusion5)
        feature5 = self.cim5(feature5)
        feature4 = self.spatial_net.decoder4(torch.cat([feature5, fusion4], 1))
        feature4 = self.cim4(feature4)
        B, C, H, W = fusion3.size()
        # sc_feature3 = F.interpolate(sc_feature, (H, W), mode='bilinear', align_corners=True)
        # fusion3_ = torch.mul(sc_feature3, fusion3) + fusion3
        feature3 = self.spatial_net.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), fusion3), 1))
        feature3 = self.cim3(feature3)
        B, C, H, W = fusion2.size()
        # sc_feature2 = F.interpolate(sc_feature, (H, W), mode='bilinear', align_corners=True)
        # fusion2_ = torch.mul(sc_feature2, fusion2) + fusion2
        feature2 = self.spatial_net.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), fusion2), 1))
        feature2 = self.cim2(feature2)
        B, C, H, W = fusion1.size()
        # sc_feature1 = F.interpolate(sc_feature, (H, W), mode='bilinear', align_corners=True)
        # fusion1_ = torch.mul(sc_feature1, fusion1) + fusion1
        feature1 = self.spatial_net.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), fusion1), 1))
        feature1 = self.cim1(feature1)

        decoder_out5 = self.spatial_net.out5(feature5, H * 4, W * 4)
        decoder_out4 = self.spatial_net.out4(feature4, H * 4, W * 4)
        decoder_out3 = self.spatial_net.out3(feature3, H * 4, W * 4)
        decoder_out2 = self.spatial_net.out2(feature2, H * 4, W * 4)
        sc_out = F.interpolate(sc_feature, (H * 4, W * 4), mode='bilinear', align_corners=True)
        # PTM module
        y = self.agant1(feature1)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        decoder_out1 = self.outconv2(y)

        return decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5, sc_out

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
