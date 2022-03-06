# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .Res2Net_v1b import res2net50_v1b_26w_4s
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from backbone import build_backbone


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out

# class xception_mseg(nn.Module):
#     def __init__(self,backbone='xception',output_stride=16,num_classes=1, sync_bn=True, freeze_bn=False):
#         super(xception_mseg, self).__init__()
#         if backbone == 'drn':
#             output_stride = 8
#
#         if sync_bn == True:
#             BatchNorm = SynchronizedBatchNorm2d
#         else:
#             BatchNorm = nn.BatchNorm2d
#
#         self.backbone = build_backbone(backbone, output_stride, BatchNorm)
#         # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
#         # self.decoder = build_decoder(num_classes, backbone, BatchNorm)
#         # self.rfb2_1 = RFB_modified(128, 32)
#         self.rfb3_1 = RFB_modified(128, 32)
#         self.rfb4_1 = RFB_modified(2048, 32)
#         # ---- Partial Decoder ----
#         # self.agg1 = aggregation(channel)
#         self.agg1 = aggregation1(32)
#         self.freeze_bn = freeze_bn
#
#     def forward(self, input):
#         x, low_level_feat = self.backbone(input)
#         # x = self.aspp(x)
#         # x = self.decoder(x, low_level_feat)
#
#         # x2_rfb = self.rfb2_1(low_level_feat2)  # channel -> 32
#         x3_rfb = self.rfb3_1(low_level_feat)  # channel -> 32
#         x4_rfb = self.rfb4_1(x)  # channel -> 32
#         # print(low_level_feat2.shape)
#         # print(low_level_feat1.shape)
#         # print(x.shape)
#         # # print(x2_rfb.shape)
#         # print(x3_rfb.shape)
#         # print(x4_rfb.shape)
#         out = self.agg1(x4_rfb, x3_rfb)
#         # print("ll")
#         x = F.interpolate(out, scale_factor=4,
#                                       mode='bilinear')  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#         # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
#
#         return x
#
#
# class aggregation1(nn.Module):
#     # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
#     # used after MSF
#     def __init__(self, channel):
#         super(aggregation1, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#
#         self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
#         self.conv5 = nn.Conv2d(2 * channel, 1, 1)
#
#     def forward(self, x1, x2):
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
#         # x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
#         #        * self.conv_upsample3(self.upsample(x2)) * x3
#
#         x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x = self.conv_concat2(x2_2)
#
#         # x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         # x3_2 = self.conv_concat3(x3_2)
#
#         # x = self.conv4(x3_2)
#         x = self.conv5(x)
#
#         return x
#
# class RFB_modified(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(RFB_modified, self).__init__()
#         self.relu = nn.ReLU(True)
#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
#         self.conv_res = BasicConv2d(in_channel, out_channel, 1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#
#         x = self.relu(x_cat + self.conv_res(x))
#         return x
# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x

# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x
#
#
# class RFB_modified(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(RFB_modified, self).__init__()
#         self.relu = nn.ReLU(True)
#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
#         self.conv_res = BasicConv2d(in_channel, out_channel, 1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#
#         x = self.relu(x_cat + self.conv_res(x))
#         return x
#
#
# class aggregation(nn.Module):
#     # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
#     # used after MSF
#     def __init__(self, channel):
#         super(aggregation, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
#
#         self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
#         self.conv5 = nn.Conv2d(3*channel, 1, 1)
#
#     def forward(self, x1, x2, x3):
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
#         x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
#                * self.conv_upsample3(self.upsample(x2)) * x3
#
#         x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_2 = self.conv_concat2(x2_2)
#
#         x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         x3_2 = self.conv_concat3(x3_2)
#
#         x = self.conv4(x3_2)
#         x = self.conv5(x)
#
#         return x
#
#
# class PraNet(nn.Module):
#     # res2net based encoder decoder
#     def __init__(self, channel=32):
#         super(PraNet, self).__init__()
#         # ---- ResNet Backbone ----
#         self.resnet = res2net50_v1b_26w_4s(pretrained=True)
#         # ---- Receptive Field Block like module ----
#         self.rfb2_1 = RFB_modified(512, channel)
#         self.rfb3_1 = RFB_modified(1024, channel)
#         self.rfb4_1 = RFB_modified(2048, channel)
#         # ---- Partial Decoder ----
#         self.agg1 = aggregation(channel)
#         # ---- reverse attention branch 4 ----
#         self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
#         self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
#         self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
#         self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
#         self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
#         # ---- reverse attention branch 3 ----
#         self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
#         self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
#         # ---- reverse attention branch 2 ----
#         self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
#         self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)      # bs, 64, 88, 88
#         # ---- low-level features ----
#         x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
#         x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
#
#         x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
#         x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
#         x2_rfb = self.rfb2_1(x2)        # channel -> 32
#         x3_rfb = self.rfb3_1(x3)        # channel -> 32
#         x4_rfb = self.rfb4_1(x4)        # channel -> 32
#
#         ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
#         lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#
#         # ---- reverse attention branch_4 ----
#         crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
#         x = -1*(torch.sigmoid(crop_4)) + 1
#         x = x.expand(-1, 2048, -1, -1).mul(x4)
#         x = self.ra4_conv1(x)
#         x = F.relu(self.ra4_conv2(x))
#         x = F.relu(self.ra4_conv3(x))
#         x = F.relu(self.ra4_conv4(x))
#         ra4_feat = self.ra4_conv5(x)
#         x = ra4_feat + crop_4
#         lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
#
#         # ---- reverse attention branch_3 ----
#         crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
#         x = -1*(torch.sigmoid(crop_3)) + 1
#         x = x.expand(-1, 1024, -1, -1).mul(x3)
#         x = self.ra3_conv1(x)
#         x = F.relu(self.ra3_conv2(x))
#         x = F.relu(self.ra3_conv3(x))
#         ra3_feat = self.ra3_conv4(x)
#         x = ra3_feat + crop_3
#         lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
#
#         # ---- reverse attention branch_2 ----
#         crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
#         x = -1*(torch.sigmoid(crop_2)) + 1
#         x = x.expand(-1, 512, -1, -1).mul(x2)
#         x = self.ra2_conv1(x)
#         x = F.relu(self.ra2_conv2(x))
#         x = F.relu(self.ra2_conv3(x))
#         ra2_feat = self.ra2_conv4(x)
#         x = ra2_feat + crop_2
#         lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#
#         return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == '__main__':
    ras = ResUnetPlusPlus(3).cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)