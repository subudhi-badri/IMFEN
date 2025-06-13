import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import RRDB, ConvReLUBlock, ConvBlock
from .attention import SAM

class MFEN(nn.Module):
    def __init__(
        self,
        en_feature_num,
        en_inter_num,
        de_feature_num,
        de_inter_num
    ):
        super(MFEN, self).__init__()

        # Encoder
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, en_feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )

        self.rdb1 = RRDB(in_channel=en_feature_num, d_list=(1, 2, 1), inter_num=en_inter_num)
        self.sam_block1 = SAM(in_channel=en_feature_num, d_list=(1, 2, 3, 2, 1), inter_num=en_inter_num)

        self.down1 = nn.Sequential(
            nn.Conv2d(en_feature_num, 2 * en_feature_num, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.rdb2 = RRDB(in_channel=2 * en_feature_num, d_list=(1, 2, 1), inter_num=en_inter_num)
        self.sam_block2 = SAM(in_channel=2 * en_feature_num, d_list=(1, 2, 3, 2, 1), inter_num=en_inter_num)

        self.down2 = nn.Sequential(
            nn.Conv2d(2 * en_feature_num, 4 * en_feature_num, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.rdb3 = RRDB(in_channel=4 * en_feature_num, d_list=(1, 2, 1), inter_num=en_inter_num)
        self.sam_block3 = SAM(in_channel=4 * en_feature_num, d_list=(1, 2, 3, 2, 1), inter_num=en_inter_num)

        # Decoder
        # Level 3
        self.preconv_3 = ConvReLUBlock(4 * en_feature_num, de_feature_num, 3, padding=1)
        self.d_rdb3 = RRDB(de_feature_num, (1, 2, 1), de_inter_num)
        self.d_sam_block3 = SAM(in_channel=de_feature_num, d_list=(1, 2, 3, 2, 1), inter_num=de_inter_num)
        self.conv_3 = ConvBlock(in_channel=de_feature_num, out_channel=12, kernel_size=3, padding=1)

        # Level 2
        self.preconv_2 = ConvReLUBlock(2 * en_feature_num + de_feature_num, de_feature_num, 3, padding=1)
        self.d_rdb2 = RRDB(de_feature_num, (1, 2, 1), de_inter_num)
        self.d_sam_block2 = SAM(in_channel=de_feature_num, d_list=(1, 2, 3, 2, 1), inter_num=de_inter_num)
        self.conv_2 = ConvBlock(in_channel=de_feature_num, out_channel=12, kernel_size=3, padding=1)

        # Level 1
        self.preconv_1 = ConvReLUBlock(en_feature_num + de_feature_num, de_feature_num, 3, padding=1)
        self.d_rdb1 = RRDB(de_feature_num, (1, 2, 1), de_inter_num)
        self.d_sam_block1 = SAM(in_channel=de_feature_num, d_list=(1, 2, 3, 2, 1), inter_num=de_inter_num)
        self.conv_1 = ConvBlock(in_channel=de_feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        x_1 = self.rdb1(x)
        x_1 = self.sam_block1(x_1)
        down_1 = self.down1(x_1)

        x_2 = self.rdb2(down_1)
        x_2 = self.sam_block2(x_2)
        down_2 = self.down2(x_2)

        x_3 = self.rdb3(down_2)
        x_3 = self.sam_block3(x_3)

        y_3 = self.preconv_3(x_3)
        y_3 = self.d_rdb3(y_3)
        y_3 = self.d_sam_block3(y_3)
        out_3 = self.conv_3(y_3)
        out_3 = F.pixel_shuffle(out_3, 2)

        y_3 = F.interpolate(y_3, scale_factor=2, mode='bilinear')

        y_2 = torch.cat([x_2, y_3], dim=1)
        y_2 = self.preconv_2(y_2)
        y_2 = self.d_rdb2(y_2)
        y_2 = self.d_sam_block2(y_2)
        out_2 = self.conv_2(y_2)
        out_2 = F.pixel_shuffle(out_2, 2)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')

        y_1 = torch.cat([x_1, y_2], dim=1)
        y_1 = self.preconv_1(y_1)
        y_1 = self.d_rdb1(y_1)
        y_1 = self.d_sam_block1(y_1)
        out_1 = self.conv_1(y_1)
        out_1 = F.pixel_shuffle(out_1, 2)

        return out_1, out_2, out_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02) 