import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            dilation=dilation_rate
        )

    def forward(self, x):
        return self.conv(x)

class ConvReLUBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                dilation=dilation_rate
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RRDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RRDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = ConvReLUBlock(
                in_channel=c,
                out_channel=inter_num,
                kernel_size=3,
                dilation_rate=d_list[i],
                padding=d_list[i]
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = ConvBlock(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t + x

class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = ConvReLUBlock(
                in_channel=c,
                out_channel=inter_num,
                kernel_size=3,
                dilation_rate=d_list[i],
                padding=d_list[i]
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = ConvBlock(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t 