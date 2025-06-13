from models.blocks import DB
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, n_heads=1, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(dim // 3, dim // 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim // 3, dim // 3, 1, 1, 0)
        self.conv3 = nn.Conv2d(dim // 3, dim // 3, 1, 1, 0)

        self.trans_conv1 = nn.ConvTranspose2d(dim // 3, dim // 3, 3, 1, 1)
        self.trans_conv2 = nn.ConvTranspose2d(dim // 3, dim // 3, 3, 1, 1)
        self.trans_conv3 = nn.ConvTranspose2d(dim // 3, dim // 3, 3, 1, 1)

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        q = self.trans_conv1(self.conv1(x))
        k = self.trans_conv2(self.conv2(x))
        v = self.trans_conv3(self.conv3(x))

        q = q.reshape(q.shape[0], q.shape[1], -1)
        k = k.reshape(k.shape[0], k.shape[1], -1).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], -1)

        dp = (q @ k) * self.scale
        attn = dp.softmax(dim=-1)
        weighted_avg = attn @ v

        weighted_avg = weighted_avg.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        return weighted_avg

class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.attn = Attention(3 * in_chnls)

    def forward(self, x0, x2, x4):
        out = torch.cat([x0, x2, x4], dim=1)
        out = self.attn(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4
        return x

class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        
        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y
        return y 