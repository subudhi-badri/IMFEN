import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PSNR(nn.Module):
    def __init__(self, crop_border=4, only_test_y_channel=True, data_range=1.0):
        super(PSNR, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.data_range = data_range

    def forward(self, img1, img2):
        if self.only_test_y_channel:
            img1 = img1[:, 0:1, :, :]
            img2 = img2[:, 0:1, :, :]

        if self.crop_border > 0:
            img1 = img1[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            img2 = img2[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        mse = F.mse_loss(img1, img2)
        psnr = 10. * torch.log10(self.data_range * self.data_range / mse)
        return psnr

class SSIM(nn.Module):
    def __init__(self, crop_border=4, only_test_y_channel=True, data_range=255.0):
        super(SSIM, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.data_range = data_range

    def forward(self, img1, img2):
        if self.only_test_y_channel:
            img1 = img1[:, 0:1, :, :]
            img2 = img2[:, 0:1, :, :]

        if self.crop_border > 0:
            img1 = img1[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            img2 = img2[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        img1 = img1.float()
        img2 = img2.float()

        kernel = torch.ones(1, 1, 11, 11).to(img1.device) / 121
        kernel = kernel.expand(img1.size(1), 1, 11, 11)

        mu1 = F.conv2d(img1, kernel, padding=5, groups=img1.size(1))
        mu2 = F.conv2d(img2, kernel, padding=5, groups=img2.size(1))

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=5, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=5, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=5, groups=img1.size(1)) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() 