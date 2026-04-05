"""Minimal RRDBNet architecture for Real-ESRGAN .pth weight loading.

This is a standalone implementation of the RRDBNet (Residual-in-Residual
Dense Block Network) used by Real-ESRGAN. It allows loading custom .pth
weight files without requiring the full basicsr or realesrgan packages.

Reference: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative
Adversarial Networks", ECCV 2018 Workshops.
"""

import torch
from torch import nn
from torch.nn import functional as F


def pixel_unshuffle(x, scale):
    """Reverse of PixelShuffle: rearrange spatial pixels into channels.

    [B, C, H, W] → [B, C*scale*scale, H//scale, W//scale]
    """
    b, c, h, w = x.shape
    assert h % scale == 0 and w % scale == 0
    h_new, w_new = h // scale, w // scale
    x = x.view(b, c, h_new, scale, w_new, scale)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h_new, w_new)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 convolutions."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3 RDBs)."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for image super-resolution.

    Args:
        in_nc: Number of input channels (default: 3).
        out_nc: Number of output channels (default: 3).
        num_feat: Number of intermediate features (default: 64).
        num_block: Number of RRDB blocks (default: 23).
        num_grow_ch: Growth channels in each dense block (default: 32).
        scale: Upscale factor, must be power of 2 (default: 4).
    """

    def __init__(self, in_nc=3, out_nc=3, num_feat=64, num_block=23,
                 num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale

        # For scale=2, the input is pixel-unshuffled from [B,C,H,W] to
        # [B,C*4,H/2,W/2] before conv_first, matching basicsr convention.
        # For scale=1 (denoise), pixel_unshuffle by 4.
        if scale == 2:
            in_nc = in_nc * 4
        elif scale == 1:
            in_nc = in_nc * 16

        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        self.body = nn.ModuleList(
            [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # basicsr RRDBNet always has exactly conv_up1 + conv_up2 (hardcoded).
        # Scale difference is handled by pixel_unshuffle at input, not by
        # the number of upsample layers.
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(
            feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(
            feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
