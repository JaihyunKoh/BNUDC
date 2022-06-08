import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def color_affine(rgb, trans_mat):
    c, h, w = rgb.shape
    rgb = rgb.permute(1, 2, 0)
    rgb_vec = torch.reshape(rgb, [-1, c, 1])
    trans_mat = trans_mat.permute(1, 2, 0)
    trans_mat = torch.reshape(trans_mat, [-1, c, c])
    restored = torch.bmm(trans_mat, rgb_vec)
    restored = restored.squeeze()
    restored = torch.reshape(restored, [h, w, c])
    restored = restored.permute(2, 0, 1)
    return restored

def unpixel_shuffle(feature, r: int = 1):
    b, c, h, w = feature.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    feature_view = feature.contiguous().view(b, c, out_h, r, out_w, r)
    feature_prime = (
        feature_view.permute(0, 1, 3, 5, 2, 4)
        .contiguous()
        .view(b, out_channel, out_h, out_w))
    return feature_prime

class ShareSepConv(nn.Module):
    "Gated Context Aggregation Network for Image Dehazing and Deraining, WACV, 2019"
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, "kernel size should be odd"
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1 # center = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(
            inc, 1, self.kernel_size, self.kernel_size
        ).contiguous()
        return F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)

class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.pre_conv1 = ShareSepConv(1)
        self.pre_conv2 = ShareSepConv(3)
        self.pre_conv4 = ShareSepConv(7)
        self.pre_conv8 = ShareSepConv(15)

        self.conv1 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=1, dilation=1, bias=False,))
        self.conv2 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=2, dilation=2, groups=1, bias=False))
        self.conv4 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=4, dilation=4, groups=1, bias=False))
        self.conv8 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=8, dilation=8, groups=1, bias=False))
        self.conv = nn.Conv2d(feats * 2, feats, 3, 1, padding=1, bias=False)

    def forward(self, x):
        y1 = F.leaky_relu(self.conv1(self.pre_conv1(x)), 0.2)
        y2 = F.leaky_relu(self.conv2(self.pre_conv2(x)), 0.2)
        y4 = F.leaky_relu(self.conv4(self.pre_conv4(x)), 0.2)
        y8 = F.leaky_relu(self.conv8(self.pre_conv8(x)), 0.2)
        y = torch.cat((y1, y2, y4, y8), dim=1)
        y = self.conv(y) + x
        y = F.leaky_relu(y, 0.2)
        return y

class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.down(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class Encoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats):
        super(Encoder, self).__init__()
        self.encoder_level1 = [SmoothDilatedResidualBlock(n_feat) for _ in range(2)]
        self.encoder_level2 = [SmoothDilatedResidualBlock(n_feat + scale_unetfeats) for _ in range(2)]
        self.encoder_level3 = [SmoothDilatedResidualBlock(n_feat + (scale_unetfeats * 2)) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, scale_feats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [SmoothDilatedResidualBlock(n_feat) for _ in range(2)]
        self.decoder_level2 = [SmoothDilatedResidualBlock(n_feat + scale_feats) for _ in range(2)]
        self.decoder_level3 = [SmoothDilatedResidualBlock(n_feat + (scale_feats * 2)) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = SmoothDilatedResidualBlock(n_feat)
        self.skip_attn2 = SmoothDilatedResidualBlock(n_feat + scale_feats)

        self.up21 = SkipUpSample(n_feat, scale_feats)
        self.up32 = SkipUpSample(n_feat + scale_feats, scale_feats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1

class OriginalResolutionBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, num_blk):
        super(OriginalResolutionBlock, self).__init__()
        modules_body = [SmoothDilatedResidualBlock(n_feat) for _ in range(num_blk)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, num_blk):
        super(ORSNet, self).__init__()

        self.orb1 = OriginalResolutionBlock(n_feat, kernel_size, num_blk)
        self.orb2 = OriginalResolutionBlock(n_feat, kernel_size, num_blk)
        self.orb3 = OriginalResolutionBlock(n_feat, kernel_size, num_blk)

    def forward(self, x):
        x = self.orb1(x)
        x = self.orb2(x)
        x = self.orb3(x)
        return x

class BNUDCnet(nn.Module):
    def __init__(self, n_feat_high=72, n_feat_low=32, scale_feats=8, num_blk=6, kernel_size=3, bias=True):
        super(BNUDCnet, self).__init__()
        self.shallow_feat = nn.Sequential(conv(12, n_feat_high, kernel_size, bias=bias), SmoothDilatedResidualBlock(n_feat_high))
        self.orsnet = ORSNet(n_feat_high, kernel_size, num_blk)
        self.tail = conv(n_feat_high, 24, kernel_size, bias=bias)

        self.shallow_sub = nn.Sequential(conv(3, n_feat_low, kernel_size, bias=bias), SmoothDilatedResidualBlock(n_feat_low))
        self.encoder_sub = Encoder(n_feat_low, scale_feats)
        self.decoder_sub = Decoder(n_feat_low, scale_feats)
        self.tail_sub = conv(n_feat_low, 12, kernel_size, bias=bias)

        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, dilation=1, bias=False, groups=3)
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x):
        x_high = unpixel_shuffle(x, 2)
        x_high = self.shallow_feat(x_high)
        x_high = self.orsnet(x_high)
        x_high = self.tail(x_high)
        x_high = F.pixel_shuffle(x_high, 2)
        alpha, beta = torch.split(x_high, 3, dim=1)
        x_high = x * alpha + beta

        x_low = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=True)
        _, _, h, w = x_low.shape
        S = self.box_filter(x_low.data.new().resize_((1, 3, h, w)).fill_(1.0))
        x_low = self.box_filter(x_low) / S
        x_low = self.shallow_sub(x_low)
        x_low = self.encoder_sub(x_low)
        x_low = self.decoder_sub(x_low)
        x_low = self.tail_sub(x_low)
        x_low = F.interpolate(x_low, scale_factor=4, mode="bilinear", align_corners=True)
        gamma, delta = torch.split(x_low, 9, dim=1)

        x_low_for_test = []
        for i, data in enumerate(zip(x, gamma)):
            x_low_for_test.append(color_affine(data[0], data[1]))
        x_low_for_test = torch.stack(x_low_for_test, dim=0) + delta

        out_final = []
        for i, data in enumerate(zip(x_high, gamma)):
            out_final.append(color_affine(data[0], data[1]))
        out_final = torch.stack(out_final, dim=0) + delta
        return x_high, x_low_for_test, out_final


