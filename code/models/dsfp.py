# DeepSfP network (https://github.com/UCLA-VMG/DeepSfP)
# Use AoLP and DoLP maps as prior, instead of normal maps based on diffuse and specular models.
# Modified decoder to handle non-2^N size images.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSfP_Net(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()

        encoder = SfPEncoderBlock
        decoder = SfPDecoderBlock
        head = SfPHeadBlock

        self.enc1 = encoder(in_channels, 32)
        self.enc2 = encoder(32, 64)
        self.enc3 = encoder(64, 128)
        self.enc4 = encoder(128, 256)
        self.enc5 = encoder(256, 512)

        self.dec1 = decoder(512, 256)
        self.dec2 = decoder(256, 128)
        self.dec3 = decoder(128, 64)
        self.dec4 = decoder(64, 32)
        self.dec5 = decoder(32, 24)

        self.head = head(24, 3)
    
    def forward(self, x):
        # 4 polar images
        polar = x[:, :4]

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Now passing skip connections directly to allow proper resolution matching
        out = self.dec1(x5, x4, polar)
        out = self.dec2(out + x4, x3, polar)
        out = self.dec3(out + x3, x2, polar)
        out = self.dec4(out + x2, x1, polar)
        out = self.dec5(out + x1, x, polar)

        out = self.head(out)
        return out

    def init_weights(self, logger=None):
        if logger is not None:
            logger.info('=> initializing model weights...')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()



class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, affine=False):
        super(InstanceNorm2d, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=affine)

    def forward(self, x, *args):  # Ignore polar images
        return self.instance_norm(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.1, inplace=True):
        super(LeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x, *args):  # Ignore polar images
        return self.leaky_relu(x)


class ReLU(nn.Module):
    def __init__(self, inplace=True):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x, *args):
        return self.relu(x)


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, affine=True):
        super(BatchNorm2d, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, affine=affine)

    def forward(self, x, *args):  # Ignore polar images
        return self.batch_norm(x)


class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization block.
      Based on https://arxiv.org/abs/1903.07291
    """
    def __init__(self, inplanes: int, mlp_inplanes: int = 4, nhidden: int = 128,
                kernel_size: int = 3, padding: int = 1, interp_mode: str = 'bilinear',
                activation = ReLU, normalization = BatchNorm2d):
        super().__init__()

        self.interp = partial(F.interpolate, mode=interp_mode)
        self.mlp_conv = nn.Conv2d(mlp_inplanes, nhidden, kernel_size, padding=padding,)
        self.mlp_act = activation()
        self.mlp_gamma = nn.Conv2d(nhidden, inplanes, kernel_size, padding=padding,)
        self.mlp_beta = nn.Conv2d(nhidden, inplanes, kernel_size, padding=padding,)
        self.norm = normalization(inplanes)

    def forward(self, x, polar):
        # Produce scaling and bias conditioned on polar images
        polar = self.interp(polar, size = x.shape[2:])
        polar = self.mlp_conv(polar)
        polar = self.mlp_act(polar)
        gamma = self.mlp_gamma(polar)
        beta = self.mlp_beta(polar)

        out = self.norm(x)  # generate parameter-free normalized activations
        out = (1 + gamma) * out + beta  # apply conditioned scale & bias

        return out


class SfPEncoderBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3, padding: int = 1,
                activation = LeakyReLU, normalization = InstanceNorm2d):
        super().__init__()

        self._norm = normalization(planes)
        self.act = activation()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, padding=padding, stride=2,)
        self.norm1 = self._norm
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, padding=padding,)
        self.norm2 = self._norm

        self.downsample = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.act(out)

        return out

class SfPDecoderBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3, padding: int = 1,
                interp_mode: str = 'bilinear', activation = LeakyReLU,
                normalization = SPADE):
        super().__init__()

        self._norm = normalization(inplanes, activation = ReLU)
        self.act = activation()

        # Replace fixed scale factor upsampling with mode only
        self.interp_mode = interp_mode
        self.bottleneck = nn.Conv2d(inplanes, planes, 1)

        self.norm1 = self._norm
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, padding=padding,)
        self.norm2 = normalization(planes, activation=ReLU)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, padding=padding)

    def forward(self, x, skip, polar):
        # Resize to match skip connection size instead of fixed 2x upsampling
        x = F.interpolate(x, size=skip.shape[2:], mode=self.interp_mode, align_corners=False)

        identity = self.bottleneck(x)

        out = self.norm1(x, polar)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out, polar)
        out = self.act(out)
        out = self.conv2(out)

        out += identity

        return out


class SfPHeadBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3, padding: int = 1,
                 activation = LeakyReLU, normalization = BatchNorm2d):
        super().__init__()

        self._norm = normalization(inplanes)
        self.act = activation()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, padding=padding)
        self.norm1 = self._norm

        self.conv_out = nn.Conv2d(inplanes, planes, kernel_size, padding=padding,)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv_out(out + identity)
        out = F.normalize(out)

        return out


