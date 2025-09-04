# Kondo et al.'s network (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640222-supp.pdf)
# Implementation based on the description in the paper.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x, target_size=None):
        x = self.upconv(x)
        
        if target_size is not None:
            current_size = x.size()[2:]
            target_h, target_w = target_size
            
            # padding to match the target size
            diff_h = target_h - current_size[0]
            diff_w = target_w - current_size[1]
            if diff_h > 0 or diff_w > 0:
                pad_left = 0
                pad_right = diff_w
                pad_top = 0
                pad_bottom = diff_h
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
                
        return x

class Kondo_Net(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(Kondo_Net, self).__init__()

        # Encoder
        self.enc_conv1_1 = ConvBlock(in_channels, 64)
        self.enc_conv1_2 = ConvBlock(64, 64)
        self.enc_pool1 = DownBlock(64, 64)

        self.enc_conv2_1 = ConvBlock(64, 128)
        self.enc_conv2_2 = ConvBlock(128, 128)
        self.enc_pool2 = DownBlock(128, 128)        

        self.enc_conv3_1 = ConvBlock(128, 256)
        self.enc_conv3_2 = ConvBlock(256, 256)
        self.enc_conv3_3 = ConvBlock(256, 256)
        self.enc_pool3 = DownBlock(256, 256)
        
        self.enc_conv4_1 = ConvBlock(256,512)
        self.enc_conv4_2 = ConvBlock(512,512)
        self.enc_conv4_3 = ConvBlock(512,512)
        self.enc_pool4 = DownBlock(512, 512)

        # Bottleneck
        self.bottleneck_conv1 = ConvBlock(512, 512)
        self.bottleneck_conv2 = ConvBlock(512, 256)
        self.bottleneck_conv3 = ConvBlock(256, 256)
        self.bottleneck_conv4 = ConvBlock(256, 256)

        # Decoder
        self.dec_upconv1 = UpBlock(256, 256)
        self.dec_conv1_1 = ConvBlock(256, 128)
        self.dec_conv1_2 = ConvBlock(128, 128)
        self.dec_conv1_3 = ConvBlock(128, 128)

        self.dec_upconv2 = UpBlock(128, 128)
        self.dec_conv2_1 = ConvBlock(128, 64)
        self.dec_conv2_2 = ConvBlock(64, 64)

        self.dec_upconv3 = UpBlock(64, 64)
        self.dec_conv3_1 = ConvBlock(64, 64)
        self.dec_conv3_2 = ConvBlock(64, 64)

        self.dec_upconv4 = UpBlock(64, 64)
        self.dec_conv4_1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.enc_conv1_1(x)
        x = self.enc_conv1_2(x)
        x = self.enc_pool1(x)
        skip1 = x 

        x = self.enc_conv2_1(x)
        x = self.enc_conv2_2(x)
        x = self.enc_pool2(x)
        skip2 = x

        x = self.enc_conv3_1(x)
        x = self.enc_conv3_2(x)
        x = self.enc_conv3_3(x)
        x = self.enc_pool3(x)
        skip3 = x

        x = self.enc_conv4_1(x)
        x = self.enc_conv4_2(x)
        x = self.enc_conv4_3(x)
        x = self.enc_pool4(x)

        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_conv3(x)
        x = self.bottleneck_conv4(x)

        # Decoder
        x = self.dec_upconv1(x, skip3.size()[2:])
        x = self.dec_conv1_1(x + skip3)
        x = self.dec_conv1_2(x)
        x = self.dec_conv1_3(x)

        x = self.dec_upconv2(x, skip2.size()[2:])
        x = self.dec_conv2_1(x + skip2)
        x = self.dec_conv2_2(x)

        x = self.dec_upconv3(x, skip1.size()[2:])
        x = self.dec_conv3_1(x + skip1)
        x = self.dec_conv3_2(x)

        x = self.dec_upconv4(x, input_size)
        x = self.dec_conv4_1(x)

        return x