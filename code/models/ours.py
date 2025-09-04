# Our learning-based method
# Using CNN encoder-decoder and transformer bridge modules.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Ours_Net(nn.Module):
    """
    Args:
        in_channels (int): Number of input channels.
        dim (int, optional): Base dimension of the feature maps. Defaults to 32.
    """
    def __init__(self, in_channels, dim=32):
        super().__init__()
        self.init_conv = ResBlock(in_channels, dim)

        self.encoder0 = EncoderBlock(dim, dim * 2)
        self.encoder1 = EncoderBlock(dim * 2, dim * 4)
        self.encoder2 = EncoderBlock(dim * 4, dim * 8)
        self.encoder3 = EncoderBlock(dim * 8, dim * 8)

        self.bridge = TransformerBridge(dim * 8, dim * 8)

        self.decoder3 = DecoderBlock(dim * 8, dim * 4)
        self.decoder2 = DecoderBlock(dim * 4, dim * 2)
        self.decoder1 = DecoderBlock(dim * 2, dim * 1)
        self.decoder0 = DecoderBlock(dim * 1, dim)  # Output dimension matches init_conv

        self.head = Head(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, height, width).
        """
        # Initial Convolution
        x1 = self.init_conv(x)

        # Encoder Path
        x2 = self.encoder0(x1)
        x3 = self.encoder1(x2)
        x4 = self.encoder2(x3)
        x5 = self.encoder3(x4)

        # SPP Bridge
        x = self.bridge(x5)

        # Decoder Path
        x = self.decoder3(x, x4)
        x = self.decoder2(x, x3)
        x = self.decoder1(x, x2)
        x = self.decoder0(x, x1)

        # Output Head
        x = self.head(x)
        return x


# --- Component Modules ---

   
class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a convolutional layer, 
    group normalization, and a SiLU activation function.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_groups (int, optional): Number of groups for group normalization. Defaults to 8.
        kernel_size (int, optional): Kernel size of the convolutional layer. Defaults to 3.
        padding (int, optional): Padding size of the convolutional layer. Defaults to 1.
        dilation (int, optional): Dilation rate of the convolutional layer. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ResBlock(nn.Module):
    """
    Residual block consisting of two convolutional blocks with a skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        # Skip connection to handle different input/output channels
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        skip = self.skip_connection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip  # Add skip connection
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a max-pooling layer for downsampling
    followed by a convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2)
        self.resblock = ResBlock(in_channels, out_channels)

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.downsample(x)
        x = self.resblock(x)
        return x

class DecoderBlock(nn.Module):
    """
    Decoder block consisting of an upsampling layer followed by a convolutional block.
    It concatenates the upsampled feature map with the skip connection from the corresponding encoder block.
    The skip connection channels must be equal to the input channels of the decoder block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = ResBlock(in_channels * 2, out_channels)

    def forward(self, x, skip):
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): Input tensor from the previous layer.
            skip (torch.Tensor): Skip connection tensor from the corresponding encoder block.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x)
        return x

class TransformerBridge(nn.Module):
    """
    Transformer-based bridge module.
    It incorporates 2D sinusoidal positional encoding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. (Not used, kept for interface consistency)
        num_blocks (int, optional): Number of transformer encoder blocks. Defaults to 4.
        num_heads (int, optional): Number of attention heads in the transformer. Defaults to 4.
        dim_ff_scale (int, optional): Scaling factor for the feed-forward dimension
            within the transformer. Defaults to 4.
        dropout (float, optional): Dropout rate for the transformer. Defaults to 0.1.
    """
    def __init__(self, in_channels, out_channels, num_blocks=4, num_heads=4, dim_ff_scale=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels,
                                                    nhead=num_heads,
                                                    dim_feedforward=in_channels * dim_ff_scale,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        # out_channels is not used.  It is kept to maintain a consistent
        # interface with other bridge types (ConvBridge, SPPBridge).

    def _add_2d_sincos_pos_encoding(self, x):
        """Adds 2D sinusoidal positional encoding to the input tensor."""
        _, d, h, w = x.shape
        device = x.device

        # Create normalized 2D grid coordinates (from 0 to 1)
        y_coords = torch.linspace(0, 1, h, device=device).unsqueeze(1).repeat(1, w)  # Shape: (h, w)
        x_coords = torch.linspace(0, 1, w, device=device).unsqueeze(0).repeat(h, 1)  # Shape: (h, w)

        # Add batch and channel dimensions: (h, w) -> (1, 1, h, w)
        y_coords = y_coords.unsqueeze(0).unsqueeze(0)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0)

        # Ensure the channel dimension is divisible by 4
        assert d % 4 == 0, "Channel dimension (d) must be divisible by 4 for positional encoding"
        d_quarter = d // 4

        # Calculate the frequency scaling factor
        div_term = torch.exp(torch.arange(0, d_quarter, dtype=torch.float32, device=device) *
                            -(math.log(10000.0) / d_quarter))
        div_term = div_term.view(1, d_quarter, 1, 1)  # Reshape to (1, d_quarter, 1, 1)

        # Compute sine and cosine positional encodings
        pos_y_sin = torch.sin(y_coords * div_term)
        pos_y_cos = torch.cos(y_coords * div_term)
        pos_x_sin = torch.sin(x_coords * div_term)
        pos_x_cos = torch.cos(x_coords * div_term)

        # Concatenate the positional encodings
        pos_encoding = torch.cat([pos_y_sin, pos_y_cos, pos_x_sin, pos_x_cos], dim=1)

        return x + pos_encoding

    def forward(self, x):
        """
        Forward pass of the Transformer bridge.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        b, d, h, w = x.shape
        x = self._add_2d_sincos_pos_encoding(x)
        x = x.view(b, d, h * w)  # Reshape to (b, d, h*w)
        x = x.permute(0, 2, 1)  # Permute to (b, h*w, d) for batch-first transformer
        x = self.transformer_encoder(x)  # Apply transformer encoder
        x = x.permute(0, 2, 1)  # Permute back to (b, d, h*w)
        x = x.view(b, d, h, w)  # Reshape back to (b, d, h, w)
        return x

class Head(nn.Module):
    """
    Output head of the UNet.  Applies a 1x1 convolution to map the feature maps
    to the desired number of output channels (3 in this case) and normalizes the output.

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization along the channel dimension
        return x


