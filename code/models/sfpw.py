# SfPwild network (https://github.com/ChenyangLEI/sfp-wild)
# Use fixed view vectors.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

def calc_view_vecs(h, w, focal_length=1505, sensor_size=360):
    x_min, x_max = -sensor_size / 2, sensor_size / 2
    y_min, y_max = -sensor_size / 2, sensor_size / 2

    # Define view vector
    x_coords = torch.linspace(x_min, x_max, w).view(1, 1, w)
    y_coords = torch.linspace(y_min, y_max, h).view(1, h, 1)
    z_coords = torch.full((1, h, w), focal_length, dtype=torch.float32)
    view_vecs = torch.cat([x_coords.expand(1,h,w), y_coords.expand(1,h,w), z_coords], dim=0)

    # Normalize view vectors
    view_vecs = view_vecs / (torch.linalg.norm(view_vecs, dim=0, keepdim=True) + 1e-8)
    return view_vecs


class SfPW_Net(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes=3,
        dim=64,
        residual_num=8,
        bilinear=True,
        norm="in",
        dropout=0.0,
        skip_res=False,
    ):
        super(SfPW_Net, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip_res = skip_res

        self.inc = DoubleConv(in_channels+3, 64) # Add 3 channels for view vectors
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes)
        self.resblock_layers = nn.ModuleList([])

        # Missing positinal encoding
        for i in range(residual_num):
            self.resblock_layers.append(Block(dim * 8, dropout=dropout))

            # self.resblock_layers.append(BasicBlock(512, 512, norm_layer=nn.LayerNorm))


    def forward(self, x):
        # Add view vectors to the input
        b, c, h, w = x.size()
        view_vecs = calc_view_vecs(h, w).to(x.device)
        view_vecs = view_vecs.unsqueeze(0).expand(b, -1, -1, -1)
        x = torch.cat([x, view_vecs], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b, c, h, w = x5.size()
        x5 = torch.reshape(x5, [b, c, h * w]).permute(0, 2, 1)
        # print(x5.size())
        for resblock in self.resblock_layers:
            residual = resblock(x5)
            if self.skip_res:
                # print("residual", residual[0,0,0])
                # import ipdb; ipdb.set_trace()
                x5 = residual
                # print("x5", x5[0,0,0])
            else:
                x5 = x5 + residual
        x5 = torch.reshape(x5.permute(0, 2, 1), [b, c, h, w])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, norm="bn"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == "bn":
            norm_fn = nn.BatchNorm2d
        elif norm == "in":
            norm_fn = nn.InstanceNorm2d
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
            norm_fn(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
            norm_fn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, norm="bn"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, norm=norm),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm="bn"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Borrow from: https://github.com/talsperre/INFER/blob/master/LayerNorm2D.py


class LayerNormConv2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d
    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450
    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)).unsqueeze(-1).unsqueeze(-1)
        self.beta = nn.Parameter(torch.zeros(features)).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError(
                "got {}-feature tensor, expected {}".format(input.size(1), self.features)
            )

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1, -1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(
            x
        )


""" Full assembly of the parts to form the complete network """


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        self.heads = heads
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head
        self.all_head_dim = self.heads * self.dim_head

        ## All linear layers (including query, key, and value layers and dense block layers)
        ## preserve the dimensionality of their inputs and are tiled over input index dimensions #
        # (i.e. applied as a 1 × 1 convolution).
        self.query = nn.Linear(query_dim, self.all_head_dim)  # (b n d_q) -> (b n hd)
        self.key = nn.Linear(context_dim, self.all_head_dim)  # (b m d_c) -> (b m hd)
        self.value = nn.Linear(context_dim, self.all_head_dim)  # (b m d_c) -> (b m hd)
        self.out = nn.Linear(self.all_head_dim, query_dim)  # (b n d) -> (b n d)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)
        x = x.view(*new_x_shape)  # (b n hd) -> (b n h d)
        return x.permute(0, 2, 1, 3)  # (b n h d) -> (b h n d)

    def forward(self, query, context=None):
        if context is None:
            context = query
        mixed_query_layer = self.query(query)  # (b n d_q) -> (b n hd)
        mixed_key_layer = self.key(context)  # (b m d_c) -> (b m hd)
        mixed_value_layer = self.value(context)  # (b m d_c) -> (b m hd)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (b h n d)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (b h m d)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (b h m d)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # (b h n m)
        attention_scores = attention_scores / math.sqrt(self.dim_head)  # (b h n m)
        attention_probs = self.softmax(attention_scores)  # (b h n m)
        attention_probs = self.attn_dropout(attention_probs)  # (b h n m)

        context_layer = torch.matmul(
            attention_probs, value_layer
        )  # (b h n m) , (b h m d) -> (b h n d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (b h n d)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_size, droppath=0.0, dropout=0.0):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, dropout=dropout)
        # self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(self.attention_norm(x))) + x
        x = self.drop_path(self.ffn(self.ffn_norm(x))) + x
        return x


