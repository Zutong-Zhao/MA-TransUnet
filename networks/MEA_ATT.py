import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
import cv2
import math
from einops import rearrange

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DES(nn.Module):
    """
    Diversity-Enhanced Shortcut (DES) based on: "Gu et al.,
    Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation.
    https://github.com/facebookresearch/HRViT
    """

    def __init__(self, in_features, out_features, bias=True, act_func: nn.Module = nn.GELU):
        super().__init__()
        _, self.p = self._decompose(min(in_features, out_features))
        self.k_out = out_features // self.p
        self.k_in = in_features // self.p
        self.proj_right = nn.Linear(self.p, self.p, bias=bias)
        self.act = act_func()
        self.proj_left = nn.Linear(self.k_in, self.k_out, bias=bias)

    def _decompose(self, n):
        assert n % 2 == 0, f"Feature dimension has to be a multiple of 2, but got {n}"
        e = int(math.log2(n))
        e1 = e // 2
        e2 = e - e // 2
        return 2 ** e1, 2 ** e2

    def forward(self, x):
        B = x.shape[:-1]
        x = x.view(*B, self.k_in, self.p)
        x = self.proj_right(x).transpose(-1, -2)

        if self.act is not None:
            x = self.act(x)

        x = self.proj_left(x).transpose(-1, -2).flatten(-2, -1)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MEAttention(nn.Module):
    """
    args:
        in_channels:    (int) : Embedding Dimension.
        key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
        value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2).
        pyramid_levels  (int) : Number of pyramid levels.
    input:
        x : [B, D, H, W]
    output:
        Efficient Attention : [B, D, H, W]

    """

    def __init__(self, in_channels, key_channels, value_channels,layer):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels)
        self.values = nn.Linear(in_channels, value_channels)

        self.q = nn.Linear(in_channels, in_channels)
        self.act = nn.GELU()
        if layer == 1:  # 第1层
            self.sr1 = nn.Conv2d(in_channels, in_channels, kernel_size=8, stride=8)
            self.norm1 = nn.LayerNorm(in_channels)
            self.sr2 = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4)
            self.norm2 = nn.LayerNorm(in_channels)
        if layer == 2:  # 第2层
            self.sr1 = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4)
            self.norm1 = nn.LayerNorm(in_channels)
            self.sr2 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.norm2 = nn.LayerNorm(in_channels)
        if layer == 3:  # 第3层
            self.sr1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.norm1 = nn.LayerNorm(in_channels)
            self.sr2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
            self.norm2 = nn.LayerNorm(in_channels)
        self.kv1 = nn.Linear(in_channels, in_channels)
        self.kv2 = nn.Linear(in_channels, in_channels)
        self.local_conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1, groups=in_channels // 2)
        self.local_conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1, groups=in_channels // 2)

        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        self.reprojection12 = nn.Conv2d(value_channels, in_channels, 1)

        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False, groups=in_channels)

    def forward(self, x):
        _, _, h, w = x.size()  # b d h w
        #[10, 128, 56, 56]
        x = Rearrange('b d h w -> b d (h w)')(x).permute(0, 2, 1)
        B, N, C = x.shape
        q = F.softmax(self.q(x), dim=2).reshape(B, N, 2, C // 2).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
        x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))

        kv1 = self.kv1(x_1).reshape(B, -1, 2, C // 2).permute(0, 2, 1, 3)
        kv2 = self.kv2(x_2).reshape(B, -1, 2, C // 2).permute(0, 2, 1, 3)

        kv1 = kv1.permute(1, 0, 2, 3)
        kv2 = kv2.permute(1, 0, 2, 3)

        k1, v1 = kv1[0], kv1[1]
        k1 = F.softmax(k1, dim=1)

        k2, v2 = kv2[0], kv2[1]
        k2 = F.softmax(k2, dim=1)

        v1 = v1 + self.local_conv1(v1.transpose(1, 2).view(B,C//2, 7, 7)).view(B, C//2, -1).transpose(-1, -2)
        context1 = k1.transpose(1, 2) @ v1
        attended_value1 = (q[:, :1].permute(0, 2, 1, 3).reshape(B, N, C//2))@context1

        v2 = v2 + self.local_conv2(v2.transpose(1, 2).view(B, C // 2, 14, 14)).view(B, C // 2, -1).transpose(-1, -2)
        context2 = k2.transpose(1, 2) @ v2
        attended_value2 = (q[:, 1:].permute(0, 2, 1, 3).reshape(B, N, C//2))@context2

        eff2_attention = torch.cat([attended_value1, attended_value2], dim=-1).transpose(1, 2)
        eff2_attention = self.reprojection12(eff2_attention.reshape(B, self.value_channels, h, w))
        # torch.Size([10, 128, 56, 56])

        keys = F.softmax(self.keys(x), dim=1)
        values = self.values(x)
        context = keys.transpose(1, 2)  @ values # dk*dv
        attended_value = ((q.permute(0, 2, 1, 3).reshape(B, N, C)) @ context).permute(0, 2, 1).reshape(B, C, h, w)  # n*dv

        eff_attention = self.reprojection(attended_value)

        attention = torch.cat([eff_attention[:, :, None, ...], eff2_attention[:, :, None, ...]],dim=2)
        attention = self.conv_dw(attention)[:, :, 0, ...]  # b d h w

        return attention


class MEATransformerBlock(nn.Module):
    """
        Input:
            x : [b, (H*W), d], H, W

        Output:
            mx : [b, (H*W), d]
    """

    def __init__(self, in_dim, key_dim, value_dim,layer, token_mlp='mix'):
        super().__init__()

        self.in_dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MEAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim,layer=layer)

        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim * 4))

        self.des = DES(in_features=in_dim, out_features=in_dim, bias=True, act_func=nn.GELU)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:

        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)

        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)

        # DES Shortcut
        shortcut = self.des(x.reshape(x.shape[0], self.in_dim, -1).permute(0, 2, 1))

        tx = x + attn + shortcut
        mx = tx + self.mlp(self.norm2(tx), H, W)

        return mx