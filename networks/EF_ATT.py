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

class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.

        Input:
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1 / 3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(2 * i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def forward(self, x):
        G = x

        # Level 1
        L0 = Rearrange('b d h w -> b d (h w)')(G)
        L0_att = F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        L0_att = F.softmax(L0_att, dim=-1)  # X1

        # Next Levels
        attention_maps = [L0_att]
        pyramid = [G]

        for kernel in self.sigma_kernels:
            G = F.conv2d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels)
            pyramid.append(G)

        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            L = Rearrange('b d h w -> b d (h w)')(L)
            L_att = F.softmax(L, dim=2) @ L.transpose(1, 2)
            attention_maps.append(L_att)

        return sum(attention_maps)


class EfficientFrequencyAttention(nn.Module):  # EF-ATT
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

    def __init__(self, in_channels, key_channels, value_channels, pyramid_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

        # Build a laplacian pyramid
        self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels)

        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False, groups=in_channels)

    def forward(self, x):
        n, _, h, w = x.size()  # b d h w

        # Efficient Attention
        keys = F.softmax(self.keys(x).reshape((n, self.key_channels, h * w)), dim=2)
        queries = F.softmax(self.queries(x).reshape(n, self.key_channels, h * w), dim=1)
        values = self.values(x).reshape((n, self.value_channels, h * w))
        context = keys @ values.transpose(1, 2)  # dk*dv
        attended_value = (context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w)  # n*dv
        eff_attention = self.reprojection(attended_value)

        # Freqency Attention
        freq_context = self.freq_attention(x)  #
        freq_attention = (freq_context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w)



        attention = torch.cat([eff_attention[:, :, None, ...], freq_attention[:, :, None, ...]],
                              dim=2)
        attention = self.conv_dw(attention)[:, :, 0, ...]  # b d h w

        return attention


class FrequencyTransformerBlock(nn.Module):
    """
        Input:
            x : [b, (H*W), d], H, W

        Output:
            mx : [b, (H*W), d]
    """

    def __init__(self, in_dim, key_dim, value_dim, pyramid_levels=3, token_mlp='mix'):
        super().__init__()

        self.in_dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientFrequencyAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim,
                                                pyramid_levels=pyramid_levels)

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