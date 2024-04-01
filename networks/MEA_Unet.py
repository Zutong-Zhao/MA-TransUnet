import torch
import torch.nn as nn
# from networks.utils import *
from networks.segformer import *
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
import cv2
import math
from einops import rearrange
from networks.MEA_ATT4 import MEATransformerBlock


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers,layer,token_mlp='mix_skip'):
        super().__init__()

        patch_specs = [
            (7, 4, 3),
            (3, 2, 1),
            (3, 2, 1),
            (3, 2, 1)
        ]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(patch_specs)):
            patch_size, stride, padding = patch_specs[i]
            in_channels = in_dim[i - 1] if i > 0 else 3  # Input channels for the first patch_embed
            out_channels = in_dim[i]

            # Patch Embedding
            patch_embed = OverlapPatchEmbeddings(image_size // (2 ** i), patch_size, stride, padding,
                                                 in_channels, out_channels)
            self.patch_embeds.append(patch_embed)

            # Transformer Blocks
            transformer_block = nn.ModuleList([
                MEATransformerBlock(out_channels, key_dim[i], value_dim[i],layer[i],token_mlp)
                for _ in range(layers[i])
            ])
            self.blocks.append(transformer_block)

            # Layer Normalization
            norm = nn.LayerNorm(out_channels)
            self.norms.append(norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(len(self.patch_embeds)):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            outs.append(x)


        return outs


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale == 2 else nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim if dim_scale == 4 else dim//dim_scale
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        if self.dim_scale == 2:
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1 w p2) c', p1=2, p2=2, c=C//4)
        else:
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1 w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))

        x = self.norm(x.clone())

        return x
#

class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, token_mlp_mode,layer, n_class=9,norm_layer=nn.LayerNorm, is_last=False, is_first=False):
        """
        Custom decoder layer for a neural network.

        Args:
            input_size (int): The input resolution size.
            in_out_chan (tuple): Tuple containing input, output, key, and value channel dimensions.
            token_mlp_mode: Mode for the token-level MLPs in the transformer blocks.
            n_class (int): Number of output classes (for the last layer).
            norm_layer: Normalization layer (e.g., nn.LayerNorm).
            is_last (bool): Indicates if this is the last layer.
        """
        super().__init__()
        
        dims, out_dim, key_dim, value_dim = in_out_chan
        
        self.concat_linear = None if is_first else nn.Linear(dims * (4 if is_last else 2), out_dim)
        self.expansion_layer = PatchExpand(input_resolution=input_size, dim=out_dim, 
                                           dim_scale=2 if not is_last else 4, norm_layer=norm_layer)
        self.last_layer = nn.Conv2d(out_dim, n_class, 1) if is_last else None
        self.layer_former = nn.ModuleList([MEATransformerBlock(out_dim, key_dim, value_dim,layer,token_mlp_mode) for _ in range(2)])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layers = [cat_linear_x]
            for layer in self.layer_former:
                tran_layers.append(layer(tran_layers[-1], h, w))
    
            if self.last_layer:
                return self.last_layer(self.expansion_layer(tran_layers[-1]).view(b, 4*h, 4*w, -1).permute(0, 3, 1, 2))
            else:
                return self.expansion_layer(tran_layers[-1])
        else:
            return self.expansion_layer(x1)



class myFormer(nn.Module):
    def __init__(self, num_classes=9, n_skip_bridge=1,token_mlp_mode="mix_skip"):
        super().__init__()
    
        self.n_skip_bridge = n_skip_bridge
        
        # Encoder configurations
        params = [[96, 192, 384, 768],  # dims
                  [96, 192, 384, 768],  # key_dim
                  [96, 192, 384, 768],  # value_dim
                  [2, 2, 2, 2],  # layers
                  [1, 2, 3, 4]]  # layer
        
        self.encoder = Encoder(image_size=224, in_dim=params[0], key_dim=params[1], value_dim=params[2],
                               layers=params[3], layer=params[4],token_mlp=token_mlp_mode)

        # Decoder configurations
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224

        in_out_chan = [[48, 96, 96, 96],  # [dim, out_dim, key_dim, value_dim]
                       [192, 192, 192, 192],
                       [384, 384, 384, 384],
                       [768, 768, 768, 768]]

        self.decoders = nn.ModuleList()
        de_layer = [4, 3, 2, 1]
        for i in range(4):
            in_dim = d_base_feat_size * 2**i
            decoder = MyDecoderLayer((in_dim, in_dim), in_out_chan[3-i], token_mlp_mode,layer=de_layer[i],
                                     n_class=num_classes, is_last=(i==3), is_first=(i==0))
            self.decoders.append(decoder)
        
    def forward(self, x):
        # Encoder
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.encoder(x)
        # Decoder
        output_enc = [y.permute(0, 2, 3, 1) for y in output_enc]
        b, _, _, c = output_enc[3].shape

        out = self.decoders[0](output_enc[3].view(b,-1,c))        
        out = self.decoders[1](out, output_enc[2])
        out = self.decoders[2](out, output_enc[1])
        out = self.decoders[3](out, output_enc[0])
                
        return out