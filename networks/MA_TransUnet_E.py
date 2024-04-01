import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange
from networks.MEA_ATT import MEATransformerBlock
from torch.nn import functional as F

from networks.segformer import *

class Cross_Attention(nn.Module):
    def __init__(self, key_dim, value_dim, h, w, head_count=1):
        super().__init__()

        self.h = h
        self.w = w

        self.reprojection = nn.Conv2d(value_dim, 2 * value_dim, 1)
        self.norm = nn.LayerNorm(2 * value_dim)

    def forward(self, x1, x2):
        B, N, C = x1.size()  # (Batch, Tokens, Embedding dim)

        att_maps = []

        query = F.softmax(x2,dim=2)
        key = F.softmax(x2,dim=1)
        value = x1

        ## Choose one of the following attention:
        #-------------- channel cross-attention--------------
        pairwise_similarities = query.transpose(1, 2) @ key
        att_map = pairwise_similarities @ value.transpose(1, 2)

        ##-------------- efficient cross-Attention-------------
        # context = key.transpose(1, 2) @ value
        # att_map = (query @ context).transpose(1, 2)

        ##-------------- efficient channel cross-attention--------------
        # context = key @ value.transpose(1, 2)
        # att_map = query.transpose(1, 2) @ context
        att_maps.append(att_map)
        Att_maps = torch.cat(att_maps, dim=1).reshape(B, C, self.h, self.w)
        reprojected_value = self.reprojection(Att_maps).reshape(B, 2 * C, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value

class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)


        attn = self.attn(norm_1, norm_2)

        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx

# Encoder
class Encoder(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layer, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        resnet = eval(f"torchvision.models.resnet50(pretrained=True)")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.p1_ch = nn.Conv2d(256, 128, kernel_size=1)
        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(512, 320, kernel_size=1)
        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(1024, 512, kernel_size=1)

        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]


        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [MEATransformerBlock(in_dim[0], key_dim[0], value_dim[0],layer[0],token_mlp='mix_skip') for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [MEATransformerBlock(in_dim[1], key_dim[1], value_dim[1],layer[1],token_mlp='mix_skip') for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [MEATransformerBlock(in_dim[2], key_dim[2], value_dim[2],layer[2],token_mlp='mix_skip') for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn = x
        for i in range(5):
            cnn = self.resnet_layers[i](cnn)
            # torch.Size([10, 64, 112, 112])
            # torch.Size([10, 64, 112, 112])
            # torch.Size([10, 64, 112, 112])
            # torch.Size([10, 64, 56, 56])
            # torch.Size([10, 256, 56, 56])#𝐇/𝟒 × 𝐖/𝟒, 𝐃
        x_ch = self.p1_ch(cnn)# torch.Size([10, 128, 56, 56])
        x_cnn = Rearrange('b c h w -> b (h w) c')(x_ch)  # torch.Size([10, 3136, 128])

        B = x.shape[0]
        outs = []
        # stage 1
        #print("--------- stage 1-----------")
        x, H, W = self.patch_embed1(x)
        # torch.Size([10, 3136, 128]) 56 56

        for blk in self.block1:
            x = blk(x, H, W)
            #torch.Size([10, 3136, 128])
            #torch.Size([10, 3136, 128])
        x = x + x_cnn
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #torch.Size([10, 128, 56, 56])
        outs.append(x)

        # stage 2
        #print("--------- stage 2-----------")
        cnn2 = self.p2(cnn)  # torch.Size([10, 512, 28, 28])#𝐇/𝟖 × 𝐖/𝟖, 𝟐𝐃
        x2_ch = self.p2_ch(cnn2)  # torch.Size([10, 320, 28, 28])#𝐶𝑜𝑛𝑣 1 × 1-->𝐇/𝟖 × 𝐖/𝟖, 𝟐C
        x2_cnn = Rearrange('b c h w -> b (h w) c')(x2_ch)  # torch.Size([10, 784, 320])
        x, H, W = self.patch_embed2(x)
        #torch.Size([10, 784, 320]) 28 28

        for blk in self.block2:
            x = blk(x, H, W)
            #torch.Size([10, 784, 320])
            #torch.Size([10, 784, 320])
        x = x + x2_cnn
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #torch.Size([10, 320, 28, 28])
        outs.append(x)

        # stage 3
        #print("--------- stage 3-----------")
        cnn3 = self.p3(cnn2) #torch.Size([10, 1024, 14, 14])#𝐇/𝟏𝟔 × 𝐖/𝟏𝟔, 𝟒𝐃
        x3_ch = self.p3_ch(cnn3)  #torch.Size([10, 512, 14, 14])#𝐶𝑜𝑛𝑣 1 × 1-->𝐇/𝟏𝟔 × 𝐖/𝟏𝟔, 𝟒C
        x3_cnn = Rearrange('b c h w -> b (h w) c')(x3_ch)  #torch.Size([10, 196, 512])
        x, H, W = self.patch_embed3(x)
        #torch.Size([10, 196, 512]) 14 14

        for blk in self.block3:
            x = blk(x, H, W)
            # torch.Size([10, 196, 512])
            # torch.Size([10, 196, 512])
        x = x + x3_cnn
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #torch.Size([10, 512, 14, 14])
        outs.append(x)
        #torch.Size([10, 128, 56, 56]) torch.Size([10, 320, 28, 28]) torch.Size([10, 512, 14, 14])

        return outs


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count,layer, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
        self.layer_former_1 = MEATransformerBlock(out_dim, key_dim, value_dim,layer, token_mlp='mix_skip')
        self.layer_former_2 = MEATransformerBlock(out_dim, key_dim, value_dim,layer, token_mlp='mix_skip')

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)

            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))

            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)#PatchExpand
        else:
            out = self.layer_up(x1)#PatchExpand
        return out


class myFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers, layer = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2], [1, 2, 3]]
        self.backbone = Encoder(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
            layer=layer
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            layer[2],
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            layer[1],
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            layer[0],
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        #torch.Size([10, 1, 224, 224])
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            # torch.Size([10, 3, 224, 224])

        output_enc = self.backbone(x)
        #torch.Size([10, 128, 56, 56]) torch.Size([10, 320, 28, 28]) torch.Size([10, 512, 14, 14])

        b, c, _, _ = output_enc[2].shape

        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        #torch.Size([10, 784, 256])

        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        #torch.Size([10, 3136, 160])

        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))
        #torch.Size([10, 9, 224, 224])

        return tmp_0
