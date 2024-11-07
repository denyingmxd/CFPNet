import copy
import torch
import torch.nn as nn
from .attention import LinearAttention

import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
import math
from .convnext import Block14
Size_ = Tuple[int, int]


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None,my_cross=False,real_bs=1,p1=1,p2=1):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)

        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        if x_mask is not None:
            tmp_mask = torch.ones_like(x_mask)
        else:
            tmp_mask = None
        message = self.attention(query, key, value, q_mask=tmp_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        if x_mask is not None:
            message.masked_fill_(~x_mask[:,:,None,None], 0)
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message + x



class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.ws = ws
        self.encoder_layer = LoFTREncoderLayer(self.dim, num_heads)

    def forward(self, x, size: Size_, y):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.

        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        # print(f'pad {pad_r} {pad_b}')
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = rearrange(x, 'b (sh ws) (sw ws2) c -> (b sh sw) (ws ws2) c', sh=_h, sw=_w)
        # x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3).reshape(B*_h*_w, self.ws*self.ws, C)

        x = self.encoder_layer(x, x)
        if pad_r > 0 or pad_b > 0:
            x = rearrange(x, '(b sh sw) (ws ws2) c -> b (sh ws) (sw ws2) c', sh=_h, sw=_w, ws=self.ws)
            # x = x.view(B, _h, _w, self.ws, self.ws, C).transpose(2, 3).reshape(B, _h*self.ws, _w*self.ws, C)
            x = x[:, :H, :W, :].contiguous()
            x = rearrange(x, 'b h w c -> b (h w) c')
            # x = x.view(B, -1, C)
        else:
            x = rearrange(x, '(b sh sw) (ws ws2) c -> b (sh ws sw ws2) c', sh=_h, sw=_w, ws=self.ws)
        return x


class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.encoder_layer = LoFTREncoderLayer(self.dim, num_heads)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape

        query = x.clone()


        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        query = self.encoder_layer(query, x)
        return query



class TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)

    def forward(self, x, size: Size_,y=None):
        y=None
        x = self.lga(x, size,y)
        x = self.gsa(x, size)

        return x



class LoFTREncoderLayer_newcross9(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross9, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


        self.conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, bias=False,padding=1)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(d_model)

        self.relu = nn.ReLU()

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        old_feat0=feat0
        bs = feat0.size(0)
        zone_area = zone_mask
        inside_features = torch.masked_select(feat0, zone_area)
        outside_features = torch.masked_select(feat0, ~zone_area)

        num_inside = int(zone_mask.sum() / D / B)
        num_outside = int(H * W - num_inside)
        inside_features = inside_features.reshape(B, num_inside, D)
        outside_features = outside_features.reshape(B, num_outside, D)

        query, key, value = outside_features, inside_features, inside_features

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value)  # [N, L, (H, D)]


        tmp = torch.zeros_like(feat0)
        tmp[~zone_area] = message.flatten()

        tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=H)
        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)

        feat0 = torch.cat([feat0, tmp], dim=1)
        feat0 = self.conv1(feat0)
        feat0 = self.bn1(feat0)
        # feat0 = self.relu(feat0)
        feat0 = self.conv2(feat0)
        feat0 = self.bn2(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        feat0 = feat0+old_feat0
        return feat0


class Combine1(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine1, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """

        feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.large_kernel_path(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0
