import copy
import torch
import torch.nn as nn
from .attention import LinearAttention
from timm.models.layers import DropPath
import matplotlib.pyplot as plt
import time
import tqdm
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
import math
# from .convnext import Block15
from .convnext import Block14
from .convnext import Block26
from .cmt import LocalPerceptionUint, InvertedResidualFeedForward
import random
Size_ = Tuple[int, int]

def vis_pca(x,H,W,k):
    U, S, V = torch.svd(x.detach().cpu()[0].permute(1, 2, 0).reshape(H * W, -1))
    principal_components = V[:, :k]
    projected_data = torch.mm(x.detach().cpu()[0].permute(1, 2, 0).reshape(H * W, -1), principal_components)
    plt.imshow(projected_data.reshape(H, W, k).sum(-1))
    plt.colorbar()
    plt.show()
    return

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttentionPointModule(nn.Module):
    """ this function is used to achieve the channel attention module in CBAM paper"""
    def __init__(self, C, ratio=8): # getting from the CBAM paper, ratio=16
        super(ChannelAttentionPointModule, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels= C // ratio, out_channels=C, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = x.transpose(2, 1)
        out1 = torch.mean(x, dim=-1, keepdim=True)  # b, c, 1
        out1 = self.mlp(out1) # b, c, 1

        out2 = nn.AdaptiveMaxPool1d(1)(x) # b, c, 1
        out2 = self.mlp(out2) # b, c, 1

        out = self.sigmoid(out1 + out2)


        return (out * x).permute(0, 2, 1)



class SpatialAttentionPointModule(nn.Module):
    """ this function is used to achieve the spatial attention module in CBAM paper"""
    def __init__(self):
        super(SpatialAttentionPointModule, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(2, 1)
        out1 = torch.mean(x,dim=1,keepdim=True) #B,1,N

        out2, _ = torch.max(x, dim=1,keepdim=True)#B,1,N

        out = torch.cat([out2, out1], dim=1) #B,2,N

        out = self.conv1(out) #B,1,N
        out = self.bn(out) #B,1,N
        out =self.relu(out) #B,1,N

        out = self.sigmoid(out) #b, c, n
        return (out * x).permute(0, 2, 1)




class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, inplanes):
        super(ASPP, self).__init__()
        dilations = [1,3,5]

        self.aspp1 = _ASPPModule(inplanes, inplanes, 3, padding=dilations[0], dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, inplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3= _ASPPModule(inplanes, inplanes, 3, padding=dilations[2], dilation=dilations[2])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, inplanes, 1, stride=1, bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(inplanes*4, inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x




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

class LoFTREncoderLayer2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer2, self).__init__()

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

        return message

class LoFTREncoderLayer3(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer3, self).__init__()

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

        self.ca1 = ChannelAttention(d_model)
        self.sa1 = SpatialAttention()

        self.ca2 = ChannelAttention(d_model)
        self.sa2 = SpatialAttention()

    def forward(self, x, source, x_mask=None, source_mask=None,my_cross=False,real_bs=1,p1=1,p2=1):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        from ..config import args
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

        zone_num = int(math.sqrt(message.size(0)//real_bs))
        pp = int(math.sqrt(message.size(1)))

        message =  rearrange(message,  '(b zn zn2) (p1 p2) c -> b (zn p1) (zn2 p2) c', zn=zone_num, zn2=zone_num, p1=pp, b=real_bs)
        message = message.permute(0,3,1,2)
        message = self.ca1(message) * message
        message = self.sa1(message) * message
        message = message.permute(0,2,3,1)
        message = rearrange(message, 'b (zn p1) (zn2 p2) c  -> (b zn zn2) (p1 p2) c', zn=zone_num, zn2=zone_num, p1=pp, b=real_bs)

        x =  rearrange(x,  '(b zn zn2) (p1 p2) c -> b (zn p1) (zn2 p2) c',zn=zone_num, zn2=zone_num, p1=pp,  b=real_bs)
        x = x.permute(0,3,1,2)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = x.permute(0,2,3,1)
        x = rearrange(x, 'b (zn p1) (zn2 p2) c  -> (b zn zn2) (p1 p2) c',zn=zone_num, zn2=zone_num, p1=pp, b=real_bs)

        return message + x

class LoFTREncoderLayer5(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer5, self).__init__()

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

        self.ca1 = ChannelAttentionPointModule(d_model)
        self.sa1 = SpatialAttentionPointModule()

        self.ca2 = ChannelAttentionPointModule(d_model)
        self.sa2 = SpatialAttentionPointModule()

    def forward(self, x, source, x_mask=None, source_mask=None,my_cross=False,real_bs=1,p1=1,p2=1):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        from ..config import args
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


        x = self.ca1(x)
        x = self.sa1(x)

        message = self.ca2(message)
        message = self.sa2(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        return message

class LoFTREncoderLayer6(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer6, self).__init__()

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

        self.ca1 = ChannelAttentionPointModule(d_model)
        self.sa1 = SpatialAttentionPointModule()

        self.ca2 = ChannelAttentionPointModule(d_model)
        self.sa2 = SpatialAttentionPointModule()

    def forward(self, x, source, x_mask=None, source_mask=None,my_cross=False,real_bs=1,p1=1,p2=1):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        from ..config import args
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


        x = self.ca1(x)
        x = self.sa1(x)

        message = self.ca2(message)
        message = self.sa2(message)

        return message + x

class LoFTREncoderLayer7(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer7, self).__init__()

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

        self.ca1 = ChannelAttention(d_model)
        self.sa1 = SpatialAttention(3)

        self.ca2 = ChannelAttention(d_model)
        self.sa2 = SpatialAttention(3)

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

        x = rearrange(x, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)
        x = x * self.ca1(x)
        x = x * self.sa1(x)
        x = rearrange(x, ' b c (p1) (p2) -> b (p1 p2) c', p1=p1, p2=p2)

        message = rearrange(message, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)
        message = message * self.ca1(message)
        message = message * self.sa1(message)
        message = rearrange(message, ' b c (p1) (p2) -> b (p1 p2) c', p1=p1, p2=p2)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message

class LoFTREncoderLayer8(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer8, self).__init__()

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

        self.ca1 = ChannelAttention(2*d_model)
        self.sa1 = SpatialAttention(3)


    def forward(self, x, source, x_mask=None, source_mask=None,real_bs=1,p1=1,p2=1):
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

        x = rearrange(x, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)

        message = rearrange(message, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)

        temp = torch.cat([x,message],dim=1)
        temp = temp * self.ca1(temp)
        temp = temp * self.sa1(temp)


        temp = rearrange(temp, ' b cc (p1) (p2) -> b (p1 p2) cc', p1=p1, p2=p2)
        # feed-forward network
        message = self.mlp(temp)
        message = self.norm2(message)

        return message


class LoFTREncoderLayer9(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer9, self).__init__()

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

        self.ca1 = ChannelAttention(d_model)
        self.sa1 = SpatialAttention(1)

        self.ca2 = ChannelAttention(d_model)
        self.sa2 = SpatialAttention(1)

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

        x = rearrange(x, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)
        x = x * self.ca1(x)
        x = x * self.sa1(x)
        x = rearrange(x, ' b c (p1) (p2) -> b (p1 p2) c', p1=p1, p2=p2)

        message = rearrange(message, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)
        message = message * self.ca1(message)
        message = message * self.sa1(message)
        message = rearrange(message, ' b c (p1) (p2) -> b (p1 p2) c', p1=p1, p2=p2)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message

class LoFTREncoderLayer10(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer10, self).__init__()

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

        self.ca1 = ChannelAttention(d_model)

        self.ca2 = ChannelAttention(d_model)

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

        x = rearrange(x, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)
        x = x * self.ca1(x)
        x = rearrange(x, ' b c (p1) (p2) -> b (p1 p2) c', p1=p1, p2=p2)

        message = rearrange(message, 'b (p1 p2) c -> b c (p1) (p2)', p1=p1, p2=p2)
        message = message * self.ca1(message)
        message = rearrange(message, ' b c (p1) (p2) -> b (p1 p2) c', p1=p1, p2=p2)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message

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


class LoFTREncoderLayer_cross1(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_cross1, self).__init__()

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

class LoFTREncoderLayer_cross2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_cross2, self).__init__()

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

        return message

class LoFTREncoderLayer_cross5(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_cross5, self).__init__()

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

        self.mlp_1 = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(True),
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(True)
        )


    def forward(self, x, source, x_mask=None, source_mask=None,my_cross=False,real_bs=1,p1=1,p2=1):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        x = self.mlp_1(x)
        source = self.mlp_2(source)
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

class LoFTREncoderLayer_cross6(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_cross6, self).__init__()

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

        self.q_proj2 = nn.Linear(d_model, d_model, bias=False)
        self.k_proj2 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj2 = nn.Linear(d_model, d_model, bias=False)
        self.attention2 = LinearAttention()
        self.merge2 = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)



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

        x = message + x

        query, key, value = source, x, x

        # multi-head attention
        query = self.q_proj2(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj2(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj2(value).view(bs, -1, self.nhead, self.dim)
        if source_mask is not None:
            tmp_mask = torch.ones_like(source_mask)
        else:
            tmp_mask = None
        message = self.attention(query, key, value, q_mask=tmp_mask, kv_mask=x_mask)  # [N, L, (H, D)]
        if source_mask is not None:
            message.masked_fill_(~source_mask[:, :, None, None], 0)
        message = self.merge2(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm3(message)

        # feed-forward network
        message = self.mlp2(torch.cat([source, message], dim=2))
        message = self.norm4(message)
        source = source+message
        return x,source

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

class My_1_TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)

    def forward(self, x, size: Size_,y):
        x = self.lga(x, size,y)
        x = self.gsa(x, size)

        return x

class My_2_TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.aspp = ASPP(dim)
        self.lga2 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
    def forward(self, x, size: Size_,y):
        x = self.lga(x, size,y)
        B,N,C = x.shape
        x = x.permute(0,2,1).view(B,C,size[0],size[1])
        x = self.aspp(x)
        x = x.view(B,C,-1).permute(0,2,1)
        x = self.lga2(x, size,y)
        x = self.gsa(x, size)

        return x

class My_3_TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga1 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate1 = _ASPPModule(dim, dim, 1, padding=0, dilation=1)
        self.lga2 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate2 = _ASPPModule(dim, dim, 1, padding=0, dilation=3)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
    def forward(self, x, size: Size_,y):
        y=None
        x = self.lga1(x, size, y)
        B,N,C = x.shape
        x = x.permute(0,2,1).view(B,C,size[0],size[1])
        x = self.dilate1(x)
        x = x.view(B,C,-1).permute(0,2,1)

        x = self.lga2(x, size,y)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, size[0], size[1])
        x = self.dilate2(x)
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.gsa(x, size)

        return x

class My_4_TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga1 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate1 = _ASPPModule(dim, dim, 1, padding=0, dilation=1)
        self.lga2 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate2 = _ASPPModule(dim, dim, 1, padding=0, dilation=3)
        self.lga3= LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate3 = _ASPPModule(dim, dim, 1, padding=0, dilation=5)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
    def forward(self, x, size: Size_,y):
        y=None
        x = self.lga1(x, size, y)
        B,N,C = x.shape
        x = x.permute(0,2,1).view(B,C,size[0],size[1])
        x = self.dilate1(x)
        x = x.view(B,C,-1).permute(0,2,1)

        x = self.lga2(x, size,y)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, size[0], size[1])
        x = self.dilate2(x)
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.lga3(x, size,y)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, size[0], size[1])
        x = self.dilate3(x)
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.gsa(x, size)

        return x

class My_5_TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga1 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate1 = _ASPPModule(dim, dim, 1, padding=0, dilation=1)
        self.lga2 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate2 = _ASPPModule(dim, dim, 3, padding=3, dilation=3)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
    def forward(self, x, size: Size_,y):
        y=None
        x = self.lga1(x, size, y)
        B,N,C = x.shape
        x = x.permute(0,2,1).view(B,C,size[0],size[1])
        x = self.dilate1(x)
        x = x.view(B,C,-1).permute(0,2,1)

        x = self.lga2(x, size,y)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, size[0], size[1])
        x = self.dilate2(x)
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.gsa(x, size)

        return x

class My_6_TwinsTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga1 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate1 = _ASPPModule(dim, dim, 1, padding=0, dilation=1)
        self.lga2 = LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate2 = _ASPPModule(dim, dim, 3, padding=3, dilation=3)
        self.lga3= LocallyGroupedAttn(dim=dim, ws=ws)
        self.dilate3 = _ASPPModule(dim, dim, 3, padding=5, dilation=5)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
    def forward(self, x, size: Size_,y):
        y=None
        x = self.lga1(x, size, y)
        B,N,C = x.shape
        x = x.permute(0,2,1).view(B,C,size[0],size[1])
        x = self.dilate1(x)
        x = x.view(B,C,-1).permute(0,2,1)

        x = self.lga2(x, size,y)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, size[0], size[1])
        x = self.dilate2(x)
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.lga3(x, size,y)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, size[0], size[1])
        x = self.dilate3(x)
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.gsa(x, size)

        return x

class Inside_attn1(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)

    def forward(self, x, size: Size_,y=None):
        y=None
        x = self.lga(x, size,y)
        x = self.gsa(x, size)

        return x

class Inside_attn2(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)

    def forward(self, x, size: Size_,y=None):
        y=None
        old_x = x
        x = self.lga(x, size,y)
        x = self.gsa(x, size)

        return x + old_x

class Inside_attn3(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
        self.conv = nn.Conv2d(dim, dim, 3, padding =1,bias=False)

    def forward(self, x, size: Size_,y=None):
        y=None
        old_x = x
        x = self.lga(x, size,y)
        x = self.gsa(x, size)
        B,N,C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h=size[0]).contiguous()
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=size[0]).contiguous()

        return x + old_x

class Inside_attn4(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
        self.conv = nn.Conv2d(dim, dim, 3, padding =1,bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x, size: Size_,y=None):
        y=None
        old_x = x
        x = self.lga(x, size,y)
        x = self.gsa(x, size)
        B,N,C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h=size[0]).contiguous()
        x = self.conv(x)
        x = self.bn(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=size[0]).contiguous()

        return x + old_x

class Inside_attn5(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
        self.conv = nn.Conv2d(dim, dim, 3, padding =1,bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x, size: Size_,y=None):
        y=None
        old_x = x
        x = self.gsa(x, size)
        B,N,C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h=size[0]).contiguous()
        x = self.conv(x)
        x = self.bn(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=size[0]).contiguous()

        return x + old_x

class Inside_attn6(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=ws)
        self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=ws)
        self.conv = nn.Conv2d(dim, dim, 3, padding =1,bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x, size: Size_,y=None):
        y=None
        old_x = x
        x = self.gsa(x, size)
        x = self.lga(x, size,y)
        B,N,C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h=size[0]).contiguous()
        x = self.conv(x)
        x = self.bn(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=size[0]).contiguous()

        return x + old_x


class LoFTREncoderLayer_newcross1(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross1, self).__init__()

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


        self.conv1 = nn.Conv2d(d_model * 2, d_model, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU()

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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
        feat0 = self.relu(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0


class LoFTREncoderLayer_newcross2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross2, self).__init__()

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
        self.relu = nn.ReLU()

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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
        feat0 = self.relu(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0

class LoFTREncoderLayer_newcross3(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross3, self).__init__()

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
        self.relu = nn.ReLU()
        self.ca1 = ChannelAttention(d_model)
        self.sa1 = SpatialAttention()
    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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
        tmp = self.ca1(tmp) * tmp
        tmp = self.sa1(tmp) * tmp
        feat0 = torch.cat([feat0, tmp], dim=1)
        feat0 = self.conv1(feat0)
        feat0 = self.bn1(feat0)
        feat0 = self.relu(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0

class LoFTREncoderLayer_newcross4(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross4, self).__init__()

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
        self.relu = nn.ReLU()

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0

class LoFTREncoderLayer_newcross5(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross5, self).__init__()

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
        self.gelu = nn.GELU()

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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
        feat0 = self.gelu(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0

class LoFTREncoderLayer_newcross6(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross6, self).__init__()

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
        self.relu = nn.ReLU()
        self.ca1 = ChannelAttention(d_model)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(d_model)
        self.sa2 = SpatialAttention()
    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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
        tmp = self.ca1(tmp) * tmp
        tmp = self.sa1(tmp) * tmp
        feat0 = self.ca2(feat0) * feat0
        feat0 = self.sa2(feat0) * feat0

        feat0 = torch.cat([feat0, tmp], dim=1)
        feat0 = self.conv1(feat0)
        feat0 = self.bn1(feat0)
        feat0 = self.relu(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0


class LoFTREncoderLayer_newcross7(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross7, self).__init__()

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
        self.relu = nn.ReLU()

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
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
        tmp[zone_area] = inside_features.flatten()

        tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=H)
        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)

        feat0 = torch.cat([feat0, tmp], dim=1)
        feat0 = self.conv1(feat0)
        feat0 = self.bn1(feat0)
        # feat0 = self.relu(feat0)

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        return feat0

class LoFTREncoderLayer_newcross8(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross8, self).__init__()

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

        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        feat0 = feat0+old_feat0
        return feat0

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

        message,KV = self.attention(query, key, value)  # [N, L, (H, D)]


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

class LoFTREncoderLayer_newcross10(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross10, self).__init__()

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
        self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
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
        feat0 = self.conv3(feat0)
        feat0 = self.bn3(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
        feat0 = feat0+old_feat0
        return feat0

class LoFTREncoderLayer_newcross11(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross11, self).__init__()

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


        self.conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, bias=False,padding=1,groups=d_model)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1,groups=d_model)
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

class LoFTREncoderLayer_newcross12(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross12, self).__init__()

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
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1,groups=d_model)
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

class LoFTREncoderLayer_newcross13(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer_newcross13, self).__init__()

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
        old_feat0 = feat0
        feat0 = self.conv2(feat0)
        feat0 = self.bn2(feat0)
        feat0 = feat0 + old_feat0
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)

        return feat0


class My_Selfattn1(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.lga = LocallyGroupedAttn(dim=dim, ws=2)
        # self.gsa = GlobalSubSampleAttn(dim=dim, sr_ratio=8)

    def forward(self, x, size: Size_,y=None):
        y=None
        x = self.lga(x, size,y)
        # x = self.gsa(x, size)

        return x

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

class Combine11(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine11, self).__init__()

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
        old_feat0 = feat0
        feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.large_kernel_path(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        feat0 = feat0+old_feat0
        return feat0



class Combine2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine2, self).__init__()

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

        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.large_kernel_path(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        feat0 = self.transformer_path(feat0, zone_mask, H, W, B, D)

        return feat0

class Combine3(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine3, self).__init__()

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

        feat1 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat1 = self.large_kernel_path(feat1)
        feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()

        feat2 = self.transformer_path(feat0, zone_mask, H, W, B, D)

        feat0 = feat1+feat2

        return feat0

class Combine4(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine4, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)


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
        feat0 = self.conv1(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

class Combine4bn(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine4bn, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.bn = nn.BatchNorm2d(d_model)

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
        feat0 = self.conv1(feat0)
        feat0 = self.bn(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

class Combine8(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine8, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(d_model)

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
        feat0 = self.conv1(feat0)
        feat0 = self.bn1(feat0)
        feat0 = self.conv2(feat0)
        feat0 = self.bn2(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

class Combine5(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine5, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model , d_model, kernel_size=3, bias=False, padding=1)

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """

        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.large_kernel_path(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        feat0 = self.transformer_path(feat0, zone_mask, H, W, B, D)

        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.conv1(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

class Combine6(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine6, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, bias=False, padding=1)

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """

        feat1 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat1 = self.large_kernel_path(feat1)
        feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()

        feat2 = self.transformer_path(feat0, zone_mask, H, W, B, D)

        feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
        feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = torch.cat([feat1, feat2], dim=1)
        feat0 = self.conv1(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

class Combine7(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine7, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block26(d_model, large_kernel=large_kernel)

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

# class Combine7(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine7, self).__init__()
#
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
#         self.lpu = LocalPerceptionUint(d_model)
#
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.large_kernel_path(feat0)
#         xx = feat0
#         feat0 = self.lpu(feat0)
#         feat0 = feat0 + xx
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0
#
class Combine9(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine9, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.irffn = InvertedResidualFeedForward(d_model, 4)


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
        feat0 = self.irffn(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0


class Combine10(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine10, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.irffn = InvertedResidualFeedForward(d_model, 4)


    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        old_feat0 = feat0
        feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.large_kernel_path(feat0)
        feat0 = self.irffn(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        feat0 = feat0 + old_feat0
        return feat0

class Combine4bn(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine4bn, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(d_model)


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
        feat0 = self.conv1(feat0)
        feat0 = self.bn1(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

# class Combine4drop(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine4drop, self).__init__()
#
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         pp = random.random()
#         if pp > 0.5:
#             feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#
#         pp = random.random()
#         if pp > 0.5:
#             feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#             feat0 = self.large_kernel_path(feat0)
#
#             feat0 = self.conv1(feat0)
#             feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0


# class Combine9(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine9, self).__init__()
#
#         self.transformer_path_1 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_1 = Block14(d_model, large_kernel=large_kernel)
#         self.transformer_path_2 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_2 = Block14(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(2*d_model, d_model, kernel_size=3, bias=False, padding=1)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat1 = self.transformer_path_1(feat0, zone_mask, H, W, B, D)
#         feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat1 = self.large_kernel_path_1(feat1)
#         feat1 = self.conv1(feat1)
#         # feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat2 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.large_kernel_path_2(feat2)
#         feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat2 = self.transformer_path_2(feat2, zone_mask, H, W, B, D)
#
#         feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.conv2(feat2)
#         # feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat0 = torch.cat([feat1, feat2], dim=1)
#         feat0 = self.conv3(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0
#
# class Combine10(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine10, self).__init__()
#
#         self.transformer_path_1 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_1 = Block15(d_model, large_kernel=large_kernel)
#         self.transformer_path_2 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_2 = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(2*d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.bn3 = nn.BatchNorm2d(d_model)
#         self.conv4 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.bn4 = nn.BatchNorm2d(d_model)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat1 = self.transformer_path_1(feat0, zone_mask, H, W, B, D)
#         feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat1 = self.large_kernel_path_1(feat1)
#         feat1 = self.conv1(feat1)
#         # feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat2 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.large_kernel_path_2(feat2)
#         feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat2 = self.transformer_path_2(feat2, zone_mask, H, W, B, D)
#
#         feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.conv2(feat2)
#         # feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat0 = torch.cat([feat1, feat2], dim=1)
#         feat0 = self.conv3(feat0)
#         feat0 = self.bn3(feat0)
#         feat0 = self.conv4(feat0)
#         feat0 = self.bn4(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0
#
# class Combine9bn(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine9bn, self).__init__()
#
#         self.transformer_path_1 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_1 = Block15(d_model, large_kernel=large_kernel)
#         self.transformer_path_2 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_2 = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(2*d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.bn1 = nn.BatchNorm2d(d_model)
#         self.bn2 = nn.BatchNorm2d(d_model)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat1 = self.transformer_path_1(feat0, zone_mask, H, W, B, D)
#         feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat1 = self.large_kernel_path_1(feat1)
#         feat1 = self.conv1(feat1)
#         feat1 = self.bn1(feat1)
#         # feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat2 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.large_kernel_path_2(feat2)
#         feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat2 = self.transformer_path_2(feat2, zone_mask, H, W, B, D)
#
#         feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.conv2(feat2)
#         feat2 = self.bn2(feat2)
#         # feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat0 = torch.cat([feat1, feat2], dim=1)
#         feat0 = self.conv3(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0
#
# class Combine9ln(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine9ln, self).__init__()
#
#         self.transformer_path_1 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_1 = Block15(d_model, large_kernel=large_kernel)
#         self.transformer_path_2 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_2 = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(2*d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat1 = self.transformer_path_1(feat0, zone_mask, H, W, B, D)
#         feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat1 = self.large_kernel_path_1(feat1)
#         feat1 = self.conv1(feat1)
#         feat1 = feat1.permute(0, 2, 3, 1)
#         feat1 = self.ln1(feat1)
#         feat1 = feat1.permute(0, 3, 1, 2)
#         # feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat2 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.large_kernel_path_2(feat2)
#         feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat2 = self.transformer_path_2(feat2, zone_mask, H, W, B, D)
#
#         feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.conv2(feat2)
#         feat2 = feat2.permute(0, 2, 3, 1)
#         feat2 = self.ln2(feat2)
#         feat2 = feat2.permute(0, 3, 1, 2)
#         # feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat0 = torch.cat([feat1, feat2], dim=1)
#         feat0 = self.conv3(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0
#
# class Combine11(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine11, self).__init__()
#
#         self.transformer_path_1 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_1 = Block15(d_model, large_kernel=large_kernel)
#         self.transformer_path_2 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_2 = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat1 = self.transformer_path_1(feat0, zone_mask, H, W, B, D)
#         feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat1 = self.large_kernel_path_1(feat1)
#         feat1 = self.conv1(feat1)
#         # feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat2 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.large_kernel_path_2(feat2)
#         feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat2 = self.transformer_path_2(feat2, zone_mask, H, W, B, D)
#
#         feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.conv2(feat2)
#         # feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat0 = feat2+feat1
#         feat0 = self.conv3(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0
#
# class Combine12(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine12, self).__init__()
#
#         self.transformer_path_1 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_1 = Block15(d_model, large_kernel=large_kernel)
#         self.transformer_path_2 = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path_2 = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.bn3 = nn.BatchNorm2d(d_model)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#
#         feat1 = self.transformer_path_1(feat0, zone_mask, H, W, B, D)
#         feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat1 = self.large_kernel_path_1(feat1)
#         feat1 = self.conv1(feat1)
#         # feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat2 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.large_kernel_path_2(feat2)
#         feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat2 = self.transformer_path_2(feat2, zone_mask, H, W, B, D)
#
#         feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat2 = self.conv2(feat2)
#         # feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         feat0 = feat2+feat1
#         feat0 = self.conv3(feat0)
#         feat0 = self.bn3(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#
#         return feat0

class Combine13(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine13, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        y = feat0

        feat0 = self.large_kernel_path(feat0)
        z = feat0

        feat0 = torch.cat([x, y, z], dim=1)
        feat0 = self.conv2(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0


class Combine14(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine14, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.irffn = InvertedResidualFeedForward(d_model, 4)


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
        xx = self.irffn(feat0)
        feat0 = feat0+xx
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        return feat0


class Combine15(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine15, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.dw1 = nn.Conv2d(d_model, d_model, kernel_size=3, groups=d_model, bias=False, padding=1)
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
        feat0 = self.dw1(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0


class Combine16(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine16, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.dw1 = nn.Conv2d(d_model, d_model, kernel_size=3, groups=d_model, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(d_model)
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
        feat0 = self.dw1(feat0)
        feat0 = self.bn1(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0

class Combine17(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine17, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.dw1 = nn.Conv2d(d_model, d_model, kernel_size=3, groups=d_model, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(d_model)
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
        xx = self.dw1(feat0)
        xx = self.bn1(xx)
        feat0 = feat0+xx
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()

        return feat0


class Combine18(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine18, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.irffn = InvertedResidualFeedForward(d_model, 4)
        self.ln = nn.LayerNorm(d_model)


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
        xx = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        xx = self.ln(xx)
        xx = rearrange(xx,  'b (h w) c -> b c h w', h=H).contiguous()
        xx = self.irffn(xx)
        feat0 = feat0+xx
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        return feat0

class Combine19(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine19, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)
        self.irffn = InvertedResidualFeedForward(d_model, 4)
        self.bn = nn.BatchNorm2d(d_model)


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
        xx = self.bn(feat0)
        xx = self.irffn(xx)
        feat0 = feat0+xx
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        return feat0


class Combine20(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine20, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross13(d_model, nhead)
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

class Combine21(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine21, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross13(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """

        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat0 = self.large_kernel_path(feat0)
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
        feat0 = self.transformer_path(feat0, zone_mask, H, W, B, D)

        return feat0

class Combine22(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 large_kernel):
        super(Combine22, self).__init__()

        self.transformer_path = LoFTREncoderLayer_newcross13(d_model, nhead)
        self.large_kernel_path = Block14(d_model, large_kernel=large_kernel)

    def forward(self, feat0,zone_mask,H,W,B,D):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """

        feat1 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
        feat1 = self.large_kernel_path(feat1)
        feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=H).contiguous()

        feat2 = self.transformer_path(feat0, zone_mask, H, W, B, D)

        feat0 = feat1+feat2

        return feat0


# class Attention_Module(nn.Module):
#     def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
#         super(Attention_Module, self).__init__()
#         in_channel = high_feature_channel + low_feature_channels
#         out_channel = high_feature_channel
#         if output_channel is not None:
#             out_channel = output_channel
#         channel = in_channel
#         self.ca = ChannelAttention(channel)
#         # self.sa = SpatialAttention()
#         # self.cs = CS_Block(channel)
#         self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, prev_feature, cur_features):
#         features = [upsample(high_features)]
#         features += low_features
#         features = torch.cat(features, 1)
#
#         features = self.ca(features)
#         # features = self.sa(features)
#         # features = self.cs(features)
#
#         return self.relu(self.conv_se(features))

# class Combine15(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine15, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(3*d_model, d_model, kernel_size=3, bias=False, padding=1)
# 
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         feat0 = torch.cat([x, y, z], dim=1)
#         feat0 = self.conv2(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0
# 
# class Combine16(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine16, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.ca = ChannelAttention(3*d_model)
# 
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         feat0 = torch.cat([x, y, z], dim=1)
#         feat0 = feat0 * self.ca(feat0)
#         feat0 = self.conv2(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0
# 
# 
# class Combine17(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine17, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.ca = ChannelAttention(3*d_model)
#         self.sa = SpatialAttention()
# 
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         feat0 = torch.cat([x, y, z], dim=1)
#         feat0 = feat0 * self.ca(feat0)
#         feat0 = feat0 * self.sa(feat0)
#         feat0 = self.conv2(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0
# 
# class Combine19(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine19, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.conv3 = nn.Conv2d(2*d_model, d_model, kernel_size=1, bias=False, padding=0)
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = torch.cat([x,y], dim=1)
#         feat0 = self.conv3(feat0)
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         feat0 = torch.cat([x, y, z], dim=1)
#         feat0 = self.conv2(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0
# 
# class Combine20(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine20, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(3*d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv3 = nn.Conv2d(2*d_model, d_model, kernel_size=3, bias=False, padding=1)
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = torch.cat([x,y], dim=1)
#         feat0 = self.conv3(feat0)
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         feat0 = torch.cat([x, y, z], dim=1)
#         feat0 = self.conv2(feat0)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0

# def get_inside_outside_feature(feat0,zone_mask,H,W,B,D):
#     if len(feat0.size()) ==4:
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#     if len(zone_mask.size()) == 4:
#        zone_mask =  rearrange(zone_mask, 'b (h w) c -> b c h w', h=H).contiguous()
#     bs = feat0.size(0)
#     zone_area = zone_mask
#     inside_features = torch.masked_select(feat0, zone_area)
#     outside_features = torch.masked_select(feat0, ~zone_area)
# 
#     num_inside = int(zone_mask.sum() / D / B)
#     num_outside = int(H * W - num_inside)
#     inside_features = inside_features.reshape(B, num_inside, D)
#     outside_features = outside_features.reshape(B, num_outside, D)
#     return inside_features, outside_features
# 
# class Combine21(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine21, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.ca_inside = ChannelAttentionPointModule(3*d_model)
#         self.ca_outside = ChannelAttentionPointModule(3*d_model)
#         self.conv_inside = nn.Conv1d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.conv_outside = nn.Conv1d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
# 
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         old_feat0 = feat0
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         x_inside, x_outside = get_inside_outside_feature(x,zone_mask,H,W,B,D)
#         y_inside, y_outside = get_inside_outside_feature(y,zone_mask,H,W,B,D)
#         z_inside, z_outside = get_inside_outside_feature(z,zone_mask,H,W,B,D)
# 
# 
#         feat0_inside = torch.cat([x_inside, y_inside, z_inside], dim=-1)
#         feat0_outside = torch.cat([x_outside, y_outside, z_outside], dim=-1)
# 
#         feat0_inside = self.ca_inside(feat0_inside)
#         feat0_outside = self.ca_outside(feat0_outside)
# 
#         feat0_inside = feat0_inside.transpose(2, 1)
#         feat0_inside = self.conv_inside(feat0_inside)
#         feat0_inside = feat0_inside.permute(0, 2, 1)
# 
#         feat0_outside = feat0_outside.transpose(2, 1)
#         feat0_outside = self.conv_outside(feat0_outside)
#         feat0_outside = feat0_outside.permute(0, 2, 1)
# 
#         tmp = torch.zeros_like(old_feat0)
#         tmp[zone_mask] = feat0_inside.flatten()
#         tmp[~zone_mask] = feat0_outside.flatten()
# 
# 
#         tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=H).contiguous()
# 
#         feat0 = self.conv2(tmp)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0
# 
# 
# class Combine22(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  large_kernel):
#         super(Combine22, self).__init__()
# 
#         self.transformer_path = LoFTREncoderLayer_newcross9(d_model, nhead)
#         self.large_kernel_path = Block15(d_model, large_kernel=large_kernel)
#         self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, bias=False, padding=1)
#         self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.ca_inside = ChannelAttentionPointModule(3*d_model)
#         self.ca_outside = ChannelAttentionPointModule(3*d_model)
#         self.sa_inside= SpatialAttentionPointModule()
#         self.sa_outside= SpatialAttentionPointModule()
#         self.conv_inside = nn.Conv1d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
#         self.conv_outside = nn.Conv1d(3*d_model, d_model, kernel_size=1, bias=False, padding=0)
# 
# 
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         old_feat0 = feat0
#         x = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         feat0 = self.transformer_path(feat0,zone_mask,H,W,B,D)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         y = feat0
# 
#         feat0 = self.large_kernel_path(feat0)
#         feat0 = self.conv1(feat0)
#         z = feat0
# 
#         x_inside, x_outside = get_inside_outside_feature(x,zone_mask,H,W,B,D)
#         y_inside, y_outside = get_inside_outside_feature(y,zone_mask,H,W,B,D)
#         z_inside, z_outside = get_inside_outside_feature(z,zone_mask,H,W,B,D)
# 
# 
#         feat0_inside = torch.cat([x_inside, y_inside, z_inside], dim=-1)
#         feat0_outside = torch.cat([x_outside, y_outside, z_outside], dim=-1)
# 
#         feat0_inside = self.ca_inside(feat0_inside)
#         feat0_inside = self.sa_inside(feat0_inside)
#         feat0_outside = self.ca_outside(feat0_outside)
#         feat0_outside = self.sa_outside(feat0_outside)
# 
#         feat0_inside = feat0_inside.transpose(2, 1)
#         feat0_inside = self.conv_inside(feat0_inside)
#         feat0_inside = feat0_inside.permute(0, 2, 1)
# 
#         feat0_outside = feat0_outside.transpose(2, 1)
#         feat0_outside = self.conv_outside(feat0_outside)
#         feat0_outside = feat0_outside.permute(0, 2, 1)
# 
#         tmp = torch.zeros_like(old_feat0)
#         tmp[zone_mask] = feat0_inside.flatten()
#         tmp[~zone_mask] = feat0_outside.flatten()
# 
# 
#         tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=H).contiguous()
# 
#         feat0 = self.conv2(tmp)
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
# 
#         return feat0

# class cmt1(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead):
#         super(cmt1, self).__init__()
#
#         self.dim = d_model // nhead
#         self.nhead = nhead
#
#         # multi-head attention
#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)
#         self.attention = LinearAttention()
#         self.merge = nn.Linear(d_model, d_model, bias=False)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.LPU = LocalPerceptionUint(d_model)
#         self.IRFFN = InvertedResidualFeedForward(d_model, 4)
#         self.conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, bias=False, padding=1)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         lpu = self.LPU(feat0)
#         feat0 = feat0 + lpu
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat0 = self.norm1(feat0)
#
#         bs = feat0.size(0)
#         zone_area = zone_mask
#         inside_features = torch.masked_select(feat0, zone_area)
#         outside_features = torch.masked_select(feat0, ~zone_area)
#
#         num_inside = int(zone_mask.sum() / D / B)
#         num_outside = int(H * W - num_inside)
#         inside_features = inside_features.reshape(B, num_inside, D)
#         outside_features = outside_features.reshape(B, num_outside, D)
#
#         query, key, value = outside_features, inside_features, inside_features
#
#         # multi-head attention
#         query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
#         key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
#         value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
#
#         message = self.attention(query, key, value)  # [N, L, (H, D)]
#
#
#         tmp = torch.zeros_like(feat0)
#         tmp[~zone_area] = message.flatten()
#
#         tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=H)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
#
#         feat = torch.cat([feat0, tmp], dim=1)
#         feat = self.conv1(feat)
#         feat = rearrange(feat, 'b c h w -> b (h w) c', h=H)
#         feat = self.norm2(feat)
#         feat = rearrange(feat, 'b (h w) c -> b c h w', h=H)
#
#         ffn = self.IRFFN(feat)
#         feat0 = feat0 + ffn
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
#
#         return feat0
#
# class cmt2(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead):
#         super(cmt2, self).__init__()
#
#         self.dim = d_model // nhead
#         self.nhead = nhead
#
#         # multi-head attention
#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)
#         self.attention = LinearAttention()
#         self.merge = nn.Linear(d_model, d_model, bias=False)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.LPU = LocalPerceptionUint(d_model)
#         self.IRFFN = InvertedResidualFeedForward(d_model, 4)
#         self.conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, bias=False, padding=1)
#
#     def forward(self, feat0,zone_mask,H,W,B,D):
#         """
#         Args:
#             x (torch.Tensor): [N, L, C]
#             source (torch.Tensor): [N, S, C]
#             x_mask (torch.Tensor): [N, L] (optional)
#             source_mask (torch.Tensor): [N, S] (optional)
#         """
#         old_feat0=feat0
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()
#         lpu = self.LPU(feat0)
#         feat0 = feat0 + lpu
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H).contiguous()
#         feat0 = self.norm1(feat0)
#
#         bs = feat0.size(0)
#         zone_area = zone_mask
#         inside_features = torch.masked_select(feat0, zone_area)
#         outside_features = torch.masked_select(feat0, ~zone_area)
#
#         num_inside = int(zone_mask.sum() / D / B)
#         num_outside = int(H * W - num_inside)
#         inside_features = inside_features.reshape(B, num_inside, D)
#         outside_features = outside_features.reshape(B, num_outside, D)
#
#         query, key, value = outside_features, inside_features, inside_features
#
#         # multi-head attention
#         query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
#         key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
#         value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
#
#         message = self.attention(query, key, value)  # [N, L, (H, D)]
#
#
#         tmp = torch.zeros_like(feat0)
#         tmp[~zone_area] = message.flatten()
#
#         tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=H)
#         feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
#
#         feat = torch.cat([feat0, tmp], dim=1)
#         feat = self.conv1(feat)
#         feat = rearrange(feat, 'b c h w -> b (h w) c', h=H)
#         feat = self.norm2(feat)
#         feat = rearrange(feat, 'b (h w) c -> b c h w', h=H)
#
#         ffn = self.IRFFN(feat)
#         feat0 = feat0 + ffn
#         feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
#         feat0 = feat0+old_feat0
#         return feat0