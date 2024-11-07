import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..config import args
from .transformer import *
from .convnext import Block1 as ConvNextBlock
from .convnext import Block2 as ConvNextBlock2
# from .convnext import Block3 as ConvNextBlock3
# from .convnext import Block4 as ConvNextBlock4
# from .convnext import Block5 as ConvNextBlock5
# from .convnext import Block6 as ConvNextBlock6
# from .convnext import Block7 as ConvNextBlock7
# from .convnext import Block8 as ConvNextBlock8
# from .convnext import Block9 as ConvNextBlock9
# from .convnext import Block10 as ConvNextBlock10
# from .convnext import Block11 as ConvNextBlock11
# from .convnext import Block12 as ConvNextBlock12
# from .convnext import Block13 as ConvNextBlock13
from .convnext import Block14 as ConvNextBlock14
from .convnext import Block15 as ConvNextBlock15
from .convnext import Block16 as ConvNextBlock16
from .convnext import Block17 as ConvNextBlock17
from .convnext import Block18 as ConvNextBlock18
from .convnext import Block19 as ConvNextBlock19
from .convnext import Block20 as ConvNextBlock20
from .convnext import Block21 as ConvNextBlock21
from .convnext import Block22 as ConvNextBlock22
from .convnext import Block23 as ConvNextBlock23
from .convnext import Block24 as ConvNextBlock24
from .convnext import Block25 as ConvNextBlock25
from .convnext import Block26 as ConvNextBlock26
from .convnext import Block27 as ConvNextBlock27
from .convnext import Block28 as ConvNextBlock28

class TransformerFusion(nn.Module):
    def __init__(self, embedding_dim, max_resolution, num_heads=4,large_kernel=None,patch_size=None):
        super(TransformerFusion, self).__init__()

        self.zone_sample_num = args.zone_sample_num
        self.max_resolution = max_resolution
        self.positional_encodings = nn.Parameter(torch.rand(max_resolution[0]*max_resolution[1], embedding_dim), requires_grad=True)
        self.positional_encodings2 = nn.Parameter(torch.rand(self.zone_sample_num, embedding_dim), requires_grad=True)

        # https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L161
        nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        nn.init.trunc_normal_(self.positional_encodings2, std=0.2)

        self.layer_names = args.attention_layer
        encoder_layer = LoFTREncoderLayer(embedding_dim, num_heads)

        ws = math.ceil(math.sqrt(math.sqrt(self.max_resolution[0] * self.max_resolution[1])))
        image_encoder_layer = TwinsTransformer(embedding_dim, num_heads, ws=ws)
        layer_list = []
        for i in range(len(self.layer_names)):
            name = self.layer_names[i]
            if name == 'image': layer = copy.deepcopy(image_encoder_layer)
            elif name == 'inside_attn1': layer = Inside_attn1(embedding_dim, num_heads, ws=patch_size)
            elif name == 'inside_attn2': layer = Inside_attn2(embedding_dim, num_heads, ws=patch_size)
            elif name == 'inside_attn3': layer = Inside_attn3(embedding_dim, num_heads, ws=patch_size)
            elif name == 'inside_attn4': layer = Inside_attn4(embedding_dim, num_heads, ws=patch_size)
            elif name == 'inside_attn5': layer = Inside_attn5(embedding_dim, num_heads, ws=patch_size)
            elif name == 'inside_attn6': layer = Inside_attn6(embedding_dim, num_heads, ws=patch_size)
            elif name == 'trans1': layer = My_1_TwinsTransformer(embedding_dim, num_heads, ws=ws)
            elif name == 'trans2': layer = My_2_TwinsTransformer(embedding_dim, num_heads, ws=ws)
            elif name == 'trans3': layer = My_3_TwinsTransformer(embedding_dim, num_heads, ws=ws)
            elif name == 'trans4': layer = My_4_TwinsTransformer(embedding_dim, num_heads, ws=ws)
            elif name == 'trans5': layer = My_5_TwinsTransformer(embedding_dim, num_heads, ws=ws)
            elif name == 'trans6': layer = My_6_TwinsTransformer(embedding_dim, num_heads, ws=ws)
            elif name =='hist2image': layer = copy.deepcopy(encoder_layer)
            elif name == 'hist2image2': layer = LoFTREncoderLayer2(embedding_dim, num_heads)
            elif name == 'hist2image3': layer = LoFTREncoderLayer3(embedding_dim, num_heads)
            elif name == 'hist2image4': layer = LoFTREncoderLayer3(embedding_dim, num_heads)
            elif name == 'hist2image5': layer = LoFTREncoderLayer5(embedding_dim, num_heads)
            elif name == 'hist2image6': layer = LoFTREncoderLayer6(embedding_dim, num_heads)
            elif name == 'hist2image7': layer = LoFTREncoderLayer7(embedding_dim, num_heads)
            elif name == 'hist2image8': layer = LoFTREncoderLayer8(embedding_dim, num_heads)
            elif name == 'hist2image9': layer = LoFTREncoderLayer9(embedding_dim, num_heads)
            elif name == 'hist2image10': layer = LoFTREncoderLayer10(embedding_dim, num_heads)
            elif name == 'cross1': layer = LoFTREncoderLayer_cross1(embedding_dim, num_heads)
            elif name == 'cross2': layer = LoFTREncoderLayer_cross2(embedding_dim, num_heads)
            elif name == 'cross3': layer = LoFTREncoderLayer_cross1(embedding_dim, num_heads)
            elif name == 'cross4': layer = LoFTREncoderLayer_cross1(embedding_dim, num_heads)
            elif name == 'cross5': layer = LoFTREncoderLayer_cross5(embedding_dim, num_heads)
            elif name == 'cross6': layer = LoFTREncoderLayer_cross6(embedding_dim, num_heads)
            elif name == 'cross3_1': layer = nn.ModuleList([LoFTREncoderLayer_cross1(embedding_dim, num_heads)])
            elif name == 'cross3_2': layer = nn.ModuleList([LoFTREncoderLayer_cross1(embedding_dim, num_heads),
                                                            LoFTREncoderLayer_cross1(embedding_dim, num_heads)])
            elif name == 'cross3_3': layer = nn.ModuleList([LoFTREncoderLayer_cross1(embedding_dim, num_heads),
                                                            LoFTREncoderLayer_cross1(embedding_dim, num_heads),
                                                            LoFTREncoderLayer_cross1(embedding_dim, num_heads)])
            elif name == 'new_cross1':layer = LoFTREncoderLayer_newcross1(embedding_dim, num_heads)
            elif name == 'new_cross2':layer = LoFTREncoderLayer_newcross2(embedding_dim, num_heads)
            elif name == 'new_cross3':layer = LoFTREncoderLayer_newcross3(embedding_dim, num_heads)
            elif name == 'new_cross4':layer = LoFTREncoderLayer_newcross4(embedding_dim, num_heads)
            elif name == 'new_cross5':layer = LoFTREncoderLayer_newcross5(embedding_dim, num_heads)
            elif name == 'new_cross6':layer = LoFTREncoderLayer_newcross6(embedding_dim, num_heads)
            elif name == 'new_cross7':layer = LoFTREncoderLayer_newcross7(embedding_dim, num_heads)
            elif name == 'new_cross8':layer = LoFTREncoderLayer_newcross8(embedding_dim, num_heads)
            elif name == 'new_cross9':layer = LoFTREncoderLayer_newcross9(embedding_dim, num_heads)
            elif name == 'new_cross9_2':
                layer = nn.ModuleList([
                    LoFTREncoderLayer_newcross9(embedding_dim, num_heads),
                    LoFTREncoderLayer_newcross9(embedding_dim, num_heads),
                ])
            elif name == 'new_cross10':layer = LoFTREncoderLayer_newcross10(embedding_dim, num_heads)
            elif name == 'new_cross11':layer = LoFTREncoderLayer_newcross11(embedding_dim, num_heads)
            elif name == 'new_cross12':layer = LoFTREncoderLayer_newcross12(embedding_dim, num_heads)
            elif name == 'new_cross13':layer = LoFTREncoderLayer_newcross13(embedding_dim, num_heads)

            elif name == 'cvxt': layer = ConvNextBlock(embedding_dim)
            elif name == 'cvxt2': layer = ConvNextBlock2(embedding_dim)
            elif name == 'cvxt3': layer = ConvNextBlock3(embedding_dim)
            elif name == 'cvxt4': layer = ConvNextBlock4(embedding_dim)
            elif name == 'cvxt5': layer = ConvNextBlock5(embedding_dim)
            elif name == 'cvxt6': layer = ConvNextBlock6(embedding_dim)
            elif name == 'cvxt7': layer = ConvNextBlock7(embedding_dim)
            elif name == 'cvxt8': layer = ConvNextBlock8(embedding_dim)
            elif name == 'cvxt9': layer = ConvNextBlock9(embedding_dim)
            elif name == 'cvxt10': layer = ConvNextBlock10(embedding_dim)
            elif name == 'cvxt11': layer = ConvNextBlock11(embedding_dim)
            elif name == 'cvxt12': layer = ConvNextBlock12(embedding_dim)
            elif name == 'cvxt13': layer = ConvNextBlock13(embedding_dim)
            elif name == 'cvxt14': layer = ConvNextBlock14(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt15': layer = ConvNextBlock15(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt16': layer = ConvNextBlock16(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt17': layer = ConvNextBlock17(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt18': layer = ConvNextBlock18(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt19': layer = ConvNextBlock19(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt20': layer = ConvNextBlock20(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt21': layer = ConvNextBlock21(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt22': layer = ConvNextBlock22(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt23': layer = ConvNextBlock23(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt24': layer = ConvNextBlock24(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt25': layer = ConvNextBlock25(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt26': layer = ConvNextBlock26(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt27': layer = ConvNextBlock27(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt28': layer = ConvNextBlock28(embedding_dim,large_kernel=large_kernel)
            elif name == 'cvxt15_2': layer = nn.ModuleList([
                ConvNextBlock15(embedding_dim,large_kernel=large_kernel),
                ConvNextBlock15(embedding_dim,large_kernel=large_kernel),
            ])
            elif name == 'selfattn1': layer = My_Selfattn1(embedding_dim, num_heads, ws=ws)
            elif name =='combine1': layer = Combine1(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine2': layer = Combine2(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine3': layer = Combine3(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine4': layer = Combine4(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine4drop': layer = Combine4drop(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine4drop_2': layer = nn.ModuleList([
                Combine4drop(embedding_dim, num_heads, large_kernel=large_kernel),
                Combine4drop(embedding_dim, num_heads, large_kernel=large_kernel),
                ])
            elif name =='combine4bn': layer = Combine4bn(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine4bn_2': layer = nn.ModuleList([
                Combine4bn(embedding_dim, num_heads, large_kernel=large_kernel),
                Combine4bn(embedding_dim, num_heads, large_kernel=large_kernel),
                ])
            elif name =='combine5': layer = Combine5(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine6': layer = Combine6(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine7': layer = Combine7(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine8': layer = Combine8(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine9': layer = Combine9(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine9bn': layer = Combine9bn(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine9ln': layer = Combine9ln(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine10': layer = Combine10(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine11': layer = Combine11(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine12': layer = Combine12(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine13': layer = Combine13(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine14': layer = Combine14(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine15': layer = Combine15(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine16': layer = Combine16(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine17': layer = Combine17(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine18': layer = Combine18(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine19': layer = Combine19(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine20': layer = Combine20(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine21': layer = Combine21(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine22': layer = Combine22(embedding_dim, num_heads, large_kernel=large_kernel)
            elif name =='combine4_2': layer = nn.ModuleList([
                Combine4(embedding_dim, num_heads, large_kernel=large_kernel),
                Combine4(embedding_dim, num_heads, large_kernel=large_kernel),
                ])
            elif name =='combine4_3': layer = nn.ModuleList([
                Combine4(embedding_dim, num_heads, large_kernel=large_kernel),
                Combine4(embedding_dim, num_heads, large_kernel=large_kernel),
                Combine4(embedding_dim, num_heads, large_kernel=large_kernel),
                ])

            elif name == 'cmt1': layer = cmt1(embedding_dim, num_heads)
            elif name == 'cmt2': layer = cmt2(embedding_dim, num_heads)
            else: raise NotImplementedError
            layer_list.append(layer)
        self.layers = nn.ModuleList(layer_list)

        self.conv_patch_size = 640 / self.max_resolution[1]

    def interpolate_pos_encoding_2d(self, pos, size):
        h, w = size
        pos_n = F.interpolate(pos.unsqueeze(0), size=[h, w], mode='bilinear', align_corners=True)[0]
        return pos_n

    def interpolate_pos_encoding_1d(self, pos, l):
        pos_n = F.interpolate(pos.unsqueeze(0), size=[l], mode='linear', align_corners=True)[0]
        return pos_n

    def forward(self, x, feat1, **kwargs):
    # def forward(self, x, feat1, feat1_mask, rect_data, patch_info):
        # import ipdb; ipdb.set_trace()

        # conv -> patch embeddings
        # zone_mask -> patches covered by 8x8 zones
        # hist_mask -> patches covered by non-zero 8x8 zones (those with dist in 0-4m)

        embeddings = x
        B, D, H, W = embeddings.size()
        zone_sample_num = feat1.size(2)
        # print(feat1_mask.flatten())

        # extract patch width/height from rect_data
        # assume max_patch_size across batch are the same
        rect_data = kwargs['rect_data']
        feat1_mask = kwargs['mask']
        patch_info = kwargs['patch_info']
        zone_num = patch_info['zone_num'][0]
        pad_size = patch_info[self.conv_patch_size]['pad_size']
        patch_size = patch_info[self.conv_patch_size]['patch_size']
        index_wo_pad = patch_info[self.conv_patch_size]['index_wo_pad']

        pad_height, pad_width = torch.max(pad_size, axis=0)[0]
        p1, p2 = torch.max(patch_size, axis=0)[0]
        sy_wo_pad, sx_wo_pad = torch.min(index_wo_pad, axis=0)[0][0:2]
        ey_wo_pad, ex_wo_pad = torch.max(index_wo_pad, axis=0)[0][2:4]
        sy, ey = sy_wo_pad+pad_height, ey_wo_pad+pad_height
        sx, ex = sx_wo_pad+pad_width, ex_wo_pad+pad_width
        tzh, tzw = ey-sy, ex-sx
        interpolate = False
        if (ey-sy) != p1*zone_num or (ex-sx) != p2*zone_num:
            interpolate = True

        # add positional encoding
        offset_y, offset_x = 0, 0
        if embeddings.shape[2] < self.max_resolution[0]:
            offset_y = torch.randint(0, self.max_resolution[0]-embeddings.shape[2]+1,[1])
        if embeddings.shape[3] < self.max_resolution[1]:
            offset_x = torch.randint(0, self.max_resolution[1]-embeddings.shape[3]+1,[1])
        positional_encodings = self.positional_encodings.view([self.max_resolution[0], self.max_resolution[1], -1])
        positional_encodings = positional_encodings[offset_y:offset_y+H,offset_x:offset_x+W,:]
        positional_encodings = positional_encodings.permute(2, 0, 1)
        # org_embeddings = embeddings.clone()
        embeddings = embeddings + positional_encodings
        feat0 = embeddings.flatten(2).permute(0, 2, 1).contiguous()

        # prepare zone_mask and hist_mask
        # zone_mask -> b, (h, w), c
        # hist_mask -> (b zn zn) (p1 p2) c
        # firstly assign mask value False/True
        zone_mask = torch.zeros([B, H, W]).to(torch.bool).to(feat0.device)
        zone_mask[:, torch.clip(sy_wo_pad,0,H):torch.clip(ey_wo_pad,0,H), torch.clip(sx_wo_pad,0,W):torch.clip(ex_wo_pad,0,W)] = 1
        hist_mask = rearrange(feat1_mask, 'b (zn zn2) -> b zn zn2', zn=zone_num)
        hist_mask = repeat(hist_mask, 'b zn zn2 -> b (zn p1) (zn2 p2)', p1=p1, p2=p2)
        # secondly, reshape to corresponding shape
        zone_mask = rearrange(zone_mask, 'b h w -> b (h w)')
        zone_mask = repeat(zone_mask, 'b s -> b s c', c=D).contiguous()
        hist_mask = rearrange(hist_mask, 'b (zn p1) (zn2 p2) -> (b zn zn2) (p1 p2)', zn=zone_num, p1=p1, p2=p2)
        hist_mask = repeat(hist_mask, 'bzz p1p2 -> bzz p1p2 c', c=D).contiguous()
        if pad_height > 0 or pad_width > 0:
            pad_mask = torch.ones([B, D, tzh, tzw]).to(torch.bool).to(feat0.device)
            pad_mask[:,:,:torch.clip(0-sy_wo_pad,0,None)] = 0
            if torch.clip(ey_wo_pad-H,0,None) > 0: pad_mask[:,:,-torch.clip(ey_wo_pad-H,0,None):] = 0
            pad_mask[...,:torch.clip(0-sx_wo_pad,0,None)] = 0
            if torch.clip(ex_wo_pad-W,0,None) > 0: pad_mask[...,-torch.clip(ex_wo_pad-W,0,None):] = 0
            pad_mask = rearrange(pad_mask, 'b c tzh tzw -> (b tzh tzw c)')
        else:
            pad_mask = torch.ones([B*D*(ey-sy)*(ex-sx)]).to(torch.bool).to(feat0.device)

        # process hist feature
        positional_encodings2 = self.positional_encodings2
        feat1 = feat1 + positional_encodings2.unsqueeze(0).unsqueeze(0)
        feat1 = rearrange(feat1, 'b z n d -> (b z) n d').contiguous()
        feat1_mask = rearrange(feat1_mask, 'b z -> (b z)')
        feat1_mask = repeat(feat1_mask, 'b -> b l', l=self.zone_sample_num)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'image':
                feat0 = layer(feat0, (H, W))
            elif name == 'hist2image' or name == 'hist2image2' or name == 'hist2image4' or name == 'hist2image5'\
                    or name == 'hist2image6' or name == 'hist2image7' or  name == 'hist2image8'\
                    or name == 'hist2image9' or  name == 'hist2image10':
                # extract zone from full feature map
                if args.change_embedding:
                    embeddings = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
                feat0_unflatten = F.pad(embeddings, (pad_width, pad_width, pad_height, pad_height), 'constant', 0)
                #
                zone_feature = feat0_unflatten[:, :, int(sy):int(ey), int(sx):int(ex)]
                # if need interpolate
                if interpolate:
                    zone_feature = F.interpolate(zone_feature, size=[zone_num*p1, zone_num*p2], mode='bilinear', align_corners=True)
                zone_feature = rearrange(zone_feature, 'b c (ph p1) (pw p2) -> (b ph pw) (p1 p2) c', p1=p1, p2=p2)
                zone_feature = layer(zone_feature, feat1,real_bs=B,p1=p1,p2=p2)
                zone_feature[~hist_mask] = 0
                # interpolate back to original size
                if interpolate:
                    zone_feature = rearrange(zone_feature, '(b ph pw) (p1 p2) c -> b c (ph p1) (pw p2)', b=B, ph=zone_num, p1=p1)
                    zone_feature = F.interpolate(zone_feature, size=[tzh, tzw], mode='bilinear', align_corners=True)
                    zone_feature = rearrange(zone_feature, 'b c tzh tzw -> (b tzh tzw c)')
                else:
                    zone_feature = rearrange(zone_feature, '(b zn zn2) (p1 p2) c -> (b zn p1 zn2 p2 c)',
                                                    zn=zone_num, zn2=zone_num, p1=p1, p2=p2)

                if args.no_skip_inside:
                    feat0[zone_mask] = zone_feature[pad_mask]
                else:
                    feat0[zone_mask] += zone_feature[pad_mask]

            elif 'inside_attn' in name:
                # extract zone from full feature map
                if args.change_embedding:
                    embeddings = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
                feat0_unflatten = F.pad(embeddings, (pad_width, pad_width, pad_height, pad_height), 'constant', 0)
                #
                zone_feature = feat0_unflatten[:, :, int(sy):int(ey), int(sx):int(ex)]
                # if need interpolate
                if interpolate:
                    zone_feature = F.interpolate(zone_feature, size=[zone_num*p1, zone_num*p2], mode='bilinear', align_corners=True)
                zone_feature = rearrange(zone_feature, 'b c (ph p1) (pw p2) -> b (ph pw p1 p2) c', p1=p1, p2=p2)
                zone_feature = layer(zone_feature,(p1*zone_num,p2*zone_num))
                # interpolate back to original size
                if interpolate:
                    zone_feature = rearrange(zone_feature, 'b (ph pw p1 p2) c -> b c (ph p1) (pw p2)', b=B, ph=zone_num, p1=p1,pw=zone_num, p2=p2)
                    zone_feature = F.interpolate(zone_feature, size=[tzh, tzw], mode='bilinear', align_corners=True)
                    zone_feature = rearrange(zone_feature, 'b c tzh tzw -> (b tzh tzw c)')
                else:
                    zone_feature = rearrange(zone_feature, 'b (zn zn2 p1 p2) c -> (b zn p1 zn2 p2 c)',
                                                    zn=zone_num, zn2=zone_num, p1=p1, p2=p2)

                feat0[zone_mask] = zone_feature[pad_mask]


            elif name =='trans1' or name =='trans2' or name =='trans3' or name =='trans4' or name =='trans5' or name =='trans6':
                feat0 = layer(feat0, (H,W), org_embeddings)

            elif 'new_cross' in name:
                feat0 = layer(feat0, zone_mask,H,W,B,D)

            elif 'combine' in name:
                if '_' in name:
                    num_iters = int(name.split('_')[-1])
                    for i in range(num_iters):
                        feat0 = layer[i](feat0, zone_mask,H,W,B,D)
                else:
                    feat0 = layer(feat0, zone_mask,H,W,B,D)

            elif 'cmt' in name:
                feat0 = layer(feat0, zone_mask,H,W,B,D)

            elif 'cvxt' in name:
                if '_' in name:
                    num_iters = int(name.split('_')[-1])
                    for i in range(num_iters):
                        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
                        feat0 = layer[i](feat0)
                        feat0 = rearrange(feat0, 'b c h w -> b (h w) c', h=H)
                else:
                    feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
                    feat0 = layer(feat0)
                    feat0 = rearrange(feat0, 'b c h w -> b (h w) c',h=H)

            elif 'selfattn' in name:
                feat0 = layer(feat0, (H, W))

            # elif name=='cross3_1' or name=='cross3_2' or name=='cross3_3':
            #     if args.change_embedding:
            #         embeddings = rearrange(feat0, 'b (h w) c -> b c h w', h=H)
            #
            #     sy_wo_pad, sx_wo_pad = torch.min(index_wo_pad, axis=0)[0][0:2]
            #     ey_wo_pad, ex_wo_pad = torch.max(index_wo_pad, axis=0)[0][2:4]
            #     aa,bb,cc,dd = torch.clamp(sy_wo_pad,0,H),torch.clamp(ey_wo_pad,0,H),torch.clamp(sx_wo_pad,0,W),torch.clamp(ex_wo_pad,0,W)
            #     zone_area = torch.zeros_like(embeddings).bool()
            #     zone_area[:, :, int(aa):int(bb), int(cc):int(dd)] = 1
            #     zone_area = rearrange(zone_area, 'b c h w -> b (h w) c')
            #     num_iters = int(name.split('_')[-1])
            #     for i in range(num_iters):
            #         inside_features = torch.masked_select(feat0, zone_area)
            #         outside_features = torch.masked_select(feat0, ~zone_area)
            #
            #
            #         num_outside = int(H*W - (bb-aa)*(dd-cc))
            #         num_inside = int((bb-aa)*(dd-cc))
            #         inside_features = inside_features.reshape(B, num_inside, D)
            #         outside_features = outside_features.reshape(B, num_outside, D)
            #
            #         zone_feature = layer[i](outside_features, inside_features, real_bs=B, p1=p1, p2=p2)
            #         tmp = torch.zeros_like(feat0)
            #         tmp[~zone_area] = zone_feature.flatten()
            #         feat0 = feat0+tmp
            #
            #         tmp = torch.zeros_like(feat0)
            #         tmp[zone_area] = inside_features.flatten()
            #         feat0 = feat0+tmp

            else:
                raise NotImplementedError

        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H).contiguous()

        return feat0



