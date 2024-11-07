from __future__ import absolute_import, division, print_function
from typing import Optional, Union, Type, List, Tuple, Dict
from typing import Optional, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")




class CompletionEncoder(nn.Module):
    """ Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers: Optional[int] = 18, pretrained: Optional[bool] = True):
        super(CompletionEncoder, self).__init__()

        assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
        blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block_type = {18: resnet.BasicBlock, 50: resnet.Bottleneck}[num_layers]

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(4, self.num_ch_enc[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dilation = 1
        self.inplanes = 64
        self.layer1 = self._make_layer(block_type, self.num_ch_enc[1], blocks[0])
        self.layer2 = self._make_layer(block_type, self.num_ch_enc[2], blocks[1], stride=2)
        self.inplanes += 1
        self.layer3 = self._make_layer(block_type, self.num_ch_enc[3], blocks[2], stride=2)
        self.layer4 = self._make_layer(block_type, self.num_ch_enc[4], blocks[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load ImageNet pretrained weights for available layers
        print("Load ImageNet pretrained weights.")
        if pretrained:
            model_state = self.state_dict()
            loaded_state = model_zoo.load_url(resnet.model_urls['resnet{}'.format(num_layers)])
            for loaded_key in loaded_state:
                if loaded_key in model_state:
                    if model_state[loaded_key].shape != loaded_state[loaded_key].shape:
                        model_state[loaded_key][:, :loaded_state[loaded_key].shape[1]].copy_(loaded_state[loaded_key])
                        print("{}: model_state_shape: {}, loaded_state_shape: {}".format(
                            loaded_key, model_state[loaded_key].shape, loaded_state[loaded_key].shape))
                    else:
                        model_state[loaded_key].copy_(loaded_state[loaded_key])
                else:
                    print("{}: In checkpoint but not in model".format(loaded_key))

    def _make_layer(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        planes: int,
        blocks: int,
        stride: Optional[int] = 1,
        dilate: Optional[bool] = False
    ) -> nn.Sequential:
        """ Adapted from torchvision/models/resnet.py
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, nn.BatchNorm2d))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

    def forward(
        self,
        input_image: torch.Tensor,
        disps: torch.Tensor
    ) -> List[torch.Tensor]:
        features = []
        disps_up = F.interpolate(disps, size=input_image.shape[2:], mode='bilinear', align_corners=True)

        x = input_image
        x = torch.cat([x, disps_up], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        features.append(self.relu(x))
        features.append(self.layer1(self.maxpool(features[-1])))
        features.append(self.layer2(features[-1]))
        disps_down = F.interpolate(disps, size=features[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([features[-1], disps_down], 1)
        features.append(self.layer3(x))
        features.append(self.layer4(features[-1]))

        return features


class CompletionDecoder(nn.Module):
    """ Depth completion decoder stage
    """
    def __init__(
        self,
        scales: Optional[List[int]] = None,
        num_output_channels: Optional[int] = 1,
        use_skips: Optional[bool] = True,
        min_depth: Optional[float] = 0.3,
        max_depth: Optional[float] = 200.0,
        focal_norm: Optional[float] = 1.0,
    ):
        super(CompletionDecoder, self).__init__()

        self.scales = scales if scales is not None else [0]
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.focal_norm = focal_norm

        self.upsample_mode = 'nearest'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, "upconv_{}_0".format(i), ConvBlock(num_ch_in, num_ch_out))
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, "upconv_{}_1".format(i), ConvBlock(num_ch_in, num_ch_out))
        for s in self.scales:
            setattr(self, "dispconv_{}".format(s), Conv3x3(self.num_ch_dec[s], self.num_output_channels))

        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features: List[torch.Tensor]):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            # x = self.convs[("upconv", i, 0)](x)
            x = getattr(self, 'upconv_{}_0'.format(i))(x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # x = self.convs[("upconv", i, 1)](x)
            x = getattr(self, 'upconv_{}_1'.format(i))(x)
            if i in self.scales:
                # outputs[("depth", i)] = self.convs[("dispconv", i)](x)
                outputs[("depth", i)] =  getattr(self, 'dispconv_{}'.format(i))(x)
        return self.sigmoid(outputs[("depth", 0)])*10.


class DepthCompletion(nn.Module):
    """ Depth completion network
    """
    def __init__(
        self,
        input_disp_mode: Optional[str] = 'normalize',
        min_conf: Optional[float] = 0.0,
        disp_dist: Optional[List[float]] = None,
        disp_up_dist: Optional[List[float]] = None,
        min_depth_in: Optional[float] = 1.0,
        max_depth_in: Optional[float] = 200.0,
        min_depth_out: Optional[float] = 1.0,
        max_depth_out: Optional[float] = 200.0,
        focal_norm: Optional[float] = 1.0,
        mask_disp_clip: Optional[float] = False,
        pretrained_encoder: Optional[bool] = True,
    ):
        super(DepthCompletion, self).__init__()
        self.input_disp_mode = input_disp_mode
        self.min_conf = min_conf
        self.disp_dist = disp_dist if disp_dist is not None else [0.0, 1.0]
        self.disp_up_dist = disp_up_dist if disp_up_dist is not None else [0.0, 1.0]
        self.min_depth_in = min_depth_in
        self.max_depth_in = max_depth_in
        self.mask_disp_clip = mask_disp_clip

        self.encoder = CompletionEncoder(num_layers=18, pretrained=pretrained_encoder)
        self.decoder = CompletionDecoder(min_depth=min_depth_out, max_depth=max_depth_out, focal_norm=focal_norm)
        self.do_focal_scaling = True

    def forward(
        self,
        image: torch.Tensor,
        disps: torch.Tensor,
    ):

        features = self.encoder(image,disps)
        outputs = self.decoder(input_features=features)
        return outputs



