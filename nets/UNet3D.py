import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights
# from .MixConv import MixResBlock 
import numpy as np

def norm(in_channels, norm_type, **kwargs):
    if norm_type.upper() == "BATCH":
        return nn.BatchNorm3d(in_channels)
    elif norm_type.upper() == "GROUP":
        return nn.GroupNorm(min(kwargs["num_groups"], in_channels), in_channels)
    elif norm_type.upper() == "INSTANCE":
        return nn.InstanceNorm3d(in_channels)
    else:
        assert False, "not supported normlization method : {}".format(norm_type)


def conv3x3x3(in_planes, out_planes, stride=1, **kwargs):
    if kwargs['conv_type'].lower() == 'conv3d':
        kernel_size = 3 if 'kernel_size' not in kwargs else kwargs['kernel_size']
        padding = 1 if 'kernel_size' not in kwargs else np.floor((np.array(kernel_size)/3)).astype(int).tolist()
        return nn.Conv3d(in_planes,
                         out_planes,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=True)
    elif kwargs['conv_type'].lower() == 'conv2d':
        kernel_size = [3, 3, 3]
        padding = [1, 1, 1]
        if "view_dim" in kwargs:
            kernel_size[kwargs['view_dim']] = 1
            padding[kwargs['view_dim']] = 0
        else:
            kernel_size[2] = 1
            padding[2] = 0
        return nn.Conv3d(in_planes,
                 out_planes,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding,
                 bias=True)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class OutBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, norm_type, **kwargs):
        super(OutBlock, self).__init__()
        self.conv1 = conv1x1x1(in_channels, middle_channels, stride=1)
        self.bn1 = norm(middle_channels, norm_type, **kwargs)
        self.conv2 = conv3x3x3(middle_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, **kwargs):
        super().__init__()
        kwargs["norm_type"] = norm_type
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3x3(in_channels, out_channels, **kwargs)
        self.bn1 = norm(out_channels, **kwargs)  # GroupNorm(16, middle_channels) #
        self.conv2 = conv3x3x3(out_channels, out_channels, **kwargs)
        self.bn2 = norm(out_channels, **kwargs)  # nn.GroupNorm(16, out_channels) #nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, **kwargs):
        super().__init__()
        kwargs["norm_type"] = norm_type
        self.conv1 = conv3x3x3(in_channels, out_channels, **kwargs)
        self.bn1 = norm(out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channels, out_channels, **kwargs)
        self.bn2 = norm(out_channels, **kwargs)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1x1(in_channels, out_channels),
                norm(out_channels, **kwargs))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out

class UNet3D(nn.Module):
    def __init__(self, config):
        super(UNet3D, self).__init__()
        input_channels, num_classes = config["data"]["color_channels"], config["data"]["num_classes"]
        block_name, base_filters = config["network"]["block"], config["network"]["base_filters"],
        norm_parm = config["network"]["norm_parm"]
        self.deep_supervision = deep_supervision = config["network"]["deep_supervision"]
        self.nb_filter = nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8,
                                      base_filters * 16]
        self.num_classes = num_classes

        self.deep_supervision = deep_supervision
        self.is2dPhase = config["network"]["is2dPhase"]
        
        print(config["network"].get("view_dim", -1))
        
        if self.is2dPhase:
            view_dim = config["network"].get("view_dim", -1)
            kernel = [2, 2, 2]
            kernel[view_dim] = 1
            self.pool = nn.MaxPool3d(kernel, kernel)
            self.up = nn.Upsample(scale_factor=tuple(kernel), mode='trilinear', align_corners=True)
        else:
            self.pool = nn.MaxPool3d(2, 2)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        if config["network"]["conv_type"] == "Conv2d":
            norm_parm["view_dim"] = config["network"].get("view_dim", -1)
        print(config["network"], norm_parm)

        self.conv0_0 = globals()[block_name](input_channels, nb_filter[0], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv1_0 = globals()[block_name](nb_filter[0], nb_filter[1], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv2_0 = globals()[block_name](nb_filter[1], nb_filter[2], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv3_0 = globals()[block_name](nb_filter[2], nb_filter[3], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv4_0 = globals()[block_name](nb_filter[3], nb_filter[4], conv_type=config["network"]["conv_type"], **norm_parm)

        self.conv3_1 = globals()[block_name](nb_filter[3] + nb_filter[4], nb_filter[3], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv2_2 = globals()[block_name](nb_filter[2] + nb_filter[3], nb_filter[2], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv1_3 = globals()[block_name](nb_filter[1] + nb_filter[2], nb_filter[1], conv_type=config["network"]["conv_type"], **norm_parm)
        self.conv0_4 = globals()[block_name](nb_filter[0] + nb_filter[1], nb_filter[0], conv_type=config["network"]["conv_type"], **norm_parm)

        
        if self.deep_supervision:
            self.final = nn.ModuleList([nn.Conv3d(nbf, num_classes, kernel_size=1) for nbf in nb_filter[:4]])
        else:
            self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        if self.deep_supervision:
            output = [torch.squeeze(final(x),dim=1) for final, x in zip(self.final, [x0_4, x1_3, x2_2, x3_1])]
        else:
            output = torch.squeeze(self.final(x0_4),dim=1)
        return output