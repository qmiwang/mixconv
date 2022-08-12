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

def relu(inplace=True):
    return nn.LeakyReLU(inplace=inplace)
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
        return nn.Conv3d(in_planes,
                 out_planes,
                 kernel_size=(3,3,1),
                 stride=stride,
                 padding=(1,1,0),
                 bias=True)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, **kwargs):
        super().__init__()
        kwargs["norm_type"] = norm_type
        self.relu = relu(inplace=True)
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
        self.relu = relu(inplace=True)
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

class MixConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, **kwargs):
        super().__init__()
        kwargs["norm_type"] = norm_type
        print(in_channels, out_channels, '---')
        in_planes, out_planes, stride = in_channels, out_channels, 1
        self.conv1_1 = nn.Conv3d(in_planes, int(out_planes/2),
                               kernel_size=[3, 3, 1], stride=stride, padding=[1, 1, 0])
        self.conv1_2 = nn.Conv3d(in_planes, int(out_planes/2),
                               kernel_size=[1, 3, 3], stride=stride, padding=[0, 1, 1])
        #self.conv1_3 = nn.Conv3d(in_planes, int(out_planes/2),
        #                       kernel_size=[3, 1, 3], stride=stride, padding=[1, 0, 1])
        self.bn1 = norm(out_planes, **kwargs)
        self.relu = relu(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes,
                               kernel_size=[3, 3, 1], stride=stride, padding=[1, 1, 0])
        self.bn2 = norm(out_channels, **kwargs)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1x1(in_channels, out_channels),
                norm(out_channels, **kwargs))

    def forward(self, x):
        residual = x
        out = torch.cat([self.conv1_1(x), self.conv1_2(x)], dim=1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out

class MixUNet3D(nn.Module):
    def __init__(self, config):
        super(MixUNet3D, self).__init__()
        input_channels, num_classes = config["data"]["color_channels"], config["data"]["num_classes"]
        block_name, base_filters = config["network"]["block"], config["network"]["base_filters"],
        norm_parm = config["network"]["norm_parm"]
        self.deep_supervision = deep_supervision = config["network"]["deep_supervision"]
        self.nb_filter = nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8,
                                      base_filters * 16]
        self.num_classes = num_classes
        self.config = config

        self.deep_supervision = deep_supervision
        self.is2dPhase = config["network"]["is2dPhase"]
        
        patch_size = list(config["network"]["patch_size"])
        spacing = list(config["network"]["spacing"])
        convs, pools = get_conv_pool(patch_size, spacing)

        print(config["network"], convs, pools)
        
        self.pool1, self.up1 = self.get_pool_up(pools[0])
        self.pool2, self.up2 = self.get_pool_up(pools[1])
        self.pool3, self.up3 = self.get_pool_up(pools[2])
        self.pool4, self.up4 = self.get_pool_up(pools[3])
        print(input_channels, nb_filter[0], norm_parm)
        self.encoder0 = self.get_conv_block(convs[0], input_channels, nb_filter[0], **norm_parm)
        self.encoder1 = self.get_conv_block(convs[1], nb_filter[0], nb_filter[1], **norm_parm)
        self.encoder2 = self.get_conv_block(convs[2], nb_filter[1], nb_filter[2], **norm_parm)
        self.encoder3 = self.get_conv_block(convs[3], nb_filter[2], nb_filter[3], **norm_parm)
        self.encoder4 = self.get_conv_block(convs[4], nb_filter[3], nb_filter[4], **norm_parm)
        self.decoder3 = self.get_conv_block(convs[3], nb_filter[3] + nb_filter[4], nb_filter[3], **norm_parm)
        self.decoder2 = self.get_conv_block(convs[2], nb_filter[2] + nb_filter[3], nb_filter[2], **norm_parm)
        self.decoder1 = self.get_conv_block(convs[1], nb_filter[1] + nb_filter[2], nb_filter[1], **norm_parm)
        self.decoder0 = self.get_conv_block(convs[0], nb_filter[0] + nb_filter[1], nb_filter[0], **norm_parm)

        if self.deep_supervision:
            self.final = nn.ModuleList([nn.Conv3d(nbf, num_classes, kernel_size=1) for nbf in nb_filter[:4]])
        else:
            self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')

    def forward(self, input):
        x0_0 = self.encoder0(input)
        x1_0 = self.encoder1(self.pool1(x0_0))
        x2_0 = self.encoder2(self.pool2(x1_0))
        x3_0 = self.encoder3(self.pool3(x2_0))
        x4_0 = self.encoder4(self.pool4(x3_0))
        x3_1 = self.decoder3(torch.cat([x3_0, self.up4(x4_0)], 1))
        x2_2 = self.decoder2(torch.cat([x2_0, self.up3(x3_1)], 1))
        x1_3 = self.decoder1(torch.cat([x1_0, self.up2(x2_2)], 1))
        x0_4 = self.decoder0(torch.cat([x0_0, self.up1(x1_3)], 1))
        
        if self.deep_supervision:
            output = [torch.squeeze(final(x),dim=1) for final, x in zip(self.final, [x0_4, x1_3, x2_2, x3_1])]
        else:
            output = torch.squeeze(self.final(x0_4),dim=1)
        return output
    
    def get_pool_up(self,pool_size):
        if self.is2dPhase:
            pool = nn.MaxPool3d(tuple(pool_size[:2]+[1,]), tuple(pool_size[:2]+[1,]))
            up = nn.Upsample(scale_factor=tuple(pool_size[:2]+[1,]), mode='trilinear', align_corners=True)
        else:
            pool = nn.MaxPool3d(tuple(pool_size), tuple(pool_size))
            up = nn.Upsample(scale_factor=tuple(pool_size), mode='trilinear', align_corners=True)
        return pool, up
    
    def get_conv_block(self, conv, in_nbf, out_nbf, **kargs):
        block_name = self.config["network"]["block"]
        if conv == '2d':
            conv_type = 'Conv2d'
            block_name = block_name.replace('Mix', '')
        else:
            conv_type = self.config["network"]["conv_type"]
        print( in_nbf, out_nbf, kargs)
        block = globals()[block_name](in_nbf, out_nbf, conv_type=conv_type, **kargs)
        return block
        


def get_conv_pool(patch_size, spacing):
    current_spacing = spacing
    current_size = patch_size

    smallest_size = 4

    convs = []
    pools = []

    for i in range(1,6):
        print(current_spacing, abs(current_spacing[2] - current_spacing[1]*2))
        if current_spacing[2] > current_spacing[1]*2 or abs(current_spacing[2] - current_spacing[1]*2) < 0.2:
            convs.append('2d')
            pool = [2, 2, 1]
        else:
            convs.append('3d')
            pool = [2, 2, 2]
        for j in range(3):
            if current_size[j] < smallest_size * 2 :
                pool[j] = 1
        pools.append(pool)

        for j in range(3):
            current_spacing[j] = pool[j] * current_spacing[j]
            current_size[j] = current_size[j] / pool[j]
    return convs, pools