from layers import *
from backbone.block.mobile import Mobile_V2_Block

# The official implementations of SSD and DSSD are written in caffe.
# To the best of my knowledge, this is the closest implementation to the official of SSD321/513.
# If you find an error, please let me know through Issues.


class SSD_300_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325

    The structure is based on the implementation;
    https://github.com/weiliu89/caffe/tree/ssd

    'channels' does not include the first feature's, because it does not go through any layer.
    """

    num_levels = 6

    def __init__(self,
                 channels: list,
                 Act: nn.Module = nn.ReLU()):

        c = channels

        if len(channels) != self.num_levels - 1:
            raise ValueError('make len(channels) == 5 == num_levels - 1')


        super(SSD_300_Extra, self).__init__()

        self.levels = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c[0], c[0] // 4, 1), Act,
                          nn.Conv2d(c[0] // 4, c[1], 3, stride=2, padding=1), Act),

            nn.Sequential(nn.Conv2d(c[1], c[1] // 4, 1), Act,
                          nn.Conv2d(c[1] // 4, c[2], 3, stride=2, padding=1), Act),

            nn.Sequential(nn.Conv2d(c[2], c[2] // 2, 1), Act,
                          nn.Conv2d(c[2] // 2, c[3], 3, padding=0), Act),

            nn.Sequential(nn.Conv2d(c[3], c[3] // 2, 1), Act,
                          nn.Conv2d(c[3] // 2, c[4], 3, padding=0), Act)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 6, make len(features) == 2')

        p_features = list(features)
        p = p_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.levels[i - 1](p)
            p_features.append(p)

        return p_features



class SSD_512_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325

    The structure is based on the official implementation;
    https://github.com/weiliu89/caffe/tree/ssd
    """

    num_levels = 7

    def __init__(self,
                 channels: list,
                 Act: nn.Module = nn.ReLU()):

        c = channels

        if len(channels) != self.num_levels - 1:
            raise ValueError('make len(channels) == 6 == num_levels - 1')


        super(SSD_512_Extra, self).__init__()

        self.levels = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c[0], c[0] // 4, 1), Act,
                          nn.Conv2d(c[0] // 4, c[1], 3, stride=2, padding=1), Act),

            nn.Sequential(nn.Conv2d(c[1], c[1] // 4, 1), Act,
                          nn.Conv2d(c[1] // 4, c[2], 3, stride=2, padding=1), Act),

            nn.Sequential(nn.Conv2d(c[2], c[2] // 2, 1), Act,
                          nn.Conv2d(c[2] // 2, c[3], 3, stride=2, padding=1), Act),

            nn.Sequential(nn.Conv2d(c[3], c[3] // 2, 1), Act,
                          nn.Conv2d(c[3] // 2, c[4], 3, stride=2, padding=1), Act),

            nn.Sequential(nn.Conv2d(c[4], c[4] // 2, 1), Act,
                          nn.Conv2d(c[4] // 2, c[5], 4, padding=1), Act)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 7, make len(features) == 2')

        p_features = list(features)
        p = p_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.levels[i - 1](p)
            p_features.append(p)

        return p_features



class Extra_Res_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659

    The structure is based on the official implementation;
    https://github.com/chengyangfu/caffe

    The difference from the official implementation is that 'dilation' is given to both 'block' and 'shortcut'.
    But it only concerns the first block of 'backbone_extra' in SSD_ResNet.
    """

    reduction = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 1,
                 Act: nn.Module = nn.ReLU(),
                 shortcut: bool = False):

        channels = int(out_channels / self.reduction)
        self.downsample = stride != 1 or in_channels != out_channels

        super(Extra_Res_Block, self).__init__()

        block = [Static_ConvLayer(in_channels, channels, 1, 1, Act=Act),
                 Dynamic_ConvLayer(channels, channels, kernel_size, stride, padding, dilation, Act=Act),
                 Static_ConvLayer(channels, out_channels, 1, 1, Act=None)]

        self.block = nn.Sequential(*block)

        if shortcut or self.downsample:
            self.shortcut = Dynamic_ConvLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, Act=None)
        else:
            self.shortcut = nn.Identity()

        self.act = Act


    def forward(self, x):
        input = x
        x = self.block(x)
        x += self.shortcut(input)
        x = self.act(x)
        return x



class SSD_321_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659

    The structure is based on the official implementation;
    https://github.com/chengyangfu/caffe
    """

    num_levels = 6

    def __init__(self,
                 channels: list,
                 Act: nn.Module = nn.ReLU()):

        c = channels

        if len(channels) != self.num_levels - 1:
            raise ValueError('make len(channels) == 5 == num_levels - 1')


        super(SSD_321_Extra, self).__init__()

        self.levels = nn.ModuleList([Extra_Res_Block(c[0], c[1], 2, 2, Act=Act),
                                     Extra_Res_Block(c[1], c[2], 2, 2, Act=Act),
                                     Extra_Res_Block(c[2], c[3], 3, 1, 0, Act=Act, shortcut=True),
                                     Extra_Res_Block(c[3], c[4], 3, 1, 0, Act=Act, shortcut=True)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 6, make len(features) == 2')

        p_features = list(features)
        p = p_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.levels[i - 1](p)
            p_features.append(p)

        return p_features



class SSD_513_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659

    The structure is based on the official implementation;
    https://github.com/chengyangfu/caffe
    """

    num_levels = 7

    def __init__(self,
                 channels: list,
                 Act: nn.Module = nn.ReLU()):

        c = channels

        if len(channels) != self.num_levels - 1:
            raise ValueError('make len(channels) == 6 == num_levels - 1')


        super(SSD_513_Extra, self).__init__()

        self.levels = nn.ModuleList([Extra_Res_Block(c[0], c[1], 2, 2, Act=Act),
                                     Extra_Res_Block(c[1], c[2], 2, 2, Act=Act),
                                     Extra_Res_Block(c[2], c[3], 2, 2, Act=Act),
                                     Extra_Res_Block(c[3], c[4], 2, 2, Act=Act),
                                     Extra_Res_Block(c[4], c[5], 2, 2, Act=Act)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 7, make len(features) == 2')

        p_features = list(features)
        p = p_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.levels[i - 1](p)
            p_features.append(p)

        return p_features



class SSDLite_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381

    The structure is based on the implementation;
    https://github.com/chuanqi305/MobileNetv2-SSDLite
    """

    num_levels = 6

    def __init__(self,
                 channels: list,
                 expansions: list,
                 Act: nn.Module = nn.ReLU6()):

        c, e = channels, expansions

        if len(channels) != self.num_levels - 1:
            raise ValueError('make len(channels) == 5 == num_levels - 1')

        if len(expansions) != len(channels) - 1:
            raise ValueError('make len(expansions) != len(channels) - 1')


        super(SSDLite_Extra, self).__init__()

        self.levels = nn.ModuleList([Mobile_V2_Block(c[0], c[1], e[0], stride=2, Act=Act),
                                     Mobile_V2_Block(c[1], c[2], e[1], stride=2, Act=Act),
                                     Mobile_V2_Block(c[2], c[3], e[2], stride=2, Act=Act),
                                     Mobile_V2_Block(c[3], c[4], e[3], stride=2, Act=Act)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 6, make len(features) == 2')

        p_features = list(features)
        p = p_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.levels[i - 1](p)
            p_features.append(p)

        return p_features
