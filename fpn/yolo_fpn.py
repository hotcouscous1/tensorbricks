from fpn.utils import *

# 'branch_in' is to receive a feature to fuse in the current level.
# 'branch_out' is to send a feature to be fused in the next level.


class Level(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767
    """

    def __init__(self,
                 channels: int,
                 branch_in: bool,
                 branch_out: bool,
                 spp_kernels: list = None,
                 up_size: int = None,
                 up_mode: str = 'nearest',
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        h_channels = channels // 2
        self.branch_in, self.branch_out = branch_in, branch_out


        super(Level, self).__init__()
        self.lateral = nn.ModuleList()

        if not branch_in:
            self.lateral.append(Static_ConvLayer(channels, h_channels, 1, Act=Act))
        else:
            self.lateral.append(Static_ConvLayer(int(channels * 1.5), h_channels, 1, Act=Act))


        for i in range(1, 5):
            if i % 2 == 0:
                if not (spp_kernels and i == 2):
                    l = Static_ConvLayer(channels, h_channels, 1, Act=Act)
                else:
                    l = nn.Sequential(Static_ConvLayer(channels, h_channels, 1, Act=Act),
                                      SPP(spp_kernels, inverse=True),
                                      Static_ConvLayer((1 + len(spp_kernels)) * h_channels, h_channels, 1, Act=Act))
            else:
                l = Static_ConvLayer(h_channels, channels, 3, Act=Act)

            self.lateral.append(l)

        self.lateral.append(Static_ConvLayer(h_channels, channels, 3, Act=Act))


        if branch_out:
            if up_size:
                self.upsample = nn.Sequential(Static_ConvLayer(h_channels, h_channels // 2, 1, Act=Act),
                                              nn.Upsample(size=up_size, mode=up_mode))
            else:
                self.upsample = nn.Sequential(Static_ConvLayer(h_channels, h_channels // 2, 1, Act=Act),
                                              nn.Upsample(scale_factor=2, mode=up_mode))


    def forward(self, f, b=None):
        if self.branch_in and b is not None:
            f = torch.cat((b, f), 1)

        for l in self.lateral[:-1]:
            f = l(f)

        if self.branch_out:
            b = self.upsample(f)
            f = self.lateral[-1](f)
            return f, b
        else:
            f = self.lateral[-1](f)
            return f



class Yolo_FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767

    The structure is based on the implementation;
    https://github.com/AlexeyAB/darknet
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 spp_kernels: list = None,
                 sizes: list = None,
                 up_mode: str = 'nearest',
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        self.num_levels = num_levels

        if len(channels) != num_levels:
            raise ValueError('make len(channels) == num_levels')

        if sizes:
            if len(sizes) != num_levels:
                raise ValueError('make len(sizes) == num_levels')
        else:
            sizes = num_levels * [None]


        super(Yolo_FPN, self).__init__()

        self.levels = nn.ModuleList()

        for i, c in enumerate(channels):
            up_size = sizes[i - 1] if i > 0 else None

            if i == num_levels - 1:
                self.levels.append(Level(c, False, True, spp_kernels, up_size, up_mode, Act))
            elif i > 0:
                self.levels.append(Level(c, True, True, None, up_size, up_mode, Act))
            else:
                self.levels.append(Level(c, True, False, None, up_size, up_mode, Act))


    def forward(self, features):
        p_features = []
        b = None

        for i in range(self.num_levels - 1, -1, -1):

            if i == len(features) - 1:
                p, b = self.levels[i](features[i])
                p_features.append(p)

            elif i > 0:
                p, b = self.levels[i](features[i], b)
                p_features.append(p)

            else:
                p = self.levels[i](features[i], b)
                p_features.append(p)

        return p_features[::-1]



class Yolo_V3_Tiny_FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767

    The structure is based on the implementation;
    https://github.com/AlexeyAB/darknet

    This module only takes features of # channels [256, 512].
    """

    def __init__(self,
                 up_mode: str = 'nearest',
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        super(Yolo_V3_Tiny_FPN, self).__init__()

        self.lateral5 = nn.ModuleList([Static_ConvLayer(512, 1024, 3, Act=Act),
                                       Static_ConvLayer(1024, 256, 1, Act=Act),
                                       Static_ConvLayer(256, 512,  3, Act=Act)])

        self.upsample5 = nn.Sequential(Static_ConvLayer(256, 128, 1, Act=Act),
                                       nn.Upsample(scale_factor=2, mode=up_mode))

        self.lateral4 = Static_ConvLayer(384, 256, 3, Act=Act)


    def forward(self, features):
        c4, c5 = features

        p5 = self.lateral5[0](c5)
        p5 = self.lateral5[1](p5)
        b4, p5 = self.upsample5(p5), self.lateral5[2](p5)

        p4 = torch.cat((b4, c4), 1)
        p4 = self.lateral4(p4)

        return p4, p5



class Yolo_V4_Tiny_FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2011.08036

    The structure is based on the implementation;
    https://github.com/AlexeyAB/darknet

    This module only takes features of # channels [256, 512].
    """

    def __init__(self,
                 up_mode: str = 'nearest',
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        super(Yolo_V4_Tiny_FPN, self).__init__()

        self.lateral5 = nn.ModuleList([Static_ConvLayer(512, 256, 1, Act=Act),
                                       Static_ConvLayer(256, 512, 3, Act=Act)])

        self.upsample5 = nn.Sequential(Static_ConvLayer(256, 128, 1, Act=Act),
                                       nn.Upsample(scale_factor=2, mode=up_mode))

        self.lateral4 = Static_ConvLayer(384, 256, 3, Act=Act)


    def forward(self, features):
        c4, c5 = features

        p5 = self.lateral5[0](c5)
        b4, p5 = self.upsample5(p5), self.lateral5[1](p5)

        p4 = torch.cat((b4, c4), 1)
        p4 = self.lateral4(p4)

        return p4, p5

