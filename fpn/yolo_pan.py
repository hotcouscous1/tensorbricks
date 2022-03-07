from fpn.utils import *

# 'branch_in' is to receive a feature to fuse in the current level.
# 'branch_out' is to send a feature to be fused in the next level.
# 'branch_channels' is channels of branch-out feature.


class TD_Level(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2004.10934
    """

    def __init__(self,
                 channels: int,
                 branch_in: bool,
                 branch_out: bool,
                 branch_channels: int,
                 spp_kernels: list = None,
                 up_size: int = None,
                 up_mode: str = 'nearest',
                 Act:nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        h_channels = channels // 2
        self.branch_in, self.branch_out = branch_in, branch_out


        super(TD_Level, self).__init__()

        if branch_in:
            self.fuse = Static_ConvLayer(channels, h_channels, 1, Act=Act)

        self.lateral = nn.ModuleList()

        for i in range(5):
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


        if branch_out:
            if up_size:
                self.upsample = nn.Sequential(Static_ConvLayer(h_channels, branch_channels, 1, Act=Act),
                                              nn.Upsample(size=up_size, mode=up_mode))
            else:
                self.upsample = nn.Sequential(Static_ConvLayer(h_channels, branch_channels, 1, Act=Act),
                                              nn.Upsample(scale_factor=2, mode=up_mode))


    def forward(self, f, b=None):
        if self.branch_in and b is not None:
            f = self.fuse(f)
            f = torch.cat((f, b), 1)

        for l in self.lateral:
            f = l(f)

        if self.branch_out:
            b = self.upsample(f)
            return f, b
        else:
            return f



class BU_Level(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2004.10934
    """

    def __init__(self,
                 channels: int,
                 branch_in: bool,
                 branch_out: bool,
                 branch_channels: int,
                 sizes: list = None,
                 stride: int = None,
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        h_channels = channels // 2
        self.branch_in, self.branch_out = branch_in, branch_out


        super(BU_Level, self).__init__()
        self.lateral = nn.ModuleList()

        for i in range(5):
            if i % 2 == 0:
                l = Static_ConvLayer(channels, h_channels, 1, Act=Act)
            else:
                l = Static_ConvLayer(h_channels, channels, 3, Act=Act)

            self.lateral.append(l)

        self.lateral.append(Static_ConvLayer(h_channels, channels, 3, Act=Act))


        if branch_out:
            if sizes and stride:
                self.downsample = nn.Sequential(Downsampler_Conv(sizes[0], sizes[1], h_channels, branch_channels, 3, stride, bias=False),
                                                nn.BatchNorm2d(branch_channels),
                                                Act)
            else:
                self.downsample = Static_ConvLayer(h_channels, branch_channels, 3, 2, Act=Act)


    def forward(self, f, b=None):
        if self.branch_in and b is not None:
            f = torch.cat((b, f), 1)

        for l in self.lateral[:-1]:
            f = l(f)

        if self.branch_out:
            b = self.downsample(f)
            f = self.lateral[-1](f)
            return f, b
        else:
            f = self.lateral[-1](f)
            return f



class Yolo_PAN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2004.10934

    The structure is based on the implementation;
    https://github.com/AlexeyAB/darknet
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 spp_kernels: list = None,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest',
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        self.num_levels = num_levels

        if len(channels) != num_levels:
            raise ValueError('make len(channels) == num_levels')

        if sizes:
            if len(sizes) != num_levels or len(strides) != num_levels - 1:
                raise ValueError('make len(sizes) == num_levels, and len(strides) == num_levels - 1')
        else:
            sizes, strides = num_levels * [None], (num_levels - 1) * [None]



        super(Yolo_PAN, self).__init__()

        self.top_down = nn.ModuleList()

        for i, c in enumerate(channels):
            up_size = sizes[i - 1] if i > 0 else None

            if i == num_levels - 1:
                self.top_down.append(TD_Level(c, False, True, channels[i - 1] // 2, spp_kernels, up_size, up_mode, Act))
            elif i > 0:
                self.top_down.append(TD_Level(c, True, True, channels[i - 1] // 2, None, up_size, up_mode, Act))
            else:
                self.top_down.append(TD_Level(c, True, False, None, None, up_size, up_mode, Act))


        self.bottom_up = nn.ModuleList()

        for i, c in enumerate(channels):
            if i == 0:
                l = nn.ModuleDict()
                l['lateral'] = Static_ConvLayer(c // 2, c, 3, Act=Act)

                if sizes[i] and strides[i]:
                    l['downsample'] = nn.Sequential(Downsampler_Conv(sizes[i], sizes[i + 1], c // 2, c, 3, strides[i], bias=False),
                                                    nn.BatchNorm2d(c),
                                                    Act)
                else:
                    l['downsample'] = Static_ConvLayer(c // 2, c, 3, 2, Act=Act)

                self.bottom_up.append(l)

            elif i < num_levels - 1:
                self.bottom_up.append(BU_Level(c, True, True, channels[i + 1] // 2, [sizes[i], sizes[i + 1]], strides[i], Act))
            else:
                self.bottom_up.append(BU_Level(c, True, False, None, None, None, Act))



    def forward(self, features):
        td_features = []
        b = None

        for i in range(self.num_levels - 1, -1, -1):

            if i == self.num_levels - 1:
                p, b = self.top_down[i](features[i])
                td_features.append(p)

            elif i > 0:
                p, b = self.top_down[i](features[i], b)
                td_features.append(p)

            else:
                p = self.top_down[i](features[i], b)
                td_features.append(p)

        td_features = td_features[::-1]


        bu_features = []

        for i in range(self.num_levels):

            if i == 0:
                bu_features.append(self.bottom_up[i].lateral(td_features[i]))
                b = self.bottom_up[i].downsample(td_features[i])

            elif i < self.num_levels - 1:
                p, b = self.bottom_up[i](td_features[i], b)
                bu_features.append(p)

            else:
                p = self.bottom_up[i](td_features[i], b)
                bu_features.append(p)

        return bu_features


