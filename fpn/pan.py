from fpn.fpn import *
from fpn.utils import *


class BU_FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1803.01534

    The structure is decribed in <Figure 1.(b)>, <Figure 2.> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None):

        self.num_levels = num_levels

        if len(in_channels) != num_levels:
            raise ValueError('make len(in_channels) == num_levels')

        if sizes:
            if len(sizes) != num_levels or len(strides) != num_levels - 1:
                raise ValueError('make len(sizes) == num_levels, and len(strides) == num_levels - 1')


        super(BU_FPN, self).__init__()

        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        if sizes and strides:
            self.downsamples = nn.ModuleList([Downsampler_Conv(sizes[i], sizes[i + 1], out_channels, out_channels, 1, strides[i], bias=True)
                                              for i in range(len(sizes) - 1)])
        else:
            self.downsamples = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1, 2, padding=0, bias=True)
                                              for _ in range(num_levels - 1)])

        self.fuses = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) for _ in range(num_levels)])


    def forward(self, features):
        p_features = []

        for i in range(self.num_levels):
            p = self.laterals[i](features[i])

            if p_features:
                d = self.downsamples[i - 1](p_features[-1])
                p += d

            p = self.fuses[i](p)
            p_features.append(p)

        return p_features



class PAN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1803.01534

    The structure is decribed in <Figure 1.(a), (b)> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest'):

        super(PAN, self).__init__()

        self.top_down = FPN(num_levels, in_channels, out_channels, sizes, up_mode)
        self.bottom_up = BU_FPN(num_levels, len(in_channels) * [out_channels], out_channels, sizes, strides)


    def forward(self, features):
        features = self.top_down(features)
        features = self.bottom_up(features)

        return features
