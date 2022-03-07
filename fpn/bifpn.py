from fpn.utils import *


class Fusion(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070
    """

    def __init__(self,
                 num: int,
                 mode: str = 'fast'):

        super(Fusion, self).__init__()

        if mode == 'unbound':
            self.fusion = FeatureFusion(num, 'sum', normalize=False)

        elif mode == 'bound':
            self.fusion = FeatureFusion(num, 'sum', normalize=True)

        elif mode == 'softmax':
            self.fusion = FeatureFusion(num, 'sum', softmax=True)

        elif mode == 'fast':
            self.fusion = FeatureFusion(num, 'sum', normalize=True, nonlinear=nn.ReLU())

        else:
            raise ValueError('please select mode in unbound, bound, softmax, fast')

    def forward(self, features):
        return self.fusion(features)



class Resample_FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070

    The structure is based on the official implementation;
    https://github.com/google/automl/tree/master/efficientdet

    It is to adjust levels and channels of output features.
    It is slightly different from RetinaNet_FPN.
    """

    def __init__(self,
                 num_in: int,
                 num_out: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None):

        self.num_in, self.num_out = num_in, num_out

        if len(in_channels) != num_in:
            raise ValueError('make len(in_channels) == num_in')

        if sizes:
            if len(sizes) != num_out or len(strides) != num_out - 1:
                raise ValueError('make len(sizes) == num_out, and len(strides) == num_out - 1')


        super(Resample_FPN, self).__init__()

        levels = []

        for i in range(num_in):
            levels.append(Static_ConvLayer(in_channels[i], out_channels, 1, bias=True, Act=None))

        for i in range(num_in, num_out):
            if i == num_in:
                if sizes and strides:
                    levels.append(nn.Sequential(Static_ConvLayer(in_channels[-1], out_channels, 1, bias=True, Act=None),
                                                Downsampler_Pool(sizes[i - 1], sizes[i], 'maxpool', 3, strides[i - 1])))
                else:
                    levels.append(nn.Sequential(Static_ConvLayer(in_channels[-1], out_channels, 1, bias=True, Act=None),
                                                nn.MaxPool2d(3, 2, 1)))
            else:
                if sizes and strides:
                    levels.append(Downsampler_Pool(sizes[i - 1], sizes[i], 'maxpool', 3, strides[i - 1]))
                else:
                    levels.append(nn.MaxPool2d(3, 2, 1))

        self.levels = nn.ModuleList(levels)


    def forward(self, features):
        p_features = []

        for i, f in enumerate(features):
            p = self.levels[i](f)
            p_features.append(p)

        for i in range(self.num_in, self.num_out):
            f = features[-1] if i == self.num_in else p_features[-1]

            p = self.levels[i](f)
            p_features.append(p)

        return p_features



class _BiFPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070

    The structure is decribed in <Figure 2.(d)> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest',
                 fusion: str = 'fast',
                 Act: nn.Module = nn.SiLU()):

        self.num_levels = num_levels
        self.first = in_channels != num_levels * [out_channels]

        if sizes:
            if len(sizes) != num_levels or len(strides) != num_levels - 1:
                raise ValueError('make len(sizes) == num_levels, and len(strides) == num_levels - 1')


        super(_BiFPN, self).__init__()

        if self.first:
            self.resample = Resample_FPN(len(in_channels), num_levels, in_channels, out_channels, sizes, strides)

            self.branches = nn.ModuleList([Static_ConvLayer(c, out_channels, 1, bias=True, Act=None)
                                           for c in in_channels[1: len(in_channels)]])

        if sizes:
            self.upsamples = nn.ModuleList([nn.Upsample(size=size, mode=up_mode) for size in sizes[:-1]])
        else:
            self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode=up_mode) for _ in range(num_levels - 1)])


        if sizes and strides:
            self.downsamples = nn.ModuleList([Downsampler_Pool(sizes[i], sizes[i + 1], 'maxpool', 3, strides[i])
                                             for i in range(num_levels - 1)])
        else:
            self.downsamples = nn.ModuleList([nn.MaxPool2d(3, 2, 1) for _ in range(num_levels - 1)])


        self.td_fuses = nn.ModuleList([self.fuse(2, fusion, out_channels, Act) for _ in range(num_levels - 1)])

        self.bu_fuses = nn.ModuleList([self.fuse(3, fusion, out_channels, Act) for _ in range(num_levels - 2)])
        self.bu_fuses.append(self.fuse(2, fusion, out_channels, Act))


    @staticmethod
    def fuse(num, mode, channels, Act):
        layer = [Fusion(num, mode),
                 Act,
                 Seperable_Conv2d(channels, channels, 3, 1, bias=True),
                 nn.BatchNorm2d(channels)]

        return nn.Sequential(*layer)



    def forward(self, features):
        td_features, bu_features = [], []

        # resample
        if not self.first:
            branches = features[1: -1]
        else:
            branches = []
            for i, b in enumerate(self.branches):
                branches.append(b(features[i + 1]))

            features = self.resample(features)
            branches = branches + features[len(branches) + 1: -1]


        # top-down path
        for i in range(self.num_levels - 1, -1, -1):
            if i == len(features) - 1:
                td_features.append(features[i])

            else:
                u = self.upsamples[i](td_features[-1])
                p = self.td_fuses[i]([features[i], u])

                td_features.append(p)

        td_features = td_features[::-1]


        # bottom-up path
        for i in range(self.num_levels):
            if i == 0:
                bu_features.append(td_features[i])
            else:
                d = self.downsamples[i - 1](bu_features[-1])

                if i != len(td_features) - 1:
                    p = self.bu_fuses[i - 1]([d, td_features[i], branches[i - 1]])
                else:
                    p = self.bu_fuses[i - 1]([d, td_features[i]])

                bu_features.append(p)

        return bu_features




class BiFPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070

    The structure is decribed in <Figure 3.> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 num_repeat: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest',
                 fusion: str = 'fast',
                 Act: nn.Module = nn.SiLU()):

        super(BiFPN, self).__init__()

        fpn = [_BiFPN(num_levels, in_channels, out_channels, sizes, strides, up_mode, fusion, Act)]

        for i in range(num_repeat - 1):
            fpn.append(_BiFPN(num_levels, num_levels * [out_channels], out_channels, sizes, strides, up_mode, fusion, Act))

        self.fpn = nn.ModuleList(fpn)


    def forward(self, features):
        for fpn in self.fpn:
            features = fpn(features)

        return features
