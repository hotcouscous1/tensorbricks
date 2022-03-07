from layers import *

# Differences of this implementation are,
# 1) you can take as many levels as you want
# 2) you can give feature sizes at each level, which make free from stride=2
#
# In top-down fpn modules, if 'sizes' is not given, 'scale_factor' of every upsampling are set to 2.
# In bottom-up fpn modules, if 'sizes' and 'strides' are not given, 'stride' of every downsampling are set to 2.


class FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1612.03144

    The structure is decribed in <Figure 3.> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 up_mode: str = 'nearest'):

        self.num_levels = num_levels

        if len(in_channels) != num_levels:
            raise ValueError('make len(in_channels) == num_levels')

        if sizes:
            if len(sizes) != num_levels:
                raise ValueError('make len(sizes) == num_levels')


        super(FPN, self).__init__()

        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        if sizes:
            self.upsamples = nn.ModuleList([nn.Upsample(size=size, mode=up_mode)
                                            for size in sizes[:-1]])
        else:
            self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode=up_mode)
                                            for _ in range(num_levels - 1)])

        self.fuses = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) for _ in range(num_levels)])


    def forward(self, features: list):
        p_features = []

        for i in range(self.num_levels - 1, -1, -1):
            p = self.laterals[i](features[i])

            if p_features:
                u = self.upsamples[i](p_features[-1])
                p += u

            p_features.append(p)

        p_features = p_features[::-1]
        p_features = [f(p) for f, p in zip(self.fuses, p_features)]

        return p_features
