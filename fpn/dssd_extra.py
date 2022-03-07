from layers import *
from fpn.ssd_extra import Extra_Res_Block

# The official implementation of DSSD is written in caffe.
# To the best of my knowledge, this is the closest implementation to the official.
# If you find an error, please let me know through Issues.


class Deconvolution_Module(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/1701.06659

    The structure is decribed in <Figure 3.> of the paper.
    """

    def __init__(self,
                 channels: int,
                 deconv_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 output_padding: int = 0,
                 Act: nn.Module = nn.ReLU()):

        super(Deconvolution_Module, self).__init__()

        self.lateral = nn.Sequential(Static_ConvLayer(channels, channels, Act=Act),
                                     Static_ConvLayer(channels, channels, Act=None))

        self.upsample = nn.Sequential(nn.ConvTranspose2d(deconv_channels, channels, kernel_size, stride, padding, output_padding),
                                      Static_ConvLayer(channels, channels, Act=None))
        self.act = Act


    def forward(self, f, deconv_f):
        f1 = self.lateral(f)
        f2 = self.upsample(deconv_f)

        f = f1 * f2
        f = self.act(f)
        return f



class DSSD_321_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/1701.06659

    The structure is based on the official implementation;
    https://github.com/chengyangfu/caffe
    """

    num_levels = 6

    def __init__(self,
                 channels: list,
                 Act: nn.Module = nn.ReLU()):

        c = channels

        if len(channels) != self.num_levels:
            raise ValueError('make len(channels) == 6 == num_levels')


        super(DSSD_321_Extra, self).__init__()

        self.bottom_up = nn.ModuleList([Extra_Res_Block(c[1], c[2], 2, 2, Act=Act),
                                        Extra_Res_Block(c[2], c[3], 2, 2, Act=Act),
                                        Extra_Res_Block(c[3], c[4], 3, 1, 0, Act=Act, shortcut=True),
                                        Extra_Res_Block(c[4], c[5], 3, 1, 0, Act=Act, shortcut=True)])

        self.top_down = nn.ModuleList([Deconvolution_Module(c[0], c[1], 2, 2, Act=Act),
                                       Deconvolution_Module(c[1], c[2], 2, 2, Act=Act),
                                       Deconvolution_Module(c[2], c[3], 2, 2, Act=Act),
                                       Deconvolution_Module(c[3], c[4], 3, 1, 0, Act=Act),
                                       Deconvolution_Module(c[4], c[5], 3, 1, 0, Act=Act)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 6, make len(features) == 2')

        bu_features = list(features)
        p = bu_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.bottom_up[i - 1](p)
            bu_features.append(p)


        td_features = [bu_features[-1]]

        for i in range(self.num_levels - 1, 0, -1):
            p = self.top_down[i - 1](bu_features[i - 1], td_features[-1])
            td_features.append(p)

        td_features = td_features[::-1]

        return td_features



class DSSD_513_Extra(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/1701.06659
    """

    num_levels = 7

    def __init__(self,
                 channels: list,
                 Act: nn.Module = nn.ReLU()):

        c = channels

        if len(channels) != self.num_levels:
            raise ValueError('make len(channels) == 7 == num_levels')


        super(DSSD_513_Extra, self).__init__()

        self.bottom_up = nn.ModuleList([Extra_Res_Block(c[1], c[2], 2, 2, Act=Act),
                                        Extra_Res_Block(c[2], c[3], 2, 2, Act=Act),
                                        Extra_Res_Block(c[3], c[4], 2, 2, Act=Act),
                                        Extra_Res_Block(c[4], c[5], 2, 2, Act=Act),
                                        Extra_Res_Block(c[5], c[6], 2, 2, Act=Act)])

        self.top_down = nn.ModuleList([Deconvolution_Module(c[0], c[1], 2, 2, Act=Act),
                                       Deconvolution_Module(c[1], c[2], 2, 2, Act=Act),
                                       Deconvolution_Module(c[2], c[3], 2, 2, Act=Act),
                                       Deconvolution_Module(c[3], c[4], 2, 2, Act=Act),
                                       Deconvolution_Module(c[4], c[5], 2, 2, Act=Act),
                                       Deconvolution_Module(c[5], c[6], 2, 1, Act=Act)])


    def forward(self, features):
        if len(features) != 2:
            raise RuntimeError('for levels to be 7, make len(features) == 2')

        bu_features = list(features)
        p = bu_features[-1]

        for i in range(1, self.num_levels - 1):
            p = self.bottom_up[i - 1](p)
            bu_features.append(p)


        td_features = [bu_features[-1]]

        for i in range(self.num_levels - 1, 0, -1):
            p = self.top_down[i - 1](bu_features[i - 1], td_features[-1])
            td_features.append(p)

        td_features = td_features[::-1]

        return td_features


