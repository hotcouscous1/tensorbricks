from layers import *
from utils import *


class Mobile_V1_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1704.04861

    The structure is decribed in <Figure 3.(Right)> of the paper.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU()):

        super(Mobile_V1_Block, self).__init__()

        self.dw_layer = nn.Sequential(Depthwise_Conv2d(in_channels, kernel_size, stride),
                                      nn.BatchNorm2d(in_channels),
                                      Act)

        self.pw_layer = nn.Sequential(Pointwise_Conv2d(in_channels, out_channels),
                                      nn.BatchNorm2d(out_channels),
                                      Act)

    def forward(self, x):
        x = self.dw_layer(x)
        x = self.pw_layer(x)
        return x



class Mobile_V2_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381

    The structure is decribed in <Table 1.>, <Figure 3.(b)> of the paper.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU6()):

        expand_channels = round_width(expansion * in_channels, 1)

        self.expansion = expansion
        self.shortcut = (stride == 1) and (in_channels == out_channels)

        super(Mobile_V2_Block, self).__init__()

        if expansion != 1:
            self.pw_layer = nn.Sequential(Pointwise_Conv2d(in_channels, expand_channels),
                                          nn.BatchNorm2d(expand_channels),
                                          Act)

        self.dw_layer = nn.Sequential(Depthwise_Conv2d(expand_channels, kernel_size, stride),
                                      nn.BatchNorm2d(expand_channels),
                                      Act)

        self.pw_linear = nn.Sequential(Pointwise_Conv2d(expand_channels, out_channels),
                                       nn.BatchNorm2d(out_channels))

    def forward(self, x):
        input = x
        if self.expansion != 1:
            x = self.pw_layer(x)

        x = self.dw_layer(x)
        x = self.pw_linear(x)

        if self.shortcut:
            x += input
        return x



class Mobile_NAS_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1807.11626

    The structure is decribed in <Figure 7.(b), (c)> of the paper.

    'se_ratio' is a denominator of squeezing in Squeeze_Excitation_Conv.
    'survival_prob' is for stochastic depth, which is adopted by EfficientNet.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU(),
                 se_ratio: float = None,
                 survival_prob:float = None):

        expand_channels = expansion * in_channels

        self.expansion, self.se_ratio, self.survival_prob = expansion, se_ratio, survival_prob
        self.shortcut = (stride == 1) and (in_channels == out_channels)

        super(Mobile_NAS_Block, self).__init__()

        if expansion != 1:
            self.pw_layer = nn.Sequential(Pointwise_Conv2d(in_channels, expand_channels),
                                          nn.BatchNorm2d(expand_channels),
                                          Act)

        self.dw_layer = nn.Sequential(Depthwise_Conv2d(expand_channels, kernel_size, stride),
                                      nn.BatchNorm2d(expand_channels),
                                      Act)
        if se_ratio:
            self.se = Squeeze_Excitation_Conv(expand_channels, expand_channels, expansion * se_ratio, Act=Act)

        self.pw_linear = nn.Sequential(Pointwise_Conv2d(expand_channels, out_channels),
                                       nn.BatchNorm2d(out_channels))

    def forward(self, x):
        input = x
        if self.expansion != 1:
            x = self.pw_layer(x)

        x = self.dw_layer(x)
        if self.se_ratio:
            x = self.se(x)
        x = self.pw_linear(x)

        if self.shortcut:
            if self.training:
                if self.survival_prob:
                    x = stochastic_depth(x, self.survival_prob, self.training)
            x += input

        return x



class Mobile_V3_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244

    The structure is decribed in <Figure 4.> of the paper.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU(),
                 se_ratio: float = None):

        expand_channels = round_width(expansion * in_channels)

        self.expansion, self.se_ratio = expansion, se_ratio
        self.shortcut = (stride == 1) and (in_channels == out_channels)

        super(Mobile_V3_Block, self).__init__()

        if expansion != 1:
            self.pw_layer = nn.Sequential(Pointwise_Conv2d(in_channels, expand_channels),
                                          nn.BatchNorm2d(expand_channels),
                                          Act)

        self.dw_layer = nn.Sequential(Depthwise_Conv2d(expand_channels, kernel_size, stride),
                                      nn.BatchNorm2d(expand_channels),
                                      Act)
        if se_ratio:
            self.se = Squeeze_Excitation_Conv(expand_channels, expand_channels, se_ratio, False,
                                              nn.ReLU(), H_Sigmoid(), divisor=8)

        self.pw_linear = nn.Sequential(Pointwise_Conv2d(expand_channels, out_channels),
                                       nn.BatchNorm2d(out_channels))

    def forward(self, x):
        input = x
        if self.expansion != 1:
            x = self.pw_layer(x)

        x = self.dw_layer(x)
        if self.se_ratio:
            x = self.se(x)
        x = self.pw_linear(x)

        if self.shortcut:
            x += input
        return x



class Mobile_ReX_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2007.00992

    The structure is based on Mobile_V2_Block according to the paper.

    For non-linearity, ReLU6 remains after the no-expansion layer,
    but it is replaced with SiLU after the expansion layer ('pw_layer').

    Non-linearity of 'dw_layer' comes after the squeeze-excitation, according to the official implementation;
    https://github.com/clovaai/rexnet
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 se_ratio: float = 2):

        expand_channels = expansion * in_channels

        self.expansion, self.se_ratio, self.in_channels = expansion, se_ratio, in_channels
        self.shortcut = (stride == 1) and (in_channels <= out_channels)

        super(Mobile_ReX_Block, self).__init__()

        if expansion != 1:
            self.pw_layer = nn.Sequential(Pointwise_Conv2d(in_channels, expand_channels),
                                          nn.BatchNorm2d(expand_channels),
                                          nn.SiLU())

        self.dw_layer = nn.Sequential(Depthwise_Conv2d(expand_channels, kernel_size, stride),
                                      nn.BatchNorm2d(expand_channels))

        if se_ratio:
            self.se = Squeeze_Excitation_Conv(expand_channels, expand_channels, expansion * se_ratio, True, nn.ReLU())

        self.act = nn.ReLU6()

        self.pw_linear = nn.Sequential(Pointwise_Conv2d(expand_channels, out_channels),
                                       nn.BatchNorm2d(out_channels))

    def forward(self, x):
        input = x
        if self.expansion != 1:
            x = self.pw_layer(x)

        x = self.dw_layer(x)
        if self.se_ratio:
            x = self.se(x)
        x = self.act(x)
        x = self.pw_linear(x)

        if self.shortcut:
            x[:, :self.in_channels] += input

        return x
