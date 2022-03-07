from layers import *


class Downsampler_Conv(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hot-couscous.

    This module is to adjust padding to get a desired feature size from the given size,
    and downsample a feature by nn.Conv2d.

    Parameters are identical to nn.Conv2d except padding, which is replaced with 'in_size' and 'out_size'.
    The size-error is raised under the same condition as nn.Conv2d.

    'stride' is not a ratio of input and output features, but parameter for nn.Conv2d.
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):

        super(Downsampler_Conv, self).__init__()

        padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)

        if padding < 0:
            raise ValueError('negative padding is not supported for Conv2d')

        if stride < 2:
            raise ValueError('downsampling stride must be greater than 1')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


    def forward(self, x):
        return self.conv(x)



class Downsampler_Pool(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hot-couscous.

    This module is created to adjust padding to get a desired feature size from the given size,
    and downsample a feature by nn.Maxpool2d or nn.AvgPool2d.

    Parameters are from nn.MaxPool2d and nn.AvgPool2d except padding, which is replaced with 'in_size' and 'out_size'.
    The size-error is raised under the same condition as nn.Conv2d.

    'stride' is not a ratio of input and output features, but parameter for pooling module of nn.
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 mode: str = 'maxpool',
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True):

        super(Downsampler_Pool, self).__init__()

        padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)

        if padding > kernel_size / 2:
            raise ValueError('pad should be smaller than half of kernel size in Pool2d')

        if stride < 2:
            raise ValueError('downsampling stride must be greater than 1')


        if mode == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)

        elif mode == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

        else:
            raise ValueError('please select the mode between maxpool and avgpool')


    def forward(self, x):
        return self.pool(x)



class FeatureFusion(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hotcouscous.

    This module is to fuse features in different modes.

    if 'softmax' is True, 'normalize' is not repeatably applied.
    """

    def __init__(self,
                 num: int,
                 mode: str = 'sum',
                 normalize: bool = True,
                 nonlinear: nn.Module = None,
                 softmax: bool = False):

        super(FeatureFusion, self).__init__()

        self.weight = nn.Parameter(torch.ones(num, dtype=torch.float32, device=device))
        self.mode = mode
        self.normalize = normalize
        self.nonlinear = nonlinear
        self.softmax = softmax


    def forward(self, features):
        weight = self.weight
        fusion = 0

        if self.nonlinear:
            weight = self.nonlinear(weight)

        if self.softmax:
            weight = weight.softmax(dim=0)

        if self.mode == 'sum':
            for w, f in zip(weight, features):
                fusion += w * f

        elif self.mode == 'mul':
            for w, f in zip(weight, features):
                fusion *= w * f

        elif self.mode == 'concat':
            features = [w * f for w, f in zip(weight, features)]
            fusion = torch.cat(features, dim=1)

        else:
            raise RuntimeError('select mode in sum, mul and concat')


        if self.normalize and not self.softmax:
            fusion /= (weight.sum() + 1e-4)

        return fusion
