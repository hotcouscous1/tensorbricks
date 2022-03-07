from fpn.fpn import *
from fpn.utils import *


class RetinaNet_FPN(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1708.02002

    The structure is decribed in <4. RetinaNet Detector> of the paper.
    """

    def __init__(self,
                 num_in: int,
                 num_out: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: list = None,
                 strides: list = None,
                 up_mode: str = 'nearest',
                 Act: nn.Module = nn.ReLU()):

        self.num_in, self.num_out = num_in, num_out

        if len(in_channels) != num_in:
            raise ValueError('make len(in_channels) == num_in')

        if sizes:
            if len(sizes) != num_out or len(strides) != num_out - 1:
                raise ValueError('make len(sizes) == num_out, and len(strides) == num_out - 1')


        super(RetinaNet_FPN, self).__init__()

        if sizes:
            self.fpn = FPN(num_in, in_channels, out_channels, sizes[:num_in], up_mode)
        else:
            self.fpn = FPN(num_in, in_channels, out_channels, sizes, up_mode)

        extra = []

        for i in range(num_in, num_out):
            if i == num_in:
                if sizes and strides:
                    extra.append(Downsampler_Conv(sizes[i - 1], sizes[i], in_channels[-1], out_channels, 3, strides[i - 1]))
                else:
                    extra.append(nn.Conv2d(in_channels[-1], out_channels, 3, 2, 1))

            else:
                if sizes and strides:
                    extra.append(nn.Sequential(Act, Downsampler_Conv(sizes[i - 1], sizes[i], out_channels, out_channels, 3, strides[i - 1])))
                else:
                    extra.append(nn.Sequential(Act, nn.Conv2d(out_channels, out_channels, 3, 2, 1)))

        self.extra = nn.ModuleList(extra)


    def forward(self, features):
        p_features = self.fpn(features)

        for i in range(self.num_in, self.num_out):
            f = features[-1] if i == self.num_in else p_features[-1]

            p = self.extra[i - self.num_in](f)
            p_features.append(p)

        return p_features
