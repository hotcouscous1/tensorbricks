from layers import *


class ResNet_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.03385

    The structure is decribed in <Figure 5.(Left)> of the paper.
    """

    reduction = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU()):

        self.downsample = stride != 1 and in_channels != out_channels

        super(ResNet_Block, self).__init__()

        block = [Static_ConvLayer(in_channels, out_channels, kernel_size, stride, Act=Act),
                 Static_ConvLayer(out_channels, out_channels, kernel_size, 1, Act=None)]

        self.block = nn.Sequential(*block)

        if self.downsample:
            self.shortcut = Static_ConvLayer(in_channels, out_channels, 1, stride, Act=None)
        else:
            self.shortcut = nn.Identity()

        self.act = Act


    def forward(self, x):
        input = x
        x = self.block(x)
        x += self.shortcut(input)
        x = self.act(x)
        return x



class ResNet_BottleNeck_Block(nn.Module):
    """
    __version__ = 0.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.03385

    The structure is decribed in <Figure 5.(Right)> of the paper.

    'reduction' is # channels of 1x1 layer / # channels of 3x3 layer, according to
    'the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions'.
    """

    reduction = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU()):

        channels = int(out_channels / self.reduction)
        self.downsample = stride != 1 or in_channels != out_channels

        super(ResNet_BottleNeck_Block, self).__init__()

        block = [Static_ConvLayer(in_channels, channels, 1, 1, Act=Act),
                 Static_ConvLayer(channels, channels, kernel_size, stride, Act=Act),
                 Static_ConvLayer(channels, out_channels, 1, 1, Act=None)]

        self.block = nn.Sequential(*block)

        if self.downsample:
            self.shortcut = Static_ConvLayer(in_channels, out_channels, 1, stride, Act=None)
        else:
            self.shortcut = nn.Identity()

        self.act = Act


    def forward(self, x):
        input = x
        x = self.block(x)
        x += self.shortcut(input)
        x = self.act(x)
        return x



class Wide_ResNet_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1605.07146

    The structure is decribed in <Figure 1.(c), (d)> of the paper.

    If 'drop_rate' is None or 0, nn.Dropout is excluded from the block.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU(),
                 drop_rate: float = 0.3):

        self.downsample = stride != 1 or in_channels != out_channels

        super(Wide_ResNet_Block, self).__init__()

        block = [Dynamic_ConvLayer(in_channels, out_channels, kernel_size, 1, Act=Act, reverse='BAC'),
                 Dynamic_ConvLayer(out_channels, out_channels, kernel_size, stride, Act=Act, reverse='BAC')]

        if drop_rate:
            block.insert(1, nn.Dropout(drop_rate))

        self.block = nn.Sequential(*block)

        if self.downsample:
            self.shortcut = Static_ConvLayer(in_channels, out_channels, 1, stride, batch_norm=False, Act=None)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        input = x
        x = self.block(x)
        x += self.shortcut(input)
        return x



class ResNeXt_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1611.05431

    The structure is decribed in <Figure 3.(c)> of the paper.

    Concepts used in the paper are matched as follows;
    width of bottleneck -> group_channels
    width of group conv -> channels

    In the paper, width of bottleneck is proposed to be dependent on cardinality,
    but here, it is passed via another parameter for module-flexibility.

    In <Table 2.>, the relation between cardinality and width of bottleneck is;
    (1, 64), (2, 40), (4, 24), (8, 14), (32, 4)
    """

    reduction = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cardinality: int = 32,
                 group_channels: int = 4,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU()):

        channels = cardinality * group_channels
        self.downsample = stride != 1 or in_channels != out_channels

        super(ResNeXt_Block, self).__init__()

        block = [Static_ConvLayer(in_channels, channels, 1, 1, Act=Act),
                 Dynamic_ConvLayer(channels, channels, kernel_size, stride, groups=cardinality, Act=Act),
                 Static_ConvLayer(channels, out_channels, 1, 1, Act=None)]

        self.block = nn.Sequential(*block)

        if self.downsample:
            self.shortcut = Static_ConvLayer(in_channels, out_channels, 1, stride, Act=None)
        else:
            self.shortcut = nn.Identity()

        self.act = Act


    def forward(self, x):
        input = x
        x = self.block(x)
        x += self.shortcut(input)
        x = self.act(x)
        return x



class Res2Net_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1904.01169

    The structure is decribed in <Figure 2.(b)> of the paper.

    'base_channels' is adopted in the official implementation;
    https://github.com/Res2Net/Res2Net-PretrainedModels
    """

    reduction = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scales: int = 4,
                 base_channels: int = 26,
                 kernel_size: int = 3,
                 stride: int = 1,
                 Act: nn.Module = nn.ReLU(),
                 cardinality: int = None,
                 v1b: bool = False):

        if scales != 1:
            filters = scales - 1
        else:
            filters = scales

        if not cardinality:
            cardinality = 1

        channels = cardinality * int(math.floor(base_channels * ((out_channels / self.reduction) / 64)))

        self.channels, self.scales, self.filters = channels, scales, filters
        self.downsample = stride != 1 or in_channels != out_channels


        super(Res2Net_Block, self).__init__()

        self.pre_block = Static_ConvLayer(in_channels, scales * channels, 1, Act=Act)

        block = [Dynamic_ConvLayer(channels, channels, kernel_size, stride, groups=cardinality, Act=Act)
                 for _ in range(filters)]

        self.block = nn.ModuleList(block)

        self.post_block = Static_ConvLayer(scales * channels, out_channels, 1, Act=None)


        if self.downsample:
            self.pool = nn.AvgPool2d(3, stride, 1)

            if not v1b:
                self.shortcut = Static_ConvLayer(in_channels, out_channels, 1, stride, Act=None)
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                    Static_ConvLayer(in_channels, out_channels, 1, 1, Act=None))
        else:
            self.shortcut = nn.Identity()

        self.act = Act



    def forward(self, x):
        input = x

        x = self.pre_block(x)
        features = torch.split(x, self.channels, 1)

        if self.downsample:
            for i in range(self.filters):
                s = features[i]
                s = self.block[i](s)

                x = s if i == 0 else torch.cat((x, s), 1)

            if self.scales != 1:
                x = torch.cat((x, self.pool(features[self.filters])), 1)

        else:
            for i in range(self.filters):
                s = s + features[i] if i != 0 else features[i]
                s = self.block[i](s)

                x = s if i == 0 else torch.cat((x, s), 1)

            if self.scales != 1:
                x = torch.cat((x, features[self.filters]), 1)

        x = self.post_block(x)
        x += self.shortcut(input)
        x = self.act(x)
        return x



