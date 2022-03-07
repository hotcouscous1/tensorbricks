from layers import *


class DarkNet_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767

    The structure is decribed in <Table 1.> of the paper.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 reduction: int = 2,
                 shortcut: bool = True,
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        channels = int(out_channels / reduction)
        self.shortcut = shortcut and (in_channels == out_channels)

        super(DarkNet_Block, self).__init__()

        block = [Static_ConvLayer(in_channels, channels, 1, Act=Act),
                 Static_ConvLayer(channels, out_channels, 3, Act=Act)]

        self.block = nn.Sequential(*block)


    def forward(self, x):
        input = x
        x = self.block(x)

        if self.shortcut:
            x += input
        return x



class CSP_DarkNet_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.11929

    The structure is decribed in <Figure 3. (b)> of the paper.
    """

    Block = DarkNet_Block

    def __init__(self,
                 num_blocks: int,
                 in_channels: int,
                 out_channels: int,
                 half: bool = True,
                 block_reduction: int = 1,
                 Act: nn.Module = Mish()):

        if half:
            channels = int(out_channels / 2)
            cat_channels = out_channels
        else:
            channels = out_channels
            cat_channels = 2 * out_channels


        super(CSP_DarkNet_Block, self).__init__()

        self.part1 = Static_ConvLayer(in_channels, channels, 1, Act=Act)
        self.part2 = Static_ConvLayer(in_channels, channels, 1, Act=Act)

        dense = [self.Block(channels, channels, block_reduction, True, Act) for _ in range(num_blocks)]
        self.dense = nn.Sequential(*dense)

        self.trans1 = Static_ConvLayer(channels, channels, 1, Act=Act)
        self.trans2 = Static_ConvLayer(cat_channels, out_channels, 1, Act=Act)


    def forward(self, x):
        x1 = self.part1(x)

        x2 = self.part2(x)
        x2 = self.dense(x2)
        x2 = self.trans1(x2)

        x = torch.cat((x2, x1), dim=1)
        x = self.trans2(x)
        return x



class CSP_DarkNet_Tiny_Block(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2011.08036

    The structure is decribed in <Figure 3.> of the paper.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        self.h_channels = int(out_channels / 2)

        super(CSP_DarkNet_Tiny_Block, self).__init__()

        self.part1 = Static_ConvLayer(in_channels, out_channels, 3, Act=Act)

        self.part2_1 = Static_ConvLayer(self.h_channels, self.h_channels, 3, Act=Act)
        self.part2_2 = Static_ConvLayer(self.h_channels, self.h_channels, 3, Act=Act)

        self.trans = Static_ConvLayer(out_channels, out_channels, 1, Act=Act)


    def forward(self, x):
        x1 = self.part1(x)

        x2 = torch.split(x1, self.h_channels, 1)[0]
        x2_1 = self.part2_1(x2)
        x2_2 = self.part2_2(x2_1)

        x2 = torch.cat((x2_2, x2_1), 1)
        x2 = self.trans(x2)

        x = torch.cat((x1, x2), 1)
        return x
