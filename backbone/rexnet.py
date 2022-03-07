from layers import *
from backbone.block.mobile import Mobile_ReX_Block
from utils import load_pretrained


class ReXNet_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2007.00992
    checkpoints : https://github.com/clovaai/rexnet

    The structure is decribed in <Figure A2.(a)> of the paper.

    'config_channels' returns a group of parameterized channels in double list,
    which # outer list is # stages after 'stage0', and # inner list is # blocks + 1.
    """

    Block = Mobile_ReX_Block

    @staticmethod
    def config_channels(stem_channels: int,
                        start_channels: int,
                        end_channels: int,
                        depths: list,
                        width_mult: float = 1.0):

        num_blocks = sum(depths)

        stage_widths = []
        block_widths = [stem_channels, round(start_channels * max(width_mult, 1))]

        channels = start_channels if width_mult >= 1 else start_channels / width_mult

        for i in range(num_blocks - 1):
            channels += (end_channels / num_blocks)
            block_widths.append(round(channels * width_mult))

        block = 0
        for i in depths:
            start = block
            block += i
            end = block

            stage_widths.append(block_widths[start:end + 1])

        return stage_widths



    def __init__(self,
                 start_channels: int,
                 end_channels: int,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0):

        stem_channels = round(32 * max(width_mult, 1))
        last_channels = int(1280 * width_mult)

        d = [math.ceil(i * depth_mult) for i in [1, 2, 2, 3, 3, 5]]

        c = self.config_channels(stem_channels, start_channels, end_channels, d, width_mult)
        self.widths = [stem_channels] + [i[-1] for i in c] + [last_channels]


        super(ReXNet_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, stem_channels, stride=2, Act=nn.SiLU())

        self.stage1 = self.Stage(d[0], c[0], 1, 3, 1, None)
        self.stage2 = self.Stage(d[1], c[1], 6, 3, 2, None)
        self.stage3 = self.Stage(d[2], c[2], 6, 3, 2, 2)
        self.stage4 = self.Stage(d[3], c[3], 6, 3, 2, 2)
        self.stage5 = self.Stage(d[4], c[4], 6, 3, 1, 2)
        self.stage6 = self.Stage(d[5], c[5], 6, 3, 2, 2)

        self.conv_last = Static_ConvLayer(c[-1][-1], last_channels, 1, Act=nn.SiLU())



    def Stage(self, num_blocks, channels, expansion, kernel_size, stride, se_ratio):
        if not num_blocks == len(channels) - 1:
            raise ValueError('each block must be assigned different channels')

        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(channels[0], channels[1], expansion, kernel_size, stride, se_ratio)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels[i], channels[i+1], expansion, kernel_size, 1, se_ratio)

        blocks = nn.Sequential(blocks)
        return blocks



    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.conv_last(x)

        return x




def rexnet_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 1, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_backbone')
    return model


def rexnet_3_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 3, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_3_backbone')
    return model


def rexnet_22_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 2.2, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_22_backbone')
    return model


def rexnet_2_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 2, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_2_backbone')
    return model


def rexnet_15_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 1.5, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_15_backbone')
    return model


def rexnet_13_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 1.3, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_13_backbone')
    return model


def rexnet_09_backbone(pretrained=False):
    model = ReXNet_Backbone(16, 180, 0.9, 1)
    if pretrained:
        load_pretrained(model, 'rexnet_09_backbone')
    return model

