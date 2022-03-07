from backbone.block.residual import *
from utils import load_pretrained


class ResNet_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.03385
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    The structure is decribed in <Table 1.>, <Figure 3.(Right)> of the paper.

    'num_blocks' receives a list of # layers for each stage, instead of total # layers, for flexibility.
    """

    def __init__(self,
                 num_blocks: list,
                 bottleneck: bool,
                 Act: nn.Module = nn.ReLU()):

        if len(num_blocks) != 4:
            raise ValueError('num_blocks should be list of length 4')

        Block = ResNet_Block if not bottleneck \
            else ResNet_BottleNeck_Block

        self.widths = c = [64] + [Block.reduction * i for i in [64, 128, 256, 512]]


        super(ResNet_Backbone, self).__init__()

        self.stage0 = nn.Sequential(Static_ConvLayer(3, c[0], kernel_size=7, stride=2, Act=Act),
                                    nn.MaxPool2d(3, 2, 1))

        self.stage1 = self.Stage(Block, num_blocks[0], c[0], c[1], 3, 1, Act)
        self.stage2 = self.Stage(Block, num_blocks[1], c[1], c[2], 3, 2, Act)
        self.stage3 = self.Stage(Block, num_blocks[2], c[2], c[3], 3, 2, Act)
        self.stage4 = self.Stage(Block, num_blocks[3], c[3], c[4], 3, 2, Act)


    def Stage(self, Block, num_blocks, in_channels, channels, kernel_size, stride, Act):
        blocks = OrderedDict()
        blocks['block' + str(0)] = Block(in_channels, channels, kernel_size, stride, Act)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = Block(channels, channels, kernel_size, 1, Act)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x



class Wide_ResNet_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1605.07146

    The structure is decribed in <Table 1.> of the paper.

    The implementation of pytorch is a modification of ResNet, built with bottelneck blocks.

    Symbols used in the paper are matched as follows;
    k -> width_mult
    N -> num_blocks
    """

    Block = Wide_ResNet_Block

    def __init__(self,
                 num_blocks: int,
                 width_mult: int,
                 Act: nn.Module = nn.ReLU(),
                 drop_rate: float = 0.3):

        if (num_blocks - 4) % 6 == 0:
            num_blocks = (num_blocks - 4) // 6
        else:
            raise ValueError('num_blocks should be 6n + 4, where n is a integer')

        self.widths = c = [16] + [width_mult * i for i in [16, 32, 64]]


        super(Wide_ResNet_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], batch_norm=False, Act=None)

        self.stage1 = self.Stage(num_blocks, c[0], c[1], 3, 1, Act, drop_rate)
        self.stage2 = self.Stage(num_blocks, c[1], c[2], 3, 2, Act, drop_rate)
        self.stage3 = self.Stage(num_blocks, c[2], c[3], 3, 2, Act, drop_rate)

        self.bn_last = nn.Sequential(nn.BatchNorm2d(c[3]), Act)


    def Stage(self, num_blocks, in_channels, channels, kernel_size, stride, Act, drop_rate):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, kernel_size, stride, Act, drop_rate)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, kernel_size, 1, Act, drop_rate)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bn_last(x)

        return x



class ResNeXt_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1611.05431
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    The structure is decribed in <Table 1.(Right)> of the paper.

    Symbols used in the paper are matched as follows;
    C -> cardinality
    d -> base_group_channels
    """

    Block = ResNeXt_Block

    def __init__(self,
                 num_blocks: list,
                 cardinality: int,
                 base_group_channels: int,
                 Act: nn.Module = nn.ReLU()):

        group_channels = [base_group_channels * i for i in [1, 2, 4, 8]]

        self.widths = c = [64] + [self.Block.reduction * i for i in [64, 128, 256, 512]]


        super(ResNeXt_Backbone, self).__init__()

        self.stage0 = nn.Sequential(Static_ConvLayer(3, c[0], kernel_size=7, stride=2, Act=Act),
                                    nn.MaxPool2d(3, 2, 1))

        self.stage1 = self.Stage(num_blocks[0], c[0], c[1], cardinality, group_channels[0], 3, 1, Act)
        self.stage2 = self.Stage(num_blocks[1], c[1], c[2], cardinality, group_channels[1], 3, 2, Act)
        self.stage3 = self.Stage(num_blocks[2], c[2], c[3], cardinality, group_channels[2], 3, 2, Act)
        self.stage4 = self.Stage(num_blocks[3], c[3], c[4], cardinality, group_channels[3], 3, 2, Act)


    def Stage(self, num_blocks, in_channels, channels, cardinality, group_channels, kernel_size, stride, Act):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, cardinality, group_channels, kernel_size, stride, Act)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, cardinality, group_channels, kernel_size, 1, Act)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x



class Res2Net_Backbone(nn.Module):
    """
    __version__ = 0.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1904.01169
    checkpoints : https://github.com/Res2Net/Res2Net-PretrainedModels

    The structure is based on Resnet according to the official implementation.

    The concept of 'v1b' is explained in the paper;
    https://arxiv.org/abs/1812.01187
    """

    Block = Res2Net_Block

    def __init__(self,
                 num_blocks: list,
                 scales: int,
                 base_channels: int,
                 Act: nn.Module = nn.ReLU(),
                 cardinality: int = None,
                 v1b: bool = False):

        self.widths = c = [64] + [self.Block.reduction * i for i in [64, 128, 256, 512]]


        super(Res2Net_Backbone, self).__init__()

        if not v1b:
            self.stage0 = nn.Sequential(Static_ConvLayer(3, c[0], kernel_size=7, stride=2, Act=Act),
                                        nn.MaxPool2d(3, 2, 1))
        else:
            self.stage0 = nn.Sequential(Static_ConvLayer(3, 32, 3, stride=2, Act=Act),
                                        Static_ConvLayer(32, 32, 3, Act=Act),
                                        Static_ConvLayer(32, c[0], 3, Act=Act),
                                        nn.MaxPool2d(3, 2, 1))

        self.stage1 = self.Stage(num_blocks[0], c[0], c[1], scales, base_channels, 3, 1, Act, cardinality, v1b)
        self.stage2 = self.Stage(num_blocks[1], c[1], c[2], scales, base_channels, 3, 2, Act, cardinality, v1b)
        self.stage3 = self.Stage(num_blocks[2], c[2], c[3], scales, base_channels, 3, 2, Act, cardinality, v1b)
        self.stage4 = self.Stage(num_blocks[3], c[3], c[4], scales, base_channels, 3, 2, Act, cardinality, v1b)


    def Stage(self, num_blocks, in_channels, channels, scales, base_channels, kernel_size, stride, Act, cardinality, v1b):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, scales, base_channels, kernel_size, stride, Act, cardinality, v1b)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, scales, base_channels, kernel_size, 1, Act, cardinality, v1b)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x




def resnet_18_backbone(pretrained=False):
    model = ResNet_Backbone([2, 2, 2, 2], False, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnet_18_backbone')
    return model


def resnet_34_backbone(pretrained=False):
    model = ResNet_Backbone([3, 4, 6, 3], False, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnet_34_backbone')
    return model


def resnet_50_backbone(pretrained=False):
    model = ResNet_Backbone([3, 4, 6, 3], True, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnet_50_backbone')
    return model


def resnet_101_backbone(pretrained=False):
    model = ResNet_Backbone([3, 4, 23, 3], True, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnet_101_backbone')
    return model


def resnet_152_backbone(pretrained=False):
    model = ResNet_Backbone([3, 8, 36, 3], True, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnet_152_backbone')
    return model


def wide_resnet_28_10_backbone(pretrained=False):
    model = Wide_ResNet_Backbone(28, 20, nn.ReLU(), 0.3)
    if pretrained:
        load_pretrained(model, 'wide_resnet_28_10_backbone')
    return model


def wide_resnet_40_4_backbone(pretrained=False):
    model = Wide_ResNet_Backbone(40, 4, nn.ReLU(), 0.3)
    if pretrained:
        load_pretrained(model, 'wide_resnet_40_4_backbone')
    return model


def resnext_50_32x4d_backbone(pretrained=False):
    model = ResNeXt_Backbone([3, 4, 6, 3], 32, 4, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnext_50_32x4d_backbone')
    return model


def resnext_101_32x4d_backbone(pretrained=False):
    model = ResNeXt_Backbone([3, 4, 23, 3], 32, 4, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnext_101_32x4d_backbone')
    return model


def resnext_101_32x8d_backbone(pretrained=False):
    model = ResNeXt_Backbone([3, 4, 23, 3], 32, 8, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnext_101_32x8d_backbone')
    return model


def resnext_101_64x4d_backbone(pretrained=False):
    model = ResNeXt_Backbone([3, 4, 23, 3], 64, 4, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'resnext_101_64x4d_backbone')
    return model


def res2net_50_26w_4s_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 6, 3], 4, 26, nn.ReLU(), None)
    if pretrained:
        load_pretrained(model, 'res2net_50_26w_4s_backbone')
    return model


def res2net_50_14w_8s_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 6, 3], 8, 14, nn.ReLU(), None)
    if pretrained:
        load_pretrained(model, 'res2net_50_14w_8s_backbone')
    return model


def res2net_50_26w_8s_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 6, 3], 8, 26, nn.ReLU(), None)
    if pretrained:
        load_pretrained(model, 'res2net_50_26w_8s_backbone')
    return model


def res2net_101_26w_4s_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 23, 3], 4, 26, nn.ReLU(), None)
    if pretrained:
        load_pretrained(model, 'res2net_101_26w_4s_backbone')
    return model


def res2next_50_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 6, 3], 4, 4, nn.ReLU(), 8)
    if pretrained:
        load_pretrained(model, 'res2next_50_backbone')
    return model


def res2net_50_v1b_26w_4s_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 6, 3], 4, 26, nn.ReLU(), None, True)
    if pretrained:
        load_pretrained(model, 'res2net_50_v1b_26w_4s_backbone')
    return model


def res2net_101_v1b_26w_4s_backbone(pretrained=False):
    model = Res2Net_Backbone([3, 4, 23, 3], 4, 26, nn.ReLU(), None, True)
    if pretrained:
        load_pretrained(model, 'res2net_101_v1b_26w_4s_backbone')
    return model
