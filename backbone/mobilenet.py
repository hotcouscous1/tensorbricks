from backbone.block.mobile import *
from utils import load_pretrained


class MobileNet_V1_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1704.04861

    The structure is decribed in <Table 1.> of the paper.

    Symbols used in the paper are matched as follows;
    a(alpha) -> width_mult
    """

    Block = Mobile_V1_Block

    def __init__(self,
                 width_mult: float = 1.0,
                 Act: nn.Module = nn.ReLU()):

        self.widths = c = [round_width(i * width_mult, 8)
                           for i in [32, 64, 128, 256, 512, 1024]]


        super(MobileNet_V1_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], stride=2, Act=Act)

        self.stage1 = self.Stage(1, c[0], c[1], 3, 1, Act)
        self.stage2 = self.Stage(2, c[1], c[2], 3, 2, Act)
        self.stage3 = self.Stage(2, c[2], c[3], 3, 2, Act)
        self.stage4 = self.Stage(6, c[3], c[4], 3, 2, Act)
        self.stage5 = self.Stage(2, c[4], c[5], 3, 2, Act)


    def Stage(self, num_blocks, in_channels, channels, kernel_size, stride, Act):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, kernel_size, stride, Act)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, kernel_size, 1, Act)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        return x



class MobileNet_V2_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381
    checkpoints : https://github.com/d-li14/mobilenetv2.pytorch

    The structure is decribed in <Table 2.> of the paper.

    Symbols used in the paper are matched as follows;
    t -> expansion
    c -> c
    n -> num_blocks
    s -> stride

    Width multiplier for the last channels, unlike V1, is applied only if it is greater than 1.
    """

    Block = Mobile_V2_Block

    def __init__(self,
                 width_mult: float = 1.0,
                 Act: nn.Module = nn.ReLU6()):

        self.widths = c = [round_width(i * width_mult, 8) for i in [32, 16, 24, 32, 64, 96, 160, 320]]\
                          + [round_width(1280 * max(1.0, width_mult), 8)]


        super(MobileNet_V2_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], stride=2, Act=Act)

        self.stage1 = self.Stage(1, c[0], c[1], 1, 3, 1, Act)
        self.stage2 = self.Stage(2, c[1], c[2], 6, 3, 2, Act)
        self.stage3 = self.Stage(3, c[2], c[3], 6, 3, 2, Act)
        self.stage4 = self.Stage(4, c[3], c[4], 6, 3, 2, Act)
        self.stage5 = self.Stage(3, c[4], c[5], 6, 3, 1, Act)
        self.stage6 = self.Stage(3, c[5], c[6], 6, 3, 2, Act)
        self.stage7 = self.Stage(1, c[6], c[7], 6, 3, 1, Act)

        self.conv_last = Static_ConvLayer(c[7], c[8], 1, Act=Act)


    def Stage(self, num_blocks, in_channels, channels, expansion, kernel_size, stride, Act):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, expansion, kernel_size, stride, Act)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, expansion, kernel_size, 1, Act)

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
        x = self.stage7(x)
        x = self.conv_last(x)

        return x



class MNASNet_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1807.11626
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py

    It does not follow the structure decribed in <Table 7.(a)> of the paper.
    It is based on the widespread-implementations, including pytorch.

    In the paper, the concept of width multiplier is denoted by depth multiplier.
    """

    Block = Mobile_NAS_Block

    def __init__(self,
                 width_mult: float = 1.0,
                 Act: nn.Module = nn.ReLU()):

        self.widths = c = [round_width(i * width_mult, 8) for i in [32, 16, 24, 40, 80, 96, 192, 320]]\
                          + [round_width(1280 * max(1.0, width_mult), 8)]


        super(MNASNet_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], stride=2, Act=Act)

        self.stage1 = nn.Sequential(Depthwise_Conv2d(c[0], 3, 1), nn.BatchNorm2d(c[0]), Act,
                                    Pointwise_Conv2d(c[0], c[1]), nn.BatchNorm2d(c[1]))

        self.stage2 = self.Stage(3, c[1], c[2], 3, 3, 2, Act, None)
        self.stage3 = self.Stage(3, c[2], c[3], 3, 5, 2, Act, None)
        self.stage4 = self.Stage(3, c[3], c[4], 6, 5, 2, Act, None)
        self.stage5 = self.Stage(2, c[4], c[5], 6, 3, 1, Act, None)
        self.stage6 = self.Stage(4, c[5], c[6], 6, 5, 2, Act, None)
        self.stage7 = self.Stage(1, c[6], c[7], 6, 3, 1, Act, None)

        self.conv_last = Static_ConvLayer(c[7], c[8], 1, Act=Act)


    def Stage(self, num_blocks, in_channels, channels, expansion, kernel_size, stride, Act, se_ratio):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, expansion, kernel_size, stride, Act, se_ratio)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, expansion, kernel_size, 1, Act, se_ratio)

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
        x = self.stage7(x)
        x = self.conv_last(x)

        return x



class MobileNet_V3_Large_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

    The structure is decribed in <Table 1.> of the paper.

    'reduce_factor' is for 'reduce the channels in the last block of network backbone by a factor of 2' in the paper.

    Using 'reduce_factor', further than the paper, you can tune the group of widths from behind.
    For example, if c = [16, 16, 24, 40, 80, 112, 160, 960], reduce_factor = [2, 4],
    than the final c is [16, 16, 24, 40, 80, 112, 160 // 2, 960 // 4]
    """

    Block = Mobile_V3_Block

    def __init__(self,
                 width_mult: float = 1.0,
                 reduce_factor: list = None):

        c = [round_width(i * width_mult, 8) for i in [16, 16, 24, 40, 80, 112, 160, 960]]

        if reduce_factor:
            for i, factor in enumerate(reduce_factor[::-1]):
                c[-(i + 1)] //= factor

        self.widths = c
        self.width_mult, self.reduce_factor = width_mult, reduce_factor


        super(MobileNet_V3_Large_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], stride=2, Act=H_Swish())

        self.stage1 = self.Stage(1, c[0], c[1], 1, 3, 1, nn.ReLU(), None)
        self.stage2 = self.Stage(2, c[1], c[2], [4, 3], 3, 2, nn.ReLU(), None)
        self.stage3 = self.Stage(3, c[2], c[3], 3, 5, 2, nn.ReLU(), 4)
        self.stage4 = self.Stage(4, c[3], c[4], [6, 2.5, 2.3, 2.3], 3, 2, H_Swish(), None)
        self.stage5 = self.Stage(2, c[4], c[5], 6, 3, 1, H_Swish(), 4)
        self.stage6 = self.Stage(3, c[5], c[6], 6, 5, 2, H_Swish(), 4)

        self.conv_last = Static_ConvLayer(c[6], c[7], 1, Act=H_Swish())


    def Stage(self, num_blocks, in_channels, channels, expansion, kernel_size, stride, Act, se_ratio):
        if isinstance(expansion, (int, float)):
            expansion = [expansion] * num_blocks

        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, expansion[0], kernel_size, stride, Act, se_ratio)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, expansion[i], kernel_size, 1, Act, se_ratio)

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



class MobileNet_V3_Small_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

    The structure is decribed in <Table 2.> of the paper.
    """

    Block = Mobile_V3_Block

    def __init__(self,
                 width_mult: float = 1.0,
                 reduce_factor: list = None):

        c = [round_width(i * width_mult, 8) for i in [16, 16, 24, 40, 48, 96, 576]]

        if reduce_factor:
            for i, factor in enumerate(reduce_factor[::-1]):
                c[-(i + 1)] //= factor

        self.widths = c
        self.width_mult, self.reduce_factor = width_mult, reduce_factor


        super(MobileNet_V3_Small_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], stride=2, Act=H_Swish())

        self.stage1 = self.Stage(1, c[0], c[1], 1, 3, 2, nn.ReLU(), 4)
        self.stage2 = self.Stage(2, c[1], c[2], [4.5, 3.67], 3, 2, nn.ReLU(), None)
        self.stage3 = self.Stage(3, c[2], c[3], [4, 6, 6], 5, 2, H_Swish(), 4)
        self.stage4 = self.Stage(2, c[3], c[4], 3, 5, 1, H_Swish(), 4)
        self.stage5 = self.Stage(3, c[4], c[5], 6, 5, 2, H_Swish(), 4)

        self.conv_last = Static_ConvLayer(c[5], c[6], 1, Act=H_Swish())


    def Stage(self, num_blocks, in_channels, channels, expansion, kernel_size, stride, Act, se_ratio):
        if isinstance(expansion, (int, float)):
            expansion = [expansion] * num_blocks

        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, expansion[0], kernel_size, stride, Act, se_ratio)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, expansion[i], kernel_size, 1, Act, se_ratio)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.conv_last(x)

        return x




def mobilenet_v1_backbone(pretrained=False):
    model = MobileNet_V1_Backbone(1.0, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mobilenet_v1_backbone')
    return model


def mobilenet_v1_075_backbone(pretrained=False):
    model = MobileNet_V1_Backbone(0.75, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mobilenet_v1_075_backbone')
    return model


def mobilenet_v1_05_backbone(pretrained=False):
    model = MobileNet_V1_Backbone(0.5, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mobilenet_v1_05_backbone')
    return model


def mobilenet_v1_025_backbone(pretrained=False):
    model = MobileNet_V1_Backbone(0.25, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mobilenet_v1_025_backbone')
    return model


def mobilenet_v2_backbone(pretrained=False):
    model = MobileNet_V2_Backbone(1, nn.ReLU6())
    if pretrained:
        load_pretrained(model, 'mobilenet_v2_backbone')
    return model


def mobilenet_v2_14_backbone(pretrained=False):
    model = MobileNet_V2_Backbone(1.4, nn.ReLU6())
    if pretrained:
        load_pretrained(model, 'mobilenet_v2_14_backbone')
    return model


def mobilenet_v2_075_backbone(pretrained=False):
    model = MobileNet_V2_Backbone(0.75, nn.ReLU6())
    if pretrained:
        load_pretrained(model, 'mobilenet_v2_075_backbone')
    return model


def mobilenet_v2_05_backbone(pretrained=False):
    model = MobileNet_V2_Backbone(0.5, nn.ReLU6())
    if pretrained:
        load_pretrained(model, 'mobilenet_v2_05_backbone')
    return model


def mobilenet_v2_035_backbone(pretrained=False):
    model = MobileNet_V2_Backbone(0.35, nn.ReLU6())
    if pretrained:
        load_pretrained(model, 'mobilenet_v2_035_backbone')
    return model


def mnasnet_backbone(pretrained=False):
    model = MNASNet_Backbone(1, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mnasnet_backbone', batch_eps=1e-05, batch_momentum=3e-04)
    return model


def mnasnet_14_backbone(pretrained=False):
    model = MNASNet_Backbone(1.4, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mnasnet_14_backbone', batch_eps=1e-05, batch_momentum=3e-04)
    return model


def mnasnet_075_backbone(pretrained=False):
    model = MNASNet_Backbone(0.75, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mnasnet_075_backbone', batch_eps=1e-05, batch_momentum=3e-04)
    return model


def mnasnet_05_backbone(pretrained=False):
    model = MNASNet_Backbone(0.5, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mnasnet_05_backbone', batch_eps=1e-05, batch_momentum=3e-04)
    return model


def mnasnet_035_backbone(pretrained=False):
    model = MNASNet_Backbone(0.35, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'mnasnet_035_backbone', batch_eps=1e-05, batch_momentum=3e-04)
    return model


def mobilenet_v3_large_backbone(pretrained=False):
    model = MobileNet_V3_Large_Backbone(1, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_large_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_large_125_backbone(pretrained=False):
    model = MobileNet_V3_Large_Backbone(1.25, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_large_125_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_large_075_backbone(pretrained=False):
    model = MobileNet_V3_Large_Backbone(0.75, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_large_075_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_large_05_backbone(pretrained=False):
    model = MobileNet_V3_Large_Backbone(0.5, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_large_05_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_large_035_backbone(pretrained=False):
    model = MobileNet_V3_Large_Backbone(0.35, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_large_035_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_small_backbone(pretrained=False):
    model = MobileNet_V3_Small_Backbone(1, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_small_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_small_125_backbone(pretrained=False):
    model = MobileNet_V3_Small_Backbone(1.25, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_small_125_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_small_075_backbone(pretrained=False):
    model = MobileNet_V3_Small_Backbone(0.75, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_small_075_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_small_05_backbone(pretrained=False):
    model = MobileNet_V3_Small_Backbone(0.5, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_small_05_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def mobilenet_v3_small_035_backbone(pretrained=False):
    model = MobileNet_V3_Small_Backbone(0.35, None)
    if pretrained:
        load_pretrained(model, 'mobilenet_v3_small_035_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model
