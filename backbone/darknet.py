from backbone.block.dark import *
from utils import load_pretrained


class DarkNet_53_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767
    checkpoints : https://github.com/AlexeyAB/darknet

    The structure is decribed in <Table 1.> of the paper.
    """

    Block = DarkNet_Block

    def __init__(self,
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        self.widths = c = [32, 64, 128, 256, 512, 1024]


        super(DarkNet_53_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], 3, Act=Act)

        self.stage1 = self.Stage(1, c[0], c[1], 2, True, 2, Act)
        self.stage2 = self.Stage(2, c[1], c[2], 2, True, 2, Act)
        self.stage3 = self.Stage(8, c[2], c[3], 2, True, 2, Act)
        self.stage4 = self.Stage(8, c[3], c[4], 2, True, 2, Act)
        self.stage5 = self.Stage(4, c[4], c[5], 2, True, 2, Act)


    def Stage(self, num_blocks, in_channels, channels, reduction, shortcut, stride, Act):
        blocks = OrderedDict()
        blocks['conv'] = Static_ConvLayer(in_channels, channels, 3, stride, Act=Act)

        for i in range(num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, reduction, shortcut, Act)

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



class DarkNet_Tiny_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    The structure is based on the implementation;
    https://github.com/AlexeyAB/darknet.
    """

    def __init__(self,
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        self.widths = c = [16, 32, 64, 128, 256, 512]


        super(DarkNet_Tiny_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], 3, Act=Act)

        self.stage1 = self.Stage(c[0], c[1], 2, Act)
        self.stage2 = self.Stage(c[1], c[2], 2, Act)
        self.stage3 = self.Stage(c[2], c[3], 2, Act)
        self.stage4 = self.Stage(c[3], c[4], 2, Act)
        self.stage5 = self.Stage(c[4], c[5], 2, Act)

        self.pool_last = nn.Sequential(nn.ZeroPad2d([0, 1, 0, 1]),
                                       nn.MaxPool2d(2, 1))


    def Stage(self, in_channels, channels, stride, Act):
        block = OrderedDict()
        block['pool'] = nn.MaxPool2d(stride, stride)
        block['layer'] = Static_ConvLayer(in_channels, channels, 3, Act=Act)

        block = nn.Sequential(block)
        return block


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool_last(x)

        return x



class CSP_DarkNet_53_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2004.10934 / https://arxiv.org/abs/2011.08036 (csp)
    checkpoints : https://github.com/AlexeyAB/darknet

    The structure is based on DarkNet_53_Backbone according to the implementation.

    'csp' is to change from Yolo_V4 to Yolo_V4_CSP, according to
    'to get a better speed/accuracy trade-off, we convert the first CSP stage into original Darknet residual layer'.
    """

    Block = CSP_DarkNet_Block

    def __init__(self,
                 Act: nn.Module = Mish(),
                 csp: bool = False):

        self.widths = c = [32, 64, 128, 256, 512, 1024]

        super(CSP_DarkNet_53_Backbone, self).__init__()


        self.stage0 = Static_ConvLayer(3, c[0], 3, Act=Act)

        self.stage1 = self.Stage(1, c[0], c[1], False, 2, 2, Act)
        if csp:
            self.stage1.block = DarkNet_Block(c[1], c[1], 2, True, Act=Act)

        self.stage2 = self.Stage(2, c[1], c[2], True, 1, 2, Act)
        self.stage3 = self.Stage(8, c[2], c[3], True, 1, 2, Act)
        self.stage4 = self.Stage(8, c[3], c[4], True, 1, 2, Act)
        self.stage5 = self.Stage(4, c[4], c[5], True, 1, 2, Act)


    def Stage(self, num_blocks, in_channels, channels, half, block_reduction, stride, Act):
        blocks = OrderedDict()
        blocks['conv'] = Static_ConvLayer(in_channels, channels, 3, stride, Act=Act)
        blocks['block'] = self.Block(num_blocks, channels, channels, half, block_reduction, Act)

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



class CSP_DarkNet_Large_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet

    The structure is decribed in <Figure 4.> of the paper.

    'p' is to scale # stages of backbone up, from P5 to P6, P7.
    """

    Block = CSP_DarkNet_Block

    def __init__(self,
                 Act: nn.Module = Mish(),
                 p: int = 5):

        if p not in (5, 6, 7):
            raise ValueError('scale of the backboone of Yolo V4-Large is only avaiable from p5 to p7')

        self.widths = c = [32, 64, 128, 256, 512, 1024]
        self.p = p


        super(CSP_DarkNet_Large_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, c[0], 3, Act=Act)

        self.stage1 = self.Stage(1,  c[0], c[1], True, 1, 2, Act)
        self.stage2 = self.Stage(3,  c[1], c[2], True, 1, 2, Act)
        self.stage3 = self.Stage(15, c[2], c[3], True, 1, 2, Act)
        self.stage4 = self.Stage(15, c[3], c[4], True, 1, 2, Act)
        self.stage5 = self.Stage(7,  c[4], c[5], True, 1, 2, Act)

        if p > 5:
            self.stage6 = self.Stage(7, c[5], c[5], True, 1, 2, Act)
            self.widths.append(1024)
        if p > 6:
            self.stage7 = self.Stage(7, c[5], c[5], True, 1, 2, Act)
            self.widths.append(1024)


    def Stage(self, num_blocks, in_channels, channels, half, block_reduction, stride, Act):
        blocks = OrderedDict()
        blocks['conv'] = Static_ConvLayer(in_channels, channels, 3, stride, Act=Act)
        blocks['block'] = self.Block(num_blocks, channels, channels, half, block_reduction, Act)

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        if self.p > 5:
            x = self.stage6(x)
        if self.p > 6:
            x = self.stage7(x)
        return x



class CSP_DarkNet_Tiny_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet

    The structure is based on the implementation above.
    """

    Block = CSP_DarkNet_Tiny_Block

    def __init__(self,
                 Act: nn.Module = nn.LeakyReLU(negative_slope=0.1)):

        self.widths = c = [32, 64, 128, 256, 512]

        super(CSP_DarkNet_Tiny_Backbone, self).__init__()


        self.stage0 = Static_ConvLayer(3, c[0], 3, 2, Act=Act)

        self.stage1 = self.Stage(c[0], c[1], False, 2, Act)
        self.stage2 = self.Stage(c[1], c[2], True, 2, Act)
        self.stage3 = self.Stage(c[2], c[3], True, 2, Act)

        self.stage4 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    Static_ConvLayer(c[4], c[4], 3, Act=Act))


    def Stage(self, in_channels, channels, pool, stride, Act):
        block = OrderedDict()
        if pool:
            block['pool'] = nn.MaxPool2d(stride, stride)
        else:
            block['conv'] = Static_ConvLayer(in_channels, channels, 3, stride, Act=Act)

        block['block'] = self.Block(channels, channels, Act)

        block = nn.Sequential(block)
        return block


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x




def yolo_v3_backbone(pretrained=False):
    model = DarkNet_53_Backbone(nn.LeakyReLU(negative_slope=0.1))
    if pretrained:
        load_pretrained(model, 'yolo_v3_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v3_tiny_backbone(pretrained=False):
    model = DarkNet_Tiny_Backbone(nn.LeakyReLU(negative_slope=0.1))
    if pretrained:
        load_pretrained(model, 'yolo_v3_tiny_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v4_backbone(pretrained=False):
    model = CSP_DarkNet_53_Backbone(Mish(), False)
    if pretrained:
        load_pretrained(model, 'yolo_v4_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v4_csp_backbone(pretrained=False):
    model = CSP_DarkNet_53_Backbone(Mish(), True)
    if pretrained:
        load_pretrained(model, 'yolo_v4_csp_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v4_large_p5_backbone(pretrained=False):
    model = CSP_DarkNet_Large_Backbone(Mish(), 5)
    if pretrained:
        load_pretrained(model, 'yolo_v4_large_p5_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v4_large_p6_backbone(pretrained=False):
    model = CSP_DarkNet_Large_Backbone(Mish(), 6)
    if pretrained:
        load_pretrained(model, 'yolo_v4_large_p6_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v4_large_p7_backbone(pretrained=False):
    model = CSP_DarkNet_Large_Backbone(Mish(), 7)
    if pretrained:
        load_pretrained(model, 'yolo_v4_large_p7_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model


def yolo_v4_tiny_backbone(pretrained=False):
    model = CSP_DarkNet_Tiny_Backbone(nn.LeakyReLU(negative_slope=0.1))
    if pretrained:
        load_pretrained(model, 'yolo_v4_tiny_backbone', batch_eps=1e-04, batch_momentum=0.03)
    return model

