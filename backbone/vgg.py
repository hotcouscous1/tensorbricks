from layers import *
from utils import load_pretrained


class VGG_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1409.1556
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

    The structure is decribed in <Table 1.> of the paper.

    'num_blocks' receives a list of # layers for each stage, instead of total # layers, for flexibility.
    'batch_norm' replaces LRN with batch normalization.
    """

    def __init__(self,
                 num_blocks: list,
                 batch_norm: bool = False,
                 Act: nn.Module = nn.ReLU()):

        self.widths = c = [64, 128, 256, 512, 512]

        super(VGG_Backbone, self).__init__()

        self.stage0 = self.Stage(num_blocks[0], 3, c[0], batch_norm, Act)
        self.stage1 = self.Stage(num_blocks[1], c[0], c[1], batch_norm, Act)
        self.stage2 = self.Stage(num_blocks[2], c[1], c[2], batch_norm, Act)
        self.stage3 = self.Stage(num_blocks[3], c[2], c[3], batch_norm, Act)
        self.stage4 = self.Stage(num_blocks[4], c[3], c[4], batch_norm, Act)

        self.widths = c


    def Stage(self, _num_layers, in_channels, channels, batch_norm, Act):
        layers = OrderedDict()
        layers['layer' + str(0)] = Static_ConvLayer(in_channels, channels, 3, 1, True, batch_norm, Act)

        for i in range(1, _num_layers):
            layers['layer' + str(i)] = Static_ConvLayer(channels, channels, 3, 1, True, batch_norm, Act)

        layers['pool'] = nn.MaxPool2d(2, 2)

        layers = nn.Sequential(layers)
        return layers


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x




def vgg_11_backbone(pretrained=False):
    model = VGG_Backbone([1, 1, 2, 2, 2], False, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'vgg_11_backbone')
    return model


def vgg_13_backbone(pretrained=False):
    model = VGG_Backbone([2, 2, 2, 2, 2], False, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'vgg_13_backbone')
    return model


def vgg_16_backbone(pretrained=False):
    model = VGG_Backbone([2, 2, 3, 3, 3], False, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'vgg_16_backbone')
    return model


def vgg_19_backbone(pretrained=False):
    model = VGG_Backbone([2, 2, 4, 4, 4], False, nn.ReLU())
    if pretrained:
        load_pretrained(model, 'vgg_19_backbone')
    return model
