from backbone.rexnet import *
from fpn.bifpn import BiFPN
from model.detection.ssd import *


class SSDLite_ReXNet_BiFPN(SSD_Frame):
    """
    __version__ = 1.1
    __date__ = Mar 23, 2022

    Created by hotcouscous1.

    This is the simple insertion of BiFPN into SSDLite where ReXNet is the backbone.

    Since BiFPN has the same number of channels in all strides through resampling,
    we slightly changed the output channels of Extra.
    """

    config = {320: {'anchor_sizes':   [60, 105, 150, 195, 240, 285],
                    'upper_sizes':    [105, 150, 195, 240, 285, 330],
                    'strides':        [16, 32, 64, 107, 160, 320],
                    'num_anchors':    [6, 6, 6, 6, 6, 6]}}


    def __init__(self,
                 img_size: int = 320,
                 num_classes: int = 81,
                 pretrained: bool = False,
                 finetuning: bool = False):


        super(SSDLite_ReXNet_BiFPN, self).__init__(img_size)

        self.backbone = FeatureExtractor(ReXNet_Backbone(16, 180, 1, 1),
                                         ['stage6.block0.pw_layer', 'conv_last'])

        channels = [6 * self.backbone.widths[-3], self.backbone.widths[-1]]

        channels += [512, 256, 256, 128]

        self.extra = SSDLite_Extra(channels[1:], [0.25, 0.25, 0.5, 0.25], nn.SiLU())

        self.fpn = BiFPN(6, 3, channels, 256, sizes=[20, 10, 5, 3, 2, 1], strides=[2, 2, 2, 2, 2])
        channels = 6 * [256]

        self.head = SSDLite_Head(len(channels), channels, self.num_anchors, num_classes, nn.SiLU())

        if pretrained:
            load_pretrained(self, 'ssdlite_rexnet_bifpn')
        if finetuning:
            load_pretrained(self.backbone, 'rexnet_backbone')

