from backbone.resnet import *
from fpn.bifpn import Resample_FPN
from fpn.pan import PAN
from model.detection.retinanet import *


class RetinaNet_Res2Net_PAN(RetinaNet_Frame):
    """
    __version__ = 1.1
    __date__ = Mar 23, 2022

    Created by hotcouscous1.

    This model is RetinaNet adopting Res2Net and PAN as backbone and fpn.
    You can select variants of Res2Net via 'mode'.

    For a larger input image and higher performance, it detects at 6 strides.

    PAN has the same number of input and output features.
    To pass 3 features from the feature extractor to PAN with 6 strides, it requires resampling the number and channels of them.
    For this, we adopt Resample_FPN of BiFPN which has simple structure.
    """

    anchor_sizes = [32, 64, 128, 256, 512, 1024]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[2 ** (1 / 2), 2 ** (-1 / 2)], [1, 1], [2 ** (-1 / 2), 2 ** (1 / 2)]]
    strides = [8, 16, 32, 64, 128, 256]


    def __init__(self,
                 img_size: int,
                 mode: str = 'res2net_50_26w_4s',
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage2', 'stage3', 'stage4']

        if not (len(levels) + 3 == len(self.strides)):
            raise ValueError('make len(num_levels) + 3 == len(strides)')


        super(RetinaNet_Res2Net_PAN, self).__init__(img_size)

        if mode == 'res2net_50_26w_4s':
            backbone = res2net_50_26w_4s_backbone(finetuning)

        elif mode == 'res2net_50_14w_8s':
            backbone = res2net_50_14w_8s_backbone(finetuning)

        elif mode == 'res2net_50_26w_8s':
            backbone = res2net_50_26w_8s_backbone(finetuning)

        elif mode == 'res2net_101_26w_4s':
            backbone = res2net_101_26w_4s_backbone(finetuning)

        elif mode == 'res2next_50':
            backbone = res2next_50_backbone(finetuning)

        elif mode == 'res2net_50_v1b_26w_4s':
            backbone = res2net_50_v1b_26w_4s_backbone(finetuning)

        elif mode == 'res2net_101_v1b_26w_4s':
            backbone = res2net_101_v1b_26w_4s_backbone(finetuning)

        else:
            raise ValueError('please select the backbone mode')


        self.backbone = FeatureExtractor(backbone, levels)
        channels = self.backbone.widths[-len(levels):]

        self.fpn = nn.Sequential(Resample_FPN(3, 6, channels, 256),
                                 PAN(6, 6 * [256], 256))

        self.head = RetinaNet_Head(6, 256, self.num_anchors, num_classes, nn.ReLU())

        if pretrained:
            load_pretrained(self, 'retinanet_' + mode + '_pan')