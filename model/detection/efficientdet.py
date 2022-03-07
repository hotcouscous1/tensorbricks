from backbone.efficientnet import *
from fpn.bifpn import BiFPN
from head.detection.retinanet import EfficientDet_Head
from model.detection.retinanet import *


class EfficientDet(RetinaNet_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070

    The structure and configurations are same with the official implementation;
    https://github.com/google/automl/tree/master/efficientdet

    However, they make the bbox format of prediction of regressor and anchors in (y_min, x_min, y_max, x_max).
    But there is some inefficiencies;
    1) in the implementation of focal loss, they transform anchors into (cx, cy, w, h) to assign them for labels.
    2) in the implementation of post process, to combine the prediction and anchors into (x_min, y_min, x_max, y_manx),
       they transform both of them into (cx, cy, w, h).
    3) in the implementation of generating anchors, they put each anchor on the center of grid-cell.

    Because the format of prediction is not corresponded, it is invalid to load the existing checkpoints.
    """

    resolutions = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    survival_probs = [None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    config = {'bifpn_depth': [3, 4, 5, 6, 7, 7, 8, 8, 8],
              'bifpn_width': [64, 88, 112, 160, 224, 288, 384, 384, 384],
              'head_depth':  [3, 3, 3, 4, 4, 4, 5, 5, 5],
              'head_width':  [64, 88, 112, 160, 224, 288, 384, 384, 384]}

    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[1, 1], [1.4, 0.7], [0.7, 1.4]]

    strides = [8, 16, 32, 64, 128]


    def __init__(self,
                 coeff: int,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        self.img_size = self.resolutions[coeff]

        if coeff == 7:
            self.anchor_sizes = [40, 80, 160, 320, 640]

        if coeff == 8:
            self.anchor_sizes = [32, 64, 128, 256, 512, 1024]
            self.strides = [8, 16, 32, 64, 128, 256]

        num_levels = len(self.strides)

        d_bifpn = self.config['bifpn_depth'][coeff]
        w_bifpn = self.config['bifpn_width'][coeff]
        d_head = self.config['head_depth'][coeff]
        w_head = self.config['head_width'][coeff]

        survival_prob = self.survival_probs[coeff]


        super(EfficientDet, self).__init__(self.img_size)

        if coeff < 7:
            backbone = EfficientNet_Backbone(coeff, survival_prob, nn.SiLU())
            if finetuning:
                load_pretrained(backbone, 'efficientnet_b' + str(coeff) + '_backbone')

        else:
            backbone = EfficientNet_Backbone(coeff - 1, survival_prob, nn.SiLU())
            if finetuning:
                load_pretrained(backbone, 'efficientnet_b' + str(coeff - 1) + '_backbone')

        del backbone.conv_last.layer[:]


        self.backbone = FeatureExtractor(backbone, ['stage3', 'stage5', 'stage7'])
        channels = self.backbone.widths[3: 8: 2]

        self.fpn = BiFPN(num_levels, d_bifpn, channels, w_bifpn, Act=nn.SiLU())

        self.head = EfficientDet_Head(num_levels, d_head, w_head, self.num_anchors, num_classes, nn.SiLU())

        if pretrained:
            load_pretrained(self, 'efficientdet_d' + str(coeff))
