from backbone.resnet import *
from backbone.feature_extractor import FeatureExtractor
from fpn.retinanet_fpn import RetinaNet_FPN
from head.detection.retinanet import RetinaNet_Head
from head.detection.anchor_maker import AnchorMaker


class RetinaNet_Frame(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hotcouscous1.

    This module is a framework for all models belonging to the RetinaNet series.

    The models are implemented as child classes inheriting from this module.
    The module passes objects to a model through inherited class/instance variables, instead of parameters.

    RetinaNet models necessarily include backbone, fpn, head.

    RetinaNet series share the same anchor-making.
    They configure anchor-priors by a standard size per stride, and its various scales and aspect ratios.
    """

    anchor_sizes = None
    anchor_scales = None
    anchor_ratios = None
    strides = None

    def __init__(self,
                 img_size: int):

        print('The model is for images sized in {}x{}.'.format(img_size, img_size))
        super(RetinaNet_Frame, self).__init__()

        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        self.backbone = None
        self.fpn = None
        self.head = None

        self.anchors = self.retinanet_anchors(img_size, self.anchor_sizes, self.anchor_scales, self.anchor_ratios, self.strides)


    def forward(self, input):
        features = self.backbone(input)
        features = self.fpn(features)
        cls_out, reg_out = self.head(features, self.anchors)
        return cls_out, reg_out


    def retinanet_anchors(self, img_size, anchor_sizes, anchor_scales, anchor_ratios, strides):
        anchor_priors = self.retinanet_anchor_priors(anchor_sizes, anchor_scales, anchor_ratios, strides)
        anchors = AnchorMaker(anchor_priors, strides)(img_size)
        return anchors


    @staticmethod
    def retinanet_anchor_priors(anchor_sizes, anchor_scales, anchor_ratios, strides):
        anchor_priors = []

        for stride, size in zip(strides, anchor_sizes):
            stride_priors = [[(size / stride) * s * r[0], (size / stride) * s * r[1]]
                             for s in anchor_scales
                             for r in anchor_ratios]

            anchor_priors.append(torch.Tensor(stride_priors))

        return anchor_priors



class RetinaNet(RetinaNet_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1708.02002

    The structure is based on the implementation;
    https://github.com/yhenon/pytorch-retinanet

    But there are some differences;
    1) given a standard size, anchor-priors are arranged in of ratio first, scale later.
    2) the bbox format of prediction of regressor and anchors is (x_min, y_min, x_max, y_max).

    The changes are for correspondence with other series, including EfficientDet.
    """

    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[2 ** (1 / 2), 2 ** (-1 / 2)], [1, 1], [2 ** (-1 / 2), 2 ** (1 / 2)]]
    strides = [8, 16, 32, 64, 128]


    def __init__(self,
                 img_size: int,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage2', 'stage3', 'stage4']

        if not (len(levels) + 2 == len(self.strides)):
            raise ValueError('make len(num_levels) + 2 == len(strides)')


        super(RetinaNet, self).__init__(img_size)

        self.backbone = FeatureExtractor(resnet_50_backbone(finetuning), levels)
        channels = self.backbone.widths[-len(levels):]

        self.fpn = RetinaNet_FPN(3, 5, channels, 256)

        self.head = RetinaNet_Head(5, 256, self.num_anchors, num_classes, nn.ReLU())

        if pretrained:
            load_pretrained(self, 'retinanet')
