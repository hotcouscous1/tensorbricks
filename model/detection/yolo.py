from backbone.darknet import *
from backbone.feature_extractor import FeatureExtractor
from fpn.yolo_fpn import Yolo_FPN, Yolo_V3_Tiny_FPN, Yolo_V4_Tiny_FPN
from fpn.yolo_pan import Yolo_PAN
from fpn.yolo_csp_pan import Yolo_CSP_PAN
from head.detection.yolo import Yolo_V3_Head, Yolo_V4_Head
from head.detection.anchor_maker import AnchorMaker

# In AlexeyAB implementation, the direction of stacking predictions is different for each cfg file.
# Here, every predictions are stacked in bottom-to-top manner.


class Yolo_Frame(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hotcouscous1.

    This module is a framework for all models belonging to the Yolo series.

    The models are implemented as child classes inheriting from this module.
    The module passes objects to a model through inherited class/instance variables, instead of parameters.

    Yolo models necessarily include backbone, fpn, head.

    Yolo series share the same anchor-making.
    They configure anchor-priors per stride by each of their sizes.
    """

    anchor_sizes = None
    strides = None

    def __init__(self,
                 img_size: int):

        print('The model is for images sized in {}x{}.'.format(img_size, img_size))
        super(Yolo_Frame, self).__init__()

        self.num_anchors = [len(i) for i in self.anchor_sizes]

        self.backbone = None
        self.fpn = None
        self.head = None

        self.anchors = self.yolo_anchors(img_size, self.anchor_sizes, self.strides)


    def forward(self, input):
        features = self.backbone(input)
        features = self.fpn(features)
        out = self.head(features, self.anchors)
        return out


    def yolo_anchors(self, img_size, anchor_sizes, strides):
        anchors = []
        anchor_priors = self.yolo_anchor_priors(anchor_sizes, strides)

        for i, s in enumerate(strides):
            s_anchors = [AnchorMaker(p.unsqueeze(0), [s], center=False)(img_size) for p in anchor_priors[i]]
            s_anchors = torch.cat(s_anchors, 1)

            anchors.append(s_anchors)
        return anchors


    @staticmethod
    def yolo_anchor_priors(anchor_sizes:List[list], strides:list):
        anchor_priors = []
        for stride, sizes in zip(strides, anchor_sizes):
            stride_priors = [(w / stride, h / stride) for w, h in sizes]
            anchor_priors.append(torch.Tensor(stride_priors))

        return anchor_priors




class Yolo_V3(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(10, 13), (16, 30), (33, 23)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(116, 90), (156, 198), (373, 326)]]

    strides = [8, 16, 32]


    def __init__(self,
                 img_size: int = 416,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V3, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v3_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_FPN(num_levels, channels, Act=nn.LeakyReLU(negative_slope=0.1))

        self.head = Yolo_V3_Head(num_levels, channels, self.num_anchors, num_classes, self.strides)

        if pretrained:
            load_pretrained(self, 'yolo_v3', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V3_SPP(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767
    """

    anchor_sizes = [[(10, 13), (16, 30), (33, 23)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(116, 90), (156, 198), (373, 326)]]

    strides = [8, 16, 32]


    def __init__(self,
                 img_size: int = 608,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V3_SPP, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v3_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_FPN(num_levels, channels, spp_kernels=[5, 9, 13], Act=nn.LeakyReLU(negative_slope=0.1))

        self.head = Yolo_V3_Head(num_levels, channels, self.num_anchors, num_classes, self.strides)

        if pretrained:
            load_pretrained(self, 'yolo_v3_spp', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V3_Tiny(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767
    """

    anchor_sizes = [[(10, 14), (23, 27), (37, 58)],
                    [(81, 82), (135, 169), (344, 319)]]

    strides = [16, 32]


    def __init__(self,
                 img_size: int = 416,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage4', 'pool_last']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V3_Tiny, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v3_tiny_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_V3_Tiny_FPN(Act=nn.LeakyReLU(negative_slope=0.1))

        self.head = Yolo_V3_Head(num_levels, channels, self.num_anchors, num_classes, self.strides)

        if pretrained:
            load_pretrained(self, 'yolo_v3_tiny', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V4(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2004.10934
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(12, 16), (19, 36), (40, 28)],
                    [(36, 75), (76, 55), (72, 146)],
                    [(142, 110), (192, 243), (459, 401)]]

    strides = [8, 16, 32]


    def __init__(self,
                 img_size: int = 608,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V4, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v4_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_PAN(num_levels, channels, [5, 9, 13], Act=nn.LeakyReLU(negative_slope=0.1))

        self.head = Yolo_V4_Head(num_levels, channels, self.num_anchors, num_classes, self.strides, None)

        if pretrained:
            load_pretrained(self, 'yolo_v4', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V4_CSP(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(12, 16), (19, 36), (40, 28)],
                    [(36, 75), (76, 55), (72, 146)],
                    [(142, 110), (192, 243), (459, 401)]]

    strides = [8, 16, 32]


    def __init__(self,
                 img_size: int = 512,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V4_CSP, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v4_csp_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_CSP_PAN(num_levels, channels, 2, [5, 9, 13], Act=Mish())

        self.head = Yolo_V4_Head(num_levels, channels, self.num_anchors, num_classes, self.strides, nn.Sigmoid())

        if pretrained:
            load_pretrained(self, 'yolo_v4_csp', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V4_Large_P5(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(13, 17), (31, 25), (24, 51), (61, 45)],
                    [(48, 102), (119, 96), (97, 189), (217, 184)],
                    [(171, 384), (324, 451), (616, 618), (800, 800)]]

    strides = [8, 16, 32]


    def __init__(self,
                 img_size: int = 896,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V4_Large_P5, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v4_large_p5_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_CSP_PAN(num_levels, channels, 3, [5, 9, 13], Act=Mish())

        self.head = Yolo_V4_Head(num_levels, channels, self.num_anchors, num_classes, self.strides, nn.Sigmoid())

        if pretrained:
            load_pretrained(self, 'yolo_v4_large_p5', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V4_Large_P6(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(13, 17), (31, 25), (24, 51), (61, 45)],
                    [(61, 45), (48, 102), (119, 96), (97, 189)],
                    [(97, 189), (217, 184), (171, 384), (324, 451)],
                    [(324, 451), (545, 357), (616, 618), (1024, 1024)]]

    strides = [8, 16, 32, 64]


    def __init__(self,
                 img_size: int = 1280,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5', 'stage6']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V4_Large_P6, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v4_large_p6_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_CSP_PAN(num_levels, channels, 3, [5, 9, 13], Act=Mish())

        self.head = Yolo_V4_Head(num_levels, channels, self.num_anchors, num_classes, self.strides, nn.Sigmoid())

        if pretrained:
            load_pretrained(self, 'yolo_v4_large_p6', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V4_Large_P7(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(13, 17), (22, 25), (27, 66), (55, 41)],
                    [(57, 88), (112, 69), (69, 177), (136, 138)],
                    [(136, 138), (287, 114), (134, 275), (268, 248)],
                    [(268, 248), (232, 504), (445, 416), (640, 640)],
                    [(812, 393), (477, 808), (1070, 908), (1408, 1408)]]

    strides = [8, 16, 32, 64, 128]


    def __init__(self,
                 img_size: int = 1536,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3', 'stage4', 'stage5', 'stage6', 'stage7']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V4_Large_P7, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v4_large_p7_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_CSP_PAN(num_levels, channels, 3, [5, 9, 13], Act=Mish())

        self.head = Yolo_V4_Head(num_levels, channels, self.num_anchors, num_classes, self.strides, nn.Sigmoid())

        if pretrained:
            load_pretrained(self, 'yolo_v4_large_p7', batch_eps=1e-04, batch_momentum=0.03)



class Yolo_V4_Tiny(Yolo_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/2011.08036
    checkpoints : https://github.com/AlexeyAB/darknet
    """

    anchor_sizes = [[(10, 14), (23, 27), (37, 58)],
                    [(81, 82), (135, 169), (344, 319)]]

    strides = [16, 32]


    def __init__(self,
                 img_size: int = 416,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        levels = ['stage3.block.trans', 'stage4']
        num_levels = len(levels)

        if not (num_levels == len(self.anchor_sizes) == len(self.strides)):
            raise ValueError('make num_levels == len(anchor_sizes) == len(strides)')


        super(Yolo_V4_Tiny, self).__init__(img_size)

        self.backbone = FeatureExtractor(yolo_v4_tiny_backbone(finetuning), levels)
        channels = self.backbone.widths[-num_levels:]

        self.fpn = Yolo_V4_Tiny_FPN(Act=nn.LeakyReLU(negative_slope=0.1))

        self.head = Yolo_V4_Head(num_levels, channels, self.num_anchors, num_classes, self.strides, None)

        if pretrained:
            load_pretrained(self, 'yolo_v4_tiny', batch_eps=1e-04, batch_momentum=0.03)
