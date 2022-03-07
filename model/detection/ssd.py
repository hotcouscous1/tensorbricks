from backbone.vgg import *
from backbone.resnet import *
from backbone.mobilenet import *
from backbone.feature_extractor import FeatureExtractor
from fpn.ssd_extra import *
from fpn.dssd_extra import *
from fpn.fpn import FPN
from head.detection.ssd import *
from head.detection.anchor_maker import AnchorMaker

# The official implementations of SSD and DSSD are written in caffe.
# To the best of my knowledge,this is the closest implementation to the official of SSD321/513 and DSSD.
# If you find an error, please let me know through Issues.


class SSD_Frame(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hot-couscous.

    This module is a framework for all models belonging to the SSD series.

    The models are implemented as child classes inheriting from this module.
    The module passes objects to a model through inherited class/instance variables, instead of parameters.

    SSD models necessarily include backbone, extra, classifier, regressor, and optionally include fpn.

    SSD series share the same anchor-making.
    They configure anchor-priors by a single size per stride, and its various aspect ratios.
    Aspect ratios are usually determined by # anchors.
    """

    config = None

    def __init__(self,
                 img_size: int):

        print('The model is for images sized in {}x{}.'.format(img_size, img_size))
        super(SSD_Frame, self).__init__()

        self.num_anchors = self.config[img_size]['num_anchors']

        self.backbone = None
        self.extra = None
        self.fpn = None
        self.head = None

        self.anchors = self.ssd_anchors(img_size, *self.config[img_size].values())


    def forward(self, input):
        features = self.backbone(input)
        features = self.extra(features)
        if self.fpn:
            features = self.fpn(features)

        cls_out, reg_out = self.head(features, self.anchors)
        return cls_out, reg_out



    def ssd_anchors(self, img_size, anchor_sizes, upper_sizes, strides, num_anchors):
        anchor_priors = self.ssd_anchor_priors(anchor_sizes, upper_sizes, strides, num_anchors)
        anchors = AnchorMaker(anchor_priors, strides, True, True, True)(img_size)
        return anchors


    @staticmethod
    def ssd_anchor_priors(anchor_sizes: list, upper_sizes: list, strides: list, num_anchors: list):
        anchor_scales = [size / stride for size, stride in zip(anchor_sizes, strides)]
        upper_scales = [upper / stride for upper, stride in zip(upper_sizes, strides)]

        anchor_priors = []
        for i in range(len(strides)):
            scale, upper = anchor_scales[i], upper_scales[i]
            aspect_ratios = [[1, 1], [math.sqrt(scale * upper) / scale] * 2]

            if num_anchors[i] == 4:
                ratios = [2]

            elif num_anchors[i] == 6:
                ratios = [2, 3]

            elif num_anchors[i] == 8:
                ratios = [1.60, 2, 3]

            else:
                raise ValueError('make num_anchors == 4 or 6 or 8')

            for r in ratios:
                aspect_ratios += [[math.sqrt(r), 1 / math.sqrt(r)], [1 / math.sqrt(r), math.sqrt(r)]]

            stride_priors = [[scale * a[0], scale * a[1]] for a in aspect_ratios]

            stride_priors = torch.Tensor(stride_priors)
            anchor_priors.append(stride_priors)

        return anchor_priors




class SSD(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325
    checkpoints : https://github.com/lufficc/SSD

    The structure and configurations of lufficc are identical to those of the official(weiliu89) and amdegroot.

    The difference, here, is that both image sizes support COCO dataset.
    """

    config = {
        300: {'anchor_sizes':   [21, 45, 99, 153, 207, 261],
              'upper_sizes':    [45, 99, 153, 207, 261, 315],
              'strides':        [8, 16, 32, 64, 100, 300],
              'num_anchors':    [4, 6, 6, 6, 4, 4]},

        512: {'anchor_sizes':   [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
              'upper_sizes':    [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
              'strides':        [8, 16, 32, 64, 128, 256, 512],
              'num_anchors':    [4, 6, 6, 6, 6, 4, 4]}
    }

    def __init__(self,
                 img_size: int = 300,
                 num_classes: int = 81,
                 pretrained: bool = False,
                 finetuning: bool = False):


        super(SSD, self).__init__(img_size)

        self.backbone, self.backbone_extra, self.norm = self.get_backbone(finetuning)
        channels = [self.backbone.widths[-1], 1024]

        if img_size == 300:
            channels += [channels[0], 256, 256, 256]

            self.extra = SSD_300_Extra(channels[1:], nn.ReLU())

        elif img_size == 512:
            channels += [channels[0], 256, 256, 256, 256]

            self.extra = SSD_512_Extra(channels[1:], nn.ReLU())

        else:
            raise ValueError('img_size should be 300 or 512')

        self.head = SSD_Head(len(channels), channels, self.num_anchors, num_classes)

        if pretrained:
            load_pretrained(self, 'ssd_' + str(img_size))


    def forward(self, input):
        features = self.backbone(input)
        features[0] = self.norm(features[0])
        features[1] = self.backbone_extra(features[1])

        features = self.extra(features)
        cls_out, reg_out = self.head(features, self.anchors)
        return cls_out, reg_out


    @staticmethod
    def get_backbone(finetuning):
        backbone = FeatureExtractor(vgg_16_backbone(finetuning), ['stage3.layer2', 'stage4'])

        backbone.model.stage2.pool.ceil_mode = True
        del backbone.model.stage4.pool

        backbone_extra = [nn.MaxPool2d(3, 1, 1),
                          nn.Conv2d(backbone.widths[-1], 1024, 3, padding=6, dilation=6),
                          nn.ReLU(),
                          nn.Conv2d(1024, 1024, 1),
                          nn.ReLU()]

        backbone_extra = nn.Sequential(*backbone_extra)

        norm = L2_Norm(512)

        return backbone, backbone_extra, norm




class SSD_FPN(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325
    """

    config = SSD.config

    def __init__(self,
                 img_size: int = 300,
                 num_classes: int = 81,
                 pretrained: bool = False,
                 finetuning: bool = False):


        super(SSD_FPN, self).__init__(img_size)

        self.backbone, self.backbone_extra, self.norm = SSD.get_backbone(finetuning)
        channels = [self.backbone.widths[-1], 1024]

        if img_size == 300:
            channels += [channels[0], 256, 256, 256]

            self.extra = SSD_300_Extra(channels[1:], nn.ReLU())
            self.fpn = FPN(len(channels), channels, 128, [38, 19, 10, 5, 3, 1])

        elif img_size == 512:
            channels += [channels[0], 256, 256, 256, 256]

            self.extra = SSD_512_Extra(channels[1:], nn.ReLU())
            self.fpn = FPN(len(channels), channels, 128, [64, 32, 16, 8, 4, 2, 1])

        else:
            raise ValueError('img_size should be 300 or 512')

        channels = [128] * len(channels)

        self.head = SSD_Head(len(channels), channels, self.num_anchors, num_classes)

        if pretrained:
            load_pretrained(self, 'ssd_fpn_' + str(img_size))


    def forward(self, input):
        features = self.backbone(input)
        features[0] = self.norm(features[0])
        features[1] = self.backbone_extra(features[1])

        features = self.extra(features)
        features = self.fpn(features)
        cls_out, reg_out = self.head(features, self.anchors)
        return cls_out, reg_out




class SSD_ResNet(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659
    official : https://github.com/chengyangfu/caffe

    The only difference from the official is 'dilation' of the first block of 'backbone_extra'.

    Each image size supports different dataset;
    312 -> Pascal
    512 -> COCO
    """

    config = {
        321: {'anchor_sizes':   [32.1, 64.2, 121.98, 179.76, 237.54, 295.32],
              'upper_sizes':    [64.2, 121.98, 179.76, 237.54, 295.32, 353.1],
              'strides':        [321 / i for i in [40, 20, 10, 5, 3, 1]],
              'num_anchors':    [8, 8, 8, 8, 8, 8]},

        513: {'anchor_sizes':   [20.52, 51.3, 138.51, 225.72, 312.93, 400.14, 487.35],
              'upper_sizes':    [51.3, 138.51, 225.72, 312.93, 400.14, 487.35, 574.56],
              'strides':        [513 / i for i in [64, 32, 16, 8, 4, 2, 1]],
              'num_anchors':    [8, 8, 8, 8, 8, 8, 8]}
    }

    def __init__(self,
                 img_size: int = 321,
                 num_classes: int = 81,
                 pretrained: bool = False,
                 finetuning: bool = False):


        super(SSD_ResNet, self).__init__(img_size)

        self.backbone, self.backbone_extra = self.get_backbone(finetuning)
        channels = [self.backbone.widths[-3], 2048]

        if img_size == 321:
            channels += [1024, 1024, 1024, 1024]

            self.extra = SSD_321_Extra(channels[1:], nn.ReLU())
            h_channels = [1024] * len(channels)

        elif img_size == 513:
            channels += [1024, 1024, 1024, 1024, 1024]

            self.extra = SSD_513_Extra(channels[1:], nn.ReLU())
            h_channels = channels

        else:
            raise ValueError('img_size should be 321 or 513')

        kernel_sizes = [5, 5] + [3] * (len(channels) - 2)

        self.head = DSSD_Head(len(channels), channels, h_channels, self.num_anchors, num_classes, kernel_sizes, nn.ReLU())

        if pretrained:
            load_pretrained(self, 'ssd_resnet_' + str(img_size))


    def forward(self, input):
        features = self.backbone(input)
        features[1] = self.backbone_extra(features[1])

        features = self.extra(features)
        cls_out, reg_out = self.head(features, self.anchors)
        return cls_out, reg_out


    @staticmethod
    def get_backbone(finetuning):
        backbone = FeatureExtractor(resnet_101_backbone(finetuning), ['stage2', 'stage3'])

        backbone.model.stage0[1].padding = 0
        del backbone.model.stage4[:]

        backbone_extra = [Extra_Res_Block(1024, 2048, padding=2, dilation=2, shortcut=True),
                          Extra_Res_Block(2048, 2048, padding=2, dilation=2),
                          Extra_Res_Block(2048, 2048, padding=2, dilation=2)]

        backbone_extra = nn.Sequential(*backbone_extra)

        return backbone, backbone_extra




class DSSD(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659
    official : https://github.com/chengyangfu/caffe

    Each image size supports different dataset;
    312 -> Pascal
    512 -> COCO
    """

    config = {
        321: {'anchor_sizes':   [32.1, 64.2, 121.98, 179.76, 237.54, 295.32],
              'upper_sizes':    [64.2, 121.98, 179.76, 237.54, 295.32, 353.1],
              'strides':        [321 / i for i in [40, 20, 10, 5, 3, 1]],
              'num_anchors':    [8, 8, 8, 8, 8, 8]},

        513: {'anchor_sizes':   [20.52, 51.3, 138.51, 225.72, 312.93, 400.14, 487.35],
              'upper_sizes':    [51.3, 138.51, 225.72, 312.93, 400.14, 487.35, 574.56],
              'strides':        [513 / i for i in [64, 32, 16, 8, 4, 2, 1]],
              'num_anchors':    [8, 8, 8, 8, 8, 8, 8]}
    }

    def __init__(self,
                 img_size: int = 321,
                 num_classes: int = 81,
                 pretrained: bool = False,
                 finetuning: bool = False):


        super(DSSD, self).__init__(img_size)

        self.backbone, self.backbone_extra = SSD_ResNet.get_backbone(finetuning)
        channels = [self.backbone.widths[-3], 2048]

        if img_size == 321:
            channels += [1024, 1024, 1024, 1024]

            self.extra = DSSD_321_Extra(channels, nn.ReLU())

        elif img_size == 513:
            channels += [1024, 1024, 1024, 1024, 1024]

            self.extra = DSSD_513_Extra(channels, nn.ReLU())

        else:
            raise ValueError('img_size should be 321 or 513')

        h_channels = [1024] * len(channels)
        kernel_sizes = [5, 5] + [3] * (len(channels) - 2)

        self.head = DSSD_Head(len(channels), channels, h_channels, self.num_anchors, num_classes, kernel_sizes, nn.ReLU())

        if pretrained:
            load_pretrained(self, 'dssd_' + str(img_size))


    def forward(self, input):
        features = self.backbone(input)
        features[1] = self.backbone_extra(features[1])

        features = self.extra(features)
        cls_out, reg_out = self.head(features, self.anchors)
        return cls_out, reg_out




class SSDLite(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659
    official : https://github.com/tensorflow/models/tree/master/research/object_detection

    The input size of MobileNet V2 SSDLite is 300 in the implementation,
    but in the paper, it is suggested 320.
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


        super(SSDLite, self).__init__(img_size)

        self.backbone = FeatureExtractor(mobilenet_v2_backbone(finetuning), ['stage5', 'conv_last'])
        channels = [self.backbone.widths[5], self.backbone.widths[-1]]

        channels += [512, 256, 256, 128]

        self.extra = SSDLite_Extra(channels[1:], [0.2, 0.25, 0.5, 0.25], nn.ReLU6())

        self.head = SSDLite_Head(len(channels), channels, self.num_anchors, num_classes, nn.ReLU6())

        if pretrained:
            load_pretrained(self, 'ssdlite')



class SSDLite_MobileNet_V3_Large(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659
    official : https://github.com/tensorflow/models/tree/master/research/object_detection
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


        super(SSDLite_MobileNet_V3_Large, self).__init__(img_size)

        self.backbone = FeatureExtractor(MobileNet_V3_Large_Backbone(1.0, [2, 2]),
                                         ['stage6.block0.pw_layer', 'conv_last'])

        channels = [6 * self.backbone.widths[-3], self.backbone.widths[-1]]

        channels += [512, 256, 256, 128]

        self.extra = SSDLite_Extra(channels[1:], [0.5333, 0.25, 0.5, 0.25], nn.ReLU6())     # round_width(0.5333 * 480, 1) = 256

        self.head = SSDLite_Head(len(channels), channels, self.num_anchors, num_classes, nn.ReLU6())

        if pretrained:
            load_pretrained(self, 'ssdlite_mobile_v3_large')
        if finetuning:
            load_pretrained(self.backbone, 'ssdlite_mobile_v3_large_backbone')



class SSDLite_MobileNet_V3_Small(SSD_Frame):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1701.06659
    official : https://github.com/tensorflow/models/tree/master/research/object_detection
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


        super(SSDLite_MobileNet_V3_Small, self).__init__(img_size)

        self.backbone = FeatureExtractor(MobileNet_V3_Small_Backbone(1.0, [2, 2]),
                                         ['stage5.block0.pw_layer', 'conv_last'])

        channels = [6 * self.backbone.widths[-3], self.backbone.widths[-1]]

        channels += [512, 256, 256, 128]

        self.extra = SSDLite_Extra(channels[1:], [0.89, 0.25, 0.5, 0.25], nn.ReLU6())       # round_width(0.89 * 288, 1) = 256

        self.head = SSDLite_Head(len(channels), channels, self.num_anchors, num_classes, nn.ReLU6())

        if pretrained:
            load_pretrained(self, 'ssdlite_mobile_v3_small')
        if finetuning:
            load_pretrained(self.backbone, 'ssdlite_mobile_v3_small_backbone')
