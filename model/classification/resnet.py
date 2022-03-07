from backbone.resnet import *


class ResNet(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.03385
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    **kwargs -> num_blocks: list, bottleneck: bool, Act: nn.Module
    """

    def __init__(self,
                 mode: str = 'resnet_50',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 **kwargs):

        super(ResNet, self).__init__()

        if not kwargs:
            if mode == 'resnet_18':
                self.backbone = resnet_18_backbone(finetuning)

            elif mode == 'resnet_34':
                self.backbone = resnet_34_backbone(finetuning)

            elif mode == 'resnet_50':
                self.backbone = resnet_50_backbone(finetuning)

            elif mode == 'resnet_101':
                self.backbone = resnet_101_backbone(finetuning)

            elif mode == 'resnet_152':
                self.backbone = resnet_152_backbone(finetuning)

        else:
            self.backbone = ResNet_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class Wide_ResNet(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1605.07146

    **kwargs -> num_blocks: int, width_mult: int, Act: nn.Module, drop_rate: float
    """

    def __init__(self,
                 mode: str = 'wide_resnet_28_10',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 **kwargs):

        super(Wide_ResNet, self).__init__()

        if not kwargs:
            if mode == 'wide_resnet_28_10':
                self.backbone = wide_resnet_28_10_backbone(finetuning)

            elif mode == 'wide_resnet_40_4':
                self.backbone = wide_resnet_40_4_backbone(finetuning)

        else:
            self.backbone = Wide_ResNet_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AvgPool2d(8),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class ResNeXt(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1611.05431
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    'resnext_101_32x4d', 'resnext_101_64x4d' are proposed by the official implementation, written in lua;
    https://github.com/facebookresearch/ResNeXt

    **kwargs -> num_blocks: list, cardinality: int, base_group_channels: int, Act: nn.Module
    """

    def __init__(self,
                 mode: str = 'resnext_50_32x4d',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 **kwargs):

        super(ResNeXt, self).__init__()

        if not kwargs:
            if mode == 'resnext_50_32x4d':
                self.backbone = resnext_50_32x4d_backbone(finetuning)

            elif mode == 'resnext_101_32x4d':
                self.backbone = resnext_101_32x4d_backbone(finetuning)

            elif mode == 'resnext_101_32x8d':
                self.backbone = resnext_101_32x8d_backbone(finetuning)

            elif mode == 'resnext_101_64x4d':
                self.backbone = resnext_101_64x4d_backbone(finetuning)

        else:
            self.backbone = ResNeXt_Backbone(**kwargs)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class Res2Net(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1904.01169
    checkpoints : https://github.com/Res2Net/Res2Net-PretrainedModels

    **kwargs -> num_blocks: list, scales: int, base_channels: int, Act: nn.Module, cardinality: int, v1b: bool
    """

    def __init__(self,
                 mode: str = 'res2net_50_26w_4s',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 **kwargs):

        super(Res2Net, self).__init__()


        if not kwargs:
            if mode == 'res2net_50_26w_4s':
                self.backbone = res2net_50_26w_4s_backbone(finetuning)

            elif mode == 'res2net_50_14w_8s':
                self.backbone = res2net_50_14w_8s_backbone(finetuning)

            elif mode == 'res2net_50_26w_8s':
                self.backbone = res2net_50_26w_8s_backbone(finetuning)

            elif mode == 'res2net_101_26w_4s':
                self.backbone = res2net_101_26w_4s_backbone(finetuning)

            elif mode == 'res2next_50':
                self.backbone = res2next_50_backbone(finetuning)

            elif mode == 'res2net_50_v1b_26w_4s':
                self.backbone = res2net_50_v1b_26w_4s_backbone(finetuning)

            elif mode == 'res2net_101_v1b_26w_4s':
                self.backbone = res2net_101_v1b_26w_4s_backbone(finetuning)

        else:
            self.backbone = Res2Net_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



