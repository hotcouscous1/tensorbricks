from backbone.mobilenet import *


class MobileNet_V1(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1704.04861

    **kwargs -> width_mult: float, Act: nn.Module
    """

    def __init__(self,
                 mode: str = 'mobilenet_v1',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 **kwargs):

        super(MobileNet_V1, self).__init__()

        if not kwargs:
            if mode == 'mobilenet_v1':
                self.backbone = mobilenet_v1_backbone(finetuning)

            elif mode == 'mobilenet_v1_075':
                self.backbone = mobilenet_v1_075_backbone(finetuning)

            elif mode == 'mobilenet_v1_05':
                self.backbone = mobilenet_v1_05_backbone(finetuning)

            elif mode == 'mobilenet_v1_025':
                self.backbone = mobilenet_v1_025_backbone(finetuning)

        else:
            self.backbone = MobileNet_V1_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class MobileNet_V2(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381
    checkpoints : https://github.com/d-li14/mobilenetv2.pytorch

    **kwargs -> width_mult: float, Act: nn.Module
    """

    def __init__(self,
                 mode: str = 'mobilenet_v2',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.2,
                 **kwargs):

        super(MobileNet_V2, self).__init__()

        if not kwargs:
            if mode == 'mobilenet_v2':
                self.backbone = mobilenet_v2_backbone(finetuning)

            elif mode == 'mobilenet_v2_14':
                self.backbone = mobilenet_v2_14_backbone(finetuning)

            elif mode == 'mobilenet_v2_075':
                self.backbone = mobilenet_v2_075_backbone(finetuning)

            elif mode == 'mobilenet_v2_05':
                self.backbone = mobilenet_v2_05_backbone(finetuning)

            elif mode == 'mobilenet_v2_035':
                self.backbone = mobilenet_v2_035_backbone(finetuning)

        else:
            self.backbone = MobileNet_V2_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class MNASNet(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1807.11626
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py

    **kwargs -> width_mult: float, Act: nn.Module
    """

    def __init__(self,
                 mode: str = 'mnasnet',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.2,
                 **kwargs):

        super(MNASNet, self).__init__()

        if not kwargs:
            if mode == 'mnasnet':
                self.backbone = mnasnet_backbone(finetuning)

            elif mode == 'mnasnet_14':
                self.backbone = mnasnet_14_backbone(finetuning)

            elif mode == 'mnasnet_075':
                self.backbone = mnasnet_075_backbone(finetuning)

            elif mode == 'mnasnet_05':
                self.backbone = mnasnet_05_backbone(finetuning)

            elif mode == 'mnasnet_035':
                self.backbone = mnasnet_035_backbone(finetuning)

        else:
            self.backbone = MNASNet_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(self.backbone.widths[-1], num_classes))
        if pretrained:
            load_pretrained(self, mode, batch_eps=1e-05, batch_momentum=3e-04)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class MobileNet_V3_Large(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

    **kwargs -> width_mult: float, reduce_factor: list
    """

    def __init__(self,
                 mode: str = 'mobilenet_v3_large',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.2,
                 **kwargs):

        last_channels = 1280

        super(MobileNet_V3_Large, self).__init__()

        if not kwargs:
            if mode == 'mobilenet_v3_large':
                self.backbone = mobilenet_v3_large_backbone(finetuning)

            elif mode == 'mobilenet_v3_large_125':
                self.backbone = mobilenet_v3_large_125_backbone(finetuning)

            elif mode == 'mobilenet_v3_large_075':
                self.backbone = mobilenet_v3_large_075_backbone(finetuning)

            elif mode == 'mobilenet_v3_large_05':
                self.backbone = mobilenet_v3_large_05_backbone(finetuning)

            elif mode == 'mobilenet_v3_large_035':
                self.backbone = mobilenet_v3_large_035_backbone(finetuning)

        else:
            self.backbone = MobileNet_V3_Large_Backbone(**kwargs)


        width_mult, reduce_factor = self.backbone.width_mult, self.backbone.reduce_factor

        if width_mult != 1:
            last_channels = round_width(width_mult * last_channels, 8)

        if reduce_factor:
            last_channels = last_channels // reduce_factor[-1]


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], last_channels),
                                        H_Swish(),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(last_channels, num_classes))

        if pretrained:
            load_pretrained(self, mode, batch_eps=1e-03, batch_momentum=0.01)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out



class MobileNet_V3_Small(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

    **kwargs -> width_mult: float, reduce_factor: list
    """

    def __init__(self,
                 mode: str = 'mobilenet_v3_small',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.2,
                 **kwargs):

        last_channels = 1024

        super(MobileNet_V3_Small, self).__init__()

        if not kwargs:
            if mode == 'mobilenet_v3_small':
                self.backbone = mobilenet_v3_small_backbone(finetuning)

            elif mode == 'mobilenet_v3_small_125':
                self.backbone = mobilenet_v3_small_125_backbone(finetuning)

            elif mode == 'mobilenet_v3_small_075':
                self.backbone = mobilenet_v3_small_075_backbone(finetuning)

            elif mode == 'mobilenet_v3_small_05':
                self.backbone = mobilenet_v3_small_05_backbone(finetuning)

            elif mode == 'mobilenet_v3_small_035':
                self.backbone = mobilenet_v3_small_035_backbone(finetuning)

        else:
            self.backbone = MobileNet_V3_Small_Backbone(**kwargs)


        width_mult, reduce_factor = self.backbone.width_mult, self.backbone.reduce_factor

        if width_mult != 1:
            last_channels = round_width(width_mult * last_channels, 8)

        if reduce_factor:
            last_channels = last_channels // reduce_factor[-1]


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(1),
                                        nn.Linear(self.backbone.widths[-1], last_channels),
                                        H_Swish(),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(last_channels, num_classes))

        if pretrained:
            load_pretrained(self, mode, batch_eps=1e-03, batch_momentum=0.01)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out
