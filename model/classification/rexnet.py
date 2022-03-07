from backbone.rexnet import *


class ReXNet(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/2007.00992
    checkpoints : https://github.com/clovaai/rexnet

    **kwargs -> start_channels: int, last_channels: int, width_mult: float, depth_mult: float
    """

    def __init__(self,
                 mode: str = 'rexnet',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.2,
                 **kwargs):

        super(ReXNet, self).__init__()

        if not kwargs:
            if mode == 'rexnet':
                self.backbone = rexnet_backbone(finetuning)

            elif mode == 'rexnet_3':
                self.backbone = rexnet_3_backbone(finetuning)

            elif mode == 'rexnet_22':
                self.backbone = rexnet_22_backbone(finetuning)

            elif mode == 'rexnet_2':
                self.backbone = rexnet_2_backbone(finetuning)

            elif mode == 'rexnet_15':
                self.backbone = rexnet_15_backbone(finetuning)

            elif mode == 'rexnet_13':
                self.backbone = rexnet_13_backbone(finetuning)

            elif mode == 'rexnet_09':
                self.backbone = rexnet_09_backbone(finetuning)

        else:
            self.backbone = ReXNet_Backbone(**kwargs)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Dropout(drop_rate),
                                        nn.Conv2d(self.backbone.widths[-1], num_classes, 1, bias=True),
                                        nn.Flatten(1))
        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out
