from backbone.vgg import *


class VGG(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1409.1556
    checkpoints : https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

    **kwargs -> num_blocks: list, batch_norm: bool, Act: nn.Module
    """

    def __init__(self,
                 mode: str = 'vgg_16',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.5,
                 **kwargs):

        super(VGG, self).__init__()

        if not kwargs:
            if mode == 'vgg_11':
                self.backbone = vgg_11_backbone(finetuning)

            elif mode == 'vgg_13':
                self.backbone = vgg_13_backbone(finetuning)

            elif mode == 'vgg_16':
                self.backbone = vgg_16_backbone(finetuning)

            elif mode == 'vgg_19':
                self.backbone = vgg_19_backbone(finetuning)

        else:
            self.backbone = VGG_Backbone(**kwargs)


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)),
                                        nn.Flatten(1),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(self.backbone.widths[-1] * (7 * 7), 4096), nn.ReLU(),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(4096, 4096), nn.ReLU(),
                                        nn.Linear(4096, num_classes))

        if pretrained:
            load_pretrained(self, mode)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out
