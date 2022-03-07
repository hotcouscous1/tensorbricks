from backbone.efficientnet import *


class EfficientNet(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.11946
    checkpoints : https://github.com/lukemelas/EfficientNet-PyTorch

    **kwargs -> coeff: int, survival_prob: float, Act: nn.Module
    """

    resolutions = [224, 240, 260, 300, 380, 456, 528, 600, 672]
    survival_probs = [0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5]


    def __init__(self,
                 mode: str = 'efficientnet_b0',
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 finetuning: bool = False,
                 drop_rate: float = 0.2,
                 **kwargs):

        super(EfficientNet, self).__init__()

        if not kwargs:
            if mode == 'efficientnet_b0':
                self.backbone = efficientnet_b0_backbone(finetuning, self.survival_probs[0])

            elif mode == 'efficientnet_b1':
                self.backbone = efficientnet_b1_backbone(finetuning, self.survival_probs[1])

            elif mode == 'efficientnet_b2':
                self.backbone = efficientnet_b2_backbone(finetuning, self.survival_probs[2])

            elif mode == 'efficientnet_b3':
                self.backbone = efficientnet_b3_backbone(finetuning, self.survival_probs[3])

            elif mode == 'efficientnet_b4':
                self.backbone = efficientnet_b4_backbone(finetuning, self.survival_probs[4])

            elif mode == 'efficientnet_b5':
                self.backbone = efficientnet_b5_backbone(finetuning, self.survival_probs[5])

            elif mode == 'efficientnet_b6':
                self.backbone = efficientnet_b6_backbone(finetuning, self.survival_probs[6])

            elif mode == 'efficientnet_b7':
                self.backbone = efficientnet_b7_backbone(finetuning, self.survival_probs[7])

        else:
            self.backbone = EfficientNet_Backbone(**kwargs)


        self.resolution = self.resolutions[self.backbone.coeff]

        self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(1),
                        nn.Dropout(drop_rate),
                        nn.Linear(self.backbone.widths[-1], num_classes))

        if pretrained:
            load_pretrained(self, mode, batch_eps=1e-03, batch_momentum=0.01)


    def forward(self, input):
        x = self.backbone(input)
        out = self.classifier(x)
        return out
