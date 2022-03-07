from layers import *
from backbone.block.mobile import Mobile_NAS_Block
from utils import round_width, round_depth, get_survival_probs, load_pretrained



class EfficientNet_Backbone(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.11946
    checkpoints : https://github.com/lukemelas/EfficientNet-PyTorch

    The structure is decribed in <Table 1.> of the paper.

    In <Equation 3.> of the paper, compound coefficient is an exponent of three factors,
    but it is an index for given constants in the official implementation;
    https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    """

    Block = Mobile_NAS_Block

    config = {'init_depth': [0, 1, 2, 2, 3, 3, 4, 1],
              'init_width': [32, 16, 24, 40, 80, 112, 192, 320, 1280],
              'alpha': [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1, 3.6],
              'beta': [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]}


    def __init__(self,
                 coeff: int,
                 survival_prob: float = None,
                 Act: nn.Module = nn.SiLU()):

        a = self.config['alpha'][coeff]
        b = self.config['beta'][coeff]

        d = [round_depth(a * d) for d in self.config['init_depth']]
        w = [round_width(b * w, 8) for w in self.config['init_width']]

        p1, p2, p3, p4, p5, p6, p7 = get_survival_probs(d, survival_prob)

        self.coeff, self.depths, self.widths = coeff, d, w


        super(EfficientNet_Backbone, self).__init__()

        self.stage0 = Static_ConvLayer(3, w[0], stride=2, Act=Act)

        self.stage1 = self.Stage(d[1], w[0], w[1], 1, 3, 1, Act, 4, p1)
        self.stage2 = self.Stage(d[2], w[1], w[2], 6, 3, 2, Act, 4, p2)
        self.stage3 = self.Stage(d[3], w[2], w[3], 6, 5, 2, Act, 4, p3)
        self.stage4 = self.Stage(d[4], w[3], w[4], 6, 3, 2, Act, 4, p4)
        self.stage5 = self.Stage(d[5], w[4], w[5], 6, 5, 1, Act, 4, p5)
        self.stage6 = self.Stage(d[6], w[5], w[6], 6, 5, 2, Act, 4, p6)
        self.stage7 = self.Stage(d[7], w[6], w[7], 6, 3, 1, Act, 4, p7)

        self.conv_last = Static_ConvLayer(w[7], w[8], 1, Act=Act)


    def Stage(self, num_blocks, in_channels, channels, expansion, kernel_size, stride, Act, se_ratio, survival_prob):
        blocks = OrderedDict()
        blocks['block' + str(0)] = self.Block(in_channels, channels, expansion, kernel_size, stride, Act, se_ratio)

        for i in range(1, num_blocks):
            blocks['block' + str(i)] = self.Block(channels, channels, expansion, kernel_size, 1, Act, se_ratio, survival_prob[i])

        blocks = nn.Sequential(blocks)
        return blocks


    def forward(self, input):
        x = self.stage0(input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv_last(x)

        return x




def efficientnet_b0_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(0, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b0_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b1_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(1, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b1_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b2_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(2, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b2_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b3_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(3, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b3_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b4_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(4, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b4_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b5_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(5, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b5_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b6_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(6, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b6_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model


def efficientnet_b7_backbone(pretrained=False, survival_prob=None):
    model = EfficientNet_Backbone(7, survival_prob, nn.SiLU())
    if pretrained:
        load_pretrained(model, 'efficientnet_b7_backbone', batch_eps=1e-03, batch_momentum=0.01)
    return model
