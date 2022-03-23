from model.detection.yolo import *
from model.detection.efficientdet import *


class New_Predictor(nn.Module):
    """
    __version__ = 1.1
    __date__ = Mar 23, 2022

    Created by hotcouscous1.

    The difference from the predictor of YoloV4 is that this has depth-scalable lateral layers.

    Unlike Yolo-FPNs first fuse features and then pass through enough laterals,
    BiFPN repeatedly fuses features.
    """

    def __init__(self,
                 depth: int,
                 channels: int,
                 num_anchors: int,
                 num_classes: int,
                 stride: int,
                 Act: nn.Module = None):

        num_pred = 4 + 1 + num_classes
        pred_channels = num_anchors * num_pred

        self.num_anchors, self.num_pred, self.stride = num_anchors, num_pred, stride

        super(New_Predictor, self).__init__()


        layers = [nn.Sequential(Seperable_Conv2d(channels, channels, 3, 1, bias=True),
                                nn.BatchNorm2d(channels),
                                nn.SiLU())
                  for _ in range(depth)]

        self.layers = nn.ModuleList(layers)

        self.conv_pred = nn.Sequential(Seperable_Conv2d(channels, pred_channels, 1, bias=True))
        self.act = Act


    def forward(self, f, anchors=None):
        for layer in self.layers:
            f = layer(f)

        pred = self.conv_pred(f)

        if self.act:
            pred = self.act(pred)

        pred = pred.view(f.shape[0], self.num_anchors, self.num_pred, f.shape[2], f.shape[3])
        pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(f.shape[0], -1, self.num_pred)


        if not self.training:
            if anchors.shape[1] != pred.shape[1]:
                raise RuntimeError('number of anchors and regressions must be matched')

            pred[..., :2] = (torch.sigmoid(pred[..., :2]) * 2 - 0.5) * self.stride + anchors[..., :2]
            pred[..., 2:4] = (torch.sigmoid(pred[..., 2:4]) * 2) ** 2 * anchors[..., 2:]

        return pred



class New_Head(nn.Module):
    """
    __version__ = 1.1
    __date__ = Mar 23, 2022

    Created by hotcouscous1.
    """

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 strides: list,
                 Act: nn.Module = None):

        self.num_levels = num_levels

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')

        super(New_Head, self).__init__()

        self.heads = nn.ModuleList([New_Predictor(depth, channels[i], num_anchors[i], num_classes, strides[i], Act)
                                    for i in range(num_levels)])


    def forward(self, features, anchors: List[Tensor] = None):
        out = []

        for i in range(self.num_levels):
            if anchors:
                pred = self.heads[i](features[i], anchors[i])
            else:
                pred = self.heads[i](features[i], None)

            out.append(pred)
        out = torch.cat(out, 1)

        return out



class Efficient_Yolo(Yolo_Frame):
    """
    __version__ = 1.1
    __date__ = Mar 23, 2022

    Created by hotcouscous1.

    The main idea of Efficient-Yolo​is to adopt YoloV4's prediction to EfficientDet.

    Although the models of Scaled-YoloV4 have more parameters and better performance,
    the inference speed is faster than EfficientDets, due to fewer numbers of prediction.

    While EfficientDet shares the same number of strides and stride levels regardless of coefficient,
    the new model adopts a scalable number of strides and stride levels.

    coeff = {0, 1}    ->  # anchors = 3 / # levels = 3
    coeff = {2, 3, 4} ->  # anchors = 4 / # levels = 3
    coeff = {5, 6}    ->  # anchors = 4 / # levels = 4
    coeff = {7, 8}    ->  # anchors = 4 / # levels = 5

    These numbers are determined roughly, either by similar image size or comparable performance.

    And the size of anchor priors increase linearly by the image size,
    since those of Yolo serires are K-means to the labels of COCO dataset.
    """

    resolutions = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    survival_probs = [None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    config = {'bifpn_depth': [3, 4, 5, 6, 7, 7, 8, 8, 8],
              'bifpn_width': [64, 88, 112, 160, 224, 288, 384, 384, 384],
              'head_depth': [3, 3, 3, 4, 4, 4, 5, 5, 5],
              'head_width': [64, 88, 112, 160, 224, 288, 384, 384, 384]}

    num_anchors = [3, 3, 4, 4, 4, 4, 4, 4, 4]

    num_strides = [3, 3, 3, 3, 3, 4, 4, 5, 5]


    def __init__(self,
                 coeff: int,
                 num_classes: int = 80,
                 pretrained: bool = False,
                 finetuning: bool = False):

        self.img_size = self.resolutions[coeff]
        self.num_anchors = self.num_anchors[coeff]
        self.num_levels = self.num_strides[coeff]

        self.strides = [2 ** (3 + exp) for exp in range(self.num_levels)]

        self.anchor_sizes = self.get_anchor_sizes(self.img_size, self.num_anchors, self.num_levels)

        d_bifpn = self.config['bifpn_depth'][coeff]
        w_bifpn = self.config['bifpn_width'][coeff]
        d_head = self.config['head_depth'][coeff]
        w_head = self.config['head_width'][coeff]

        survival_prob = self.survival_probs[coeff]


        super(Efficient_Yolo, self).__init__(self.img_size)

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

        self.fpn = BiFPN(self.num_levels, d_bifpn, channels, w_bifpn, Act=nn.SiLU())
        w_head = self.num_levels * [w_head]

        self.head = New_Head(self.num_levels, d_head, w_head, self.num_anchors, num_classes, self.strides, nn.Sigmoid())

        if pretrained:
            load_pretrained(self, 'efficient_yolo_d' + str(coeff))



    @staticmethod
    def get_anchor_sizes(img_size, num_anchors, num_levels):
        anchor_sizes = []

        mean_9 = []
        for stride in Yolo_V4_CSP.anchor_sizes:
            for anchor in stride:
                mean_9.append((anchor[0] / 512, anchor[1] / 512))

        mean_12 = []
        for stride in Yolo_V4_Large_P5.anchor_sizes:
            for anchor in stride:
                mean_12.append((anchor[0] / 896, anchor[1] / 896))

        mean_16 = []
        for stride in Yolo_V4_Large_P6.anchor_sizes:
            for anchor in stride:
                mean_16.append((anchor[0] / 1280, anchor[1] / 1280))

        mean_20 = []
        for stride in Yolo_V4_Large_P7.anchor_sizes:
            for anchor in stride:
                mean_20.append((anchor[0] / 1536, anchor[1] / 1536))

        k_means = {9: mean_9, 12: mean_12, 16: mean_16, 20: mean_20}
        k = num_anchors * num_levels

        for i in range(num_levels):
            anchors = k_means[k][i * num_anchors: (i + 1) * num_anchors]
            anchors = [(int(w * img_size), int(h * img_size)) for w, h in anchors]
            anchor_sizes.append(anchors)

        return anchor_sizes

