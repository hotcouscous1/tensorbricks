from layers import *


class RetinaNet_Classifier(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 num_levels: int,
                 channels: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.ReLU()):

        self.num_levels, self.num_anchors, self.num_classes \
            = num_levels, num_anchors, num_classes


        super(RetinaNet_Classifier, self).__init__()

        layers = [Static_ConvLayer(channels, channels, 3, 1, True, batch_norm=False, Act=Act)
                  for _ in range(4)]

        self.layers = nn.Sequential(*layers)

        self.conv_pred = nn.Conv2d(channels, num_anchors * num_classes, 3, 1, 1)


    def forward(self, features):
        out = []

        for i in range(self.num_levels):
            f = self.layers(features[i])
            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], pred.shape[1], pred.shape[2], self.num_anchors, self.num_classes)
            pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out



class RetinaNet_Regressor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 num_levels: int,
                 channels: int,
                 num_anchors: int,
                 Act: nn.Module = nn.ReLU()):

        self.num_levels = num_levels

        super(RetinaNet_Regressor, self).__init__()


        layers = [Static_ConvLayer(channels, channels, 3, 1, True, batch_norm=False, Act=Act)
                  for _ in range(4)]

        self.layers = nn.Sequential(*layers)

        self.conv_pred = nn.Conv2d(channels, num_anchors * 4, 3, 1, 1)


    def forward(self, features, anchors: Tensor = None):
        out = []

        for i in range(self.num_levels):
            f = self.layers(features[i])
            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1).contiguous().view(pred.shape[0], -1, 4)

            out.append(pred)
        out = torch.cat(out, dim=1)


        if not self.training:
            if anchors.shape[1:] != out.shape[1:]:
                raise RuntimeError('number of anchors and regressions must be matched')

            out[..., :2] = anchors[..., :2] + (out[..., :2] * anchors[..., 2:])
            out[..., 2:] = torch.exp(out[..., 2:]) * anchors[..., 2:]

        return out



class RetinaNet_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1708.02002

    The structure is decribed in <4. RetinaNet Detector> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 channels: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.ReLU()):

        super(RetinaNet_Head, self).__init__()

        self.classifier = RetinaNet_Classifier(num_levels, channels, num_anchors, num_classes, Act)
        self.regressor = RetinaNet_Regressor(num_levels, channels, num_anchors, Act)


    def forward(self, features, anchors: Tensor = None):
        cls_out = self.classifier(features)
        reg_out = self.regressor(features, anchors)
        return cls_out, reg_out



class EfficientDet_Classifier(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070
    """

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU()):

        self.num_levels, self.num_anchors, self.num_classes \
            = num_levels, num_anchors, num_classes


        super(EfficientDet_Classifier, self).__init__()

        self.conv_layers = nn.ModuleList([Seperable_Conv2d(width, width, 3, 1, bias=True)
                                          for _ in range(depth)])

        self.bn_layers = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
                                        for _ in range(num_levels)])
        self.act = Act

        self.conv_pred = Seperable_Conv2d(width, num_anchors * num_classes, bias=True)


    def forward(self, features):
        out = []

        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], pred.shape[1], pred.shape[2], self.num_anchors, self.num_classes)
            pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out



class EfficientDet_Regressor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070
    """

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 Act: nn.Module = nn.SiLU()):

        self.num_levels = num_levels


        super(EfficientDet_Regressor, self).__init__()

        self.conv_layers = nn.ModuleList([Seperable_Conv2d(width, width, 3, 1, bias=True)
                                          for _ in range(depth)])

        self.bn_layers = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
                                        for _ in range(num_levels)])
        self.act = Act

        self.conv_pred = Seperable_Conv2d(width, num_anchors * 4, bias=True)


    def forward(self, features, anchors: Tensor = None):
        out = []

        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1).contiguous().view(pred.shape[0], -1, 4)

            out.append(pred)
        out = torch.cat(out, dim=1)


        if not self.training:
            if anchors.shape[1:] != out.shape[1:]:
                raise RuntimeError('number of anchors and regressions must be matched')

            out[..., :2] = anchors[..., :2] + (out[..., :2] * anchors[..., 2:])
            out[..., 2:] = torch.exp(out[..., 2:]) * anchors[..., 2:]

        return out



class EfficientDet_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1911.09070

    The structure is based on the official implementation;
    https://github.com/google/automl/tree/master/efficientdet
    """

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU()):

        super(EfficientDet_Head, self).__init__()

        self.classifier = EfficientDet_Classifier(num_levels, depth, width, num_anchors, num_classes, Act)
        self.regressor = EfficientDet_Regressor(num_levels, depth, width, num_anchors, Act)


    def forward(self, features, anchors: Tensor = None):
        cls_out = self.classifier(features)
        reg_out = self.regressor(features, anchors)
        return cls_out, reg_out
