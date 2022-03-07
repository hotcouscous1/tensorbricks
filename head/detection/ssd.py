from layers import *
from fpn.ssd_extra import Extra_Res_Block


class SSD_Classifier(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 kernel_sizes: list = None):

        if not kernel_sizes:
            kernel_sizes = [3] * num_levels

        self.num_levels, self.num_classes = num_levels, num_classes

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')


        super(SSD_Classifier, self).__init__()

        conv_pred = [Static_ConvLayer(c, a * num_classes, k, bias=True, batch_norm=False, Act=None)
                     for c, a, k in zip(channels, num_anchors, kernel_sizes)]

        self.conv_pred = nn.ModuleList(conv_pred)


    def forward(self, features):
        out = []

        for i in range(self.num_levels):
            pred = self.conv_pred[i](features[i])

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out



class SSD_Regressor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325
    """

    center_variance = 0.1
    size_variance = 0.2

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 kernel_sizes: list = None):

        if not kernel_sizes:
            kernel_sizes = [3] * num_levels

        self.num_levels = num_levels

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')


        super(SSD_Regressor, self).__init__()

        conv_pred = [Static_ConvLayer(c, a * 4, k, bias=True, batch_norm=False, Act=None)
                     for c, a, k in zip(channels, num_anchors, kernel_sizes)]

        self.conv_pred = nn.ModuleList(conv_pred)


    def forward(self, features, anchors: Tensor = None):
        out = []

        for i in range(self.num_levels):
            pred = self.conv_pred[i](features[i])

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], -1, 4)

            out.append(pred)
        out = torch.cat(out, dim=1)


        if not self.training:
            if anchors.shape[1:] != out.shape[1:]:
                raise RuntimeError('number of anchors and regressions must be matched')

            out[..., :2] = anchors[..., :2] + (self.center_variance * out[..., :2] * anchors[..., 2:])
            out[..., 2:] = torch.exp(self.size_variance * out[..., 2:]) * anchors[..., 2:]

        return out



class SSD_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1512.02325

    The structure is described in <2.1 Model> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 kernel_sizes: list = None):

        super(SSD_Head, self).__init__()

        self.classifier = SSD_Classifier(num_levels, channels, num_anchors, num_classes, kernel_sizes)
        self.regressor = SSD_Regressor(num_levels, channels, num_anchors, kernel_sizes)


    def forward(self, features, anchors: Tensor = None):
        cls_out = self.classifier(features)
        reg_out = self.regressor(features, anchors)
        return cls_out, reg_out



class DSSD_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/pdf/1701.06659

    The structure is described in <Figure 2.(c)> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 kernel_sizes: list = None,
                 Act: nn.Module = nn.ReLU()):

        self.num_levels = num_levels

        super(DSSD_Head, self).__init__()

        self.layers = nn.ModuleList([Extra_Res_Block(in_channels[i], channels[i], 1, Act=Act, shortcut=True)
                                     for i in range(self.num_levels)])

        self.classifier = SSD_Classifier(num_levels, channels, num_anchors, num_classes, kernel_sizes)
        self.regressor = SSD_Regressor(num_levels, channels, num_anchors, kernel_sizes)


    def forward(self, features, anchors: Tensor = None):
        for i in range(self.num_levels):
            features[i] = self.layers[i](features[i])

        cls_out = self.classifier(features)
        reg_out = self.regressor(features, anchors)

        return cls_out, reg_out



class SSDLite_Classifier(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 Act: nn.Module = nn.ReLU6()):

        self.num_levels, self.num_classes = num_levels, num_classes

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')


        super(SSDLite_Classifier, self).__init__()

        conv_pred = [nn.Sequential(Depthwise_Conv2d(channels[i], 3, 1, bias=True),
                                   nn.BatchNorm2d(channels[i]),
                                   Act,
                                   Pointwise_Conv2d(channels[i], num_anchors[i] * num_classes, bias=True))
                     for i in range(num_levels - 1)]

        conv_pred.append(nn.Conv2d(channels[-1], num_anchors[-1] * num_classes, 1, bias=True))

        self.conv_pred = nn.ModuleList(conv_pred)


    def forward(self, features):
        out = []

        for i in range(self.num_levels):
            pred = self.conv_pred[i](features[i])

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out



class SSDLite_Regressor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381
    """

    center_variance = 0.1
    size_variance = 0.2

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 Act: nn.Module = nn.ReLU6()):

        self.num_levels = num_levels

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')


        super(SSDLite_Regressor, self).__init__()

        conv_pred = [nn.Sequential(Depthwise_Conv2d(channels[i], 3, 1, bias=True),
                                   nn.BatchNorm2d(channels[i]),
                                   Act,
                                   Pointwise_Conv2d(channels[i], num_anchors[i] * 4, bias=True))
                     for i in range(num_levels - 1)]

        conv_pred.append(nn.Conv2d(channels[-1], num_anchors[-1] * 4, 1, bias=True))

        self.conv_pred = nn.ModuleList(conv_pred)


    def forward(self, features, anchors: Tensor = None):
        out = []

        for i in range(self.num_levels):
            pred = self.conv_pred[i](features[i])

            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(pred.shape[0], -1, 4)

            out.append(pred)
        out = torch.cat(out, dim=1)


        if not self.training:
            if anchors.shape[1:] != out.shape[1:]:
                raise RuntimeError('number of anchors and regressions must be matched')

            out[..., :2] = anchors[..., :2] + (self.center_variance * out[..., :2] * anchors[..., 2:])
            out[..., 2:] = torch.exp(self.size_variance * out[..., 2:]) * anchors[..., 2:]

        return out



class SSDLite_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1801.04381

    The structure is based on the implementation;
    https://github.com/chuanqi305/MobileNetv2-SSDLite
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 Act: nn.Module = nn.ReLU6()):

        super(SSDLite_Head, self).__init__()

        self.classifier = SSDLite_Classifier(num_levels, channels, num_anchors, num_classes, Act)
        self.regressor = SSDLite_Regressor(num_levels, channels, num_anchors, Act)


    def forward(self, features, anchors: Tensor = None):
        cls_out = self.classifier(features)
        reg_out = self.regressor(features, anchors)
        return cls_out, reg_out
