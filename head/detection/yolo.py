from layers import *


class Yolo_V3_Predictor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767

    'stride' is a stride to the input image.
    """

    def __init__(self,
                 channels: int,
                 num_anchors: int,
                 num_classes: int,
                 stride: int):

        num_pred = 4 + 1 + num_classes
        pred_channels = num_anchors * num_pred

        self.num_anchors, self.num_pred, self.stride = num_anchors, num_pred, stride

        super(Yolo_V3_Predictor, self).__init__()

        self.conv_pred = nn.Conv2d(channels, pred_channels, 1, bias=True)


    def forward(self, f, anchors: Tensor = None):
        pred = self.conv_pred(f)

        pred = pred.view(f.shape[0], self.num_anchors, self.num_pred, f.shape[2], f.shape[3])
        pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(f.shape[0], -1, self.num_pred)


        if not self.training:
            if anchors.shape[1] != pred.shape[1]:
                raise RuntimeError('number of anchors and regressions must be matched')

            pred[..., :2] = torch.sigmoid(pred[..., :2]) * self.stride + anchors[..., :2]
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors[..., 2:]

        return pred



class Yolo_V3_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1804.02767

    The structure is explained in <2.3. Predictions Across Scales> of the paper.
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 strides: list):

        self.num_levels = num_levels

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')

        super(Yolo_V3_Head, self).__init__()

        self.heads = nn.ModuleList([Yolo_V3_Predictor(channels[i], num_anchors[i], num_classes, strides[i])
                                    for i in range(num_levels)])


    def forward(self, features, anchors: List[Tensor] = None):
        out = []

        for i in range(self.num_levels):
            if anchors:
                pred = self.heads[i](features[i], anchors[i])
            else:
                pred = self.heads[i](features[i])

            out.append(pred)
        out = torch.cat(out, 1)

        return out



class Yolo_V4_Predictor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    'stride' is a stride to the input image.
    """

    def __init__(self,
                 channels: int,
                 num_anchors: int,
                 num_classes: int,
                 stride: int,
                 Act: nn.Module = None):

        num_pred = 4 + 1 + num_classes
        pred_channels = num_anchors * num_pred

        self.num_anchors, self.num_pred, self.stride = num_anchors, num_pred, stride

        super(Yolo_V4_Predictor, self).__init__()

        self.conv_pred = Static_ConvLayer(channels, pred_channels, 1, bias=True, batch_norm=False, Act=Act)


    def forward(self, f, anchors=None):
        pred = self.conv_pred(f)

        pred = pred.view(f.shape[0], self.num_anchors, self.num_pred, f.shape[2], f.shape[3])
        pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(f.shape[0], -1, self.num_pred)


        if not self.training:
            if anchors.shape[1] != pred.shape[1]:
                raise RuntimeError('number of anchors and regressions must be matched')

            pred[..., :2] = (torch.sigmoid(pred[..., :2]) * 2 - 0.5) * self.stride + anchors[..., :2]
            pred[..., 2:4] = (torch.sigmoid(pred[..., 2:4]) * 2) ** 2 * anchors[..., 2:]

        return pred



class Yolo_V4_Head(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    The structure is based on the implementation;
    https://github.com/AlexeyAB/darknet
    """

    def __init__(self,
                 num_levels: int,
                 channels: list,
                 num_anchors: list,
                 num_classes: int,
                 strides: list,
                 Act: nn.Module = None):

        self.num_levels = num_levels

        if len(channels) != num_levels or len(num_anchors) != num_levels:
            raise ValueError('make len(channels) == num_levels, and len(num_anchors) == num_levels')

        super(Yolo_V4_Head, self).__init__()

        self.heads = nn.ModuleList([Yolo_V4_Predictor(channels[i], num_anchors[i], num_classes, strides[i], Act)
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
