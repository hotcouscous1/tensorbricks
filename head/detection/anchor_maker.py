from __init__ import *


class AnchorMaker(object):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hotcouscous1.

    This module is to generate anchors supporting all series of detection models.

    Anchors are arranged in the shape of
    1) order of levels of 'strides'
    2) order of columns
    3) order of rows
    4) order of 'anchor_priors' on a grid cell

    The output of AnchorMaker is mapped to all predictions of detection-heads, given 'strides' in bottom-to-top manner.
    Therefore, it can also make anchors within __init__ of a model.

    This implementation is faster, because it supports gpu and grid is made only once at the level of each stride.


    'anchor_priors' is relative sizes to a single grid cell of a feature map, regardless of its stride.
    'strides' is strides of each level on which the anchors are placed, to the input image.
    'center' is to place anchors on the center of each grid.
    if 'center' is False, anchors are placed on the left-top, which is the case of Yolo.
    'clamp' is to bound all anchor-values between 0 and image size.
    'relative' is to normalize all anchor-values by image size.
    """

    def __init__(self,
                 anchor_priors: Tensor or List[Tensor],
                 strides: List[int],
                 center: bool = True,
                 clamp: bool = False,
                 relative: bool = False):

        super(AnchorMaker, self).__init__()

        if type(anchor_priors) is Tensor or len(anchor_priors) != len(strides):
            anchor_priors = len(strides) * [anchor_priors]

        self.priors = anchor_priors
        self.strides = strides
        self.center = center
        self.clamp = clamp
        self.relative = relative


    def __call__(self, img_size):
        return self.make_anchors(img_size)


    def make_anchors(self, img_size:int):
        all_anchors = []

        for stride, priors in zip(self.strides, self.priors):
            stride_anchors = []

            num_grid = math.ceil(img_size / stride)

            if self.center:
                grid = torch.arange(num_grid, device=device).repeat(num_grid, 1).float() + 0.5
            else:
                grid = torch.arange(num_grid, device=device).repeat(num_grid, 1).float()

            cx = grid * stride
            cy = grid.t() * stride

            boxes = (stride * priors)

            for box in boxes:
                w = torch.full([num_grid, num_grid], box[0], device=device)
                h = torch.full([num_grid, num_grid], box[1], device=device)
                anchor = torch.stack((cx, cy, w, h))

                stride_anchors.append(anchor)

            stride_anchors = torch.cat(stride_anchors).unsqueeze(0)
            stride_anchors = stride_anchors.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

            all_anchors.append(stride_anchors)
        all_anchors = torch.cat(all_anchors, dim=1)


        if self.clamp:
            all_anchors = torch.clamp(all_anchors, 0, img_size)

        if self.relative:
            all_anchors /= img_size

        return all_anchors
