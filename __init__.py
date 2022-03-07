import os
import numpy as np
import math
from typing import List, Tuple
import itertools
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("You are on " + str(torch.cuda.get_device_name(device)))
else:
    print("You are on " + str(device).upper())


Numpy = np.array
Tensor = torch.Tensor


checkpoints = {
    'efficientnet_b0': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b0.pth',
    'efficientnet_b0_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b0_backbone.pth',
    'efficientnet_b1': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b1.pth',
    'efficientnet_b1_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b1_backbone.pth',
    'efficientnet_b2': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b2.pth',
    'efficientnet_b2_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b2_backbone.pth',
    'efficientnet_b3': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b3.pth',
    'efficientnet_b3_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b3_backbone.pth',
    'efficientnet_b4': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b4.pth',
    'efficientnet_b4_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b4_backbone.pth',
    'efficientnet_b5': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b5.pth',
    'efficientnet_b5_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b5_backbone.pth',
    'efficientnet_b6': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b6.pth',
    'efficientnet_b6_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b6_backbone.pth',
    'efficientnet_b7': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b7.pth',
    'efficientnet_b7_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b7_backbone.pth',
    'mnasnet': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mnasnet.pth',
    'mnasnet_05': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mnasnet_05.pth',
    'mnasnet_05_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mnasnet_05_backbone.pth',
    'mnasnet_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mnasnet_backbone.pth',
    'mobilenet_v2': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2.pth',
    'mobilenet_v2_035': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_035.pth',
    'mobilenet_v2_035_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_035_backbone.pth',
    'mobilenet_v2_05': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_05.pth',
    'mobilenet_v2_05_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_05_backbone.pth',
    'mobilenet_v2_075': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_075.pth',
    'mobilenet_v2_075_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_075_backbone.pth',
    'mobilenet_v2_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v2_backbone.pth',
    'mobilenet_v3_large': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v3_large.pth',
    'mobilenet_v3_large_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v3_large_backbone.pth',
    'mobilenet_v3_small': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v3_small.pth',
    'mobilenet_v3_small_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/mobilenet_v3_small_backbone.pth',
    'res2net_101_26w_4s': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_101_26w_4s.pth',
    'res2net_101_26w_4s_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_101_26w_4s_backbone.pth',
    'res2net_101_v1b_26w_4s': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_101_v1b_26w_4s.pth',
    'res2net_101_v1b_26w_4s_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_101_v1b_26w_4s_backbone.pth',
    'res2net_50_14w_8s': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_14w_8s.pth',
    'res2net_50_14w_8s_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_14w_8s_backbone.pth',
    'res2net_50_26w_4s': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_26w_4s.pth',
    'res2net_50_26w_4s_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_26w_4s_backbone.pth',
    'res2net_50_26w_8s': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_26w_8s.pth',
    'res2net_50_26w_8s_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_26w_8s_backbone.pth',
    'res2net_50_v1b_26w_4s': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_v1b_26w_4s.pth',
    'res2net_50_v1b_26w_4s_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2net_50_v1b_26w_4s_backbone.pth',
    'res2next_50': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2next_50.pth',
    'res2next_50_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/res2next_50_backbone.pth',
    'resnet_101': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_101.pth',
    'resnet_101_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_101_backbone.pth',
    'resnet_152': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_152.pth',
    'resnet_152_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_152_backbone.pth',
    'resnet_18': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_18.pth',
    'resnet_18_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_18_backbone.pth',
    'resnet_34': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_34.pth',
    'resnet_34_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_34_backbone.pth',
    'resnet_50': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_50.pth',
    'resnet_50_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnet_50_backbone.pth',
    'resnext_101_32x8d': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnext_101_32x8d.pth',
    'resnext_101_32x8d_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnext_101_32x8d_backbone.pth',
    'resnext_50_32x4d': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnext_50_32x4d.pth',
    'resnext_50_32x4d_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/resnext_50_32x4d_backbone.pth',
    'rexnet': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet.pth',
    'rexnet_13': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_13.pth',
    'rexnet_13_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_13_backbone.pth',
    'rexnet_15': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_15.pth',
    'rexnet_15_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_15_backbone.pth',
    'rexnet_2': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_2.pth',
    'rexnet_2_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_2_backbone.pth',
    'rexnet_3': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_3.pth',
    'rexnet_3_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_3_backbone.pth',
    'rexnet_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/rexnet_backbone.pth',
    'ssd_300': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/ssd_300.pth',
    'ssd_512': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/ssd_512.pth',
    'vgg_11': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_11.pth',
    'vgg_11_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_11_backbone.pth',
    'vgg_13': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_13.pth',
    'vgg_13_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_13_backbone.pth',
    'vgg_16': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_16.pth',
    'vgg_16_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_16_backbone.pth',
    'vgg_19': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_19.pth',
    'vgg_19_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/vgg_19_backbone.pth',
    'yolo_v3': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v3.pth',
    'yolo_v3_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v3_backbone.pth',
    'yolo_v4': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4.pth',
    'yolo_v4_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_backbone.pth',
    'yolo_v4_csp': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_csp.pth',
    'yolo_v4_csp_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_csp_backbone.pth',
    'yolo_v4_large_p5': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_large_p5.pth',
    'yolo_v4_large_p5_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_large_p5_backbone.pth',
    'yolo_v4_large_p6': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_large_p6.pth',
    'yolo_v4_large_p6_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_large_p6_backbone.pth',
    'yolo_v4_tiny': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_tiny.pth',
    'yolo_v4_tiny_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/yolo_v4_tiny_backbone.pth',
}

