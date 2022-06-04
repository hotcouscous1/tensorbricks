# TensorBricks : Toward Structuralism in Deep-Learning
PyTorch-based framework for high-level development of image models  

TensorBricks builds models up by various combinations of compatible modules, beyond a collection of implementations.  


## Customize Your Own Models 
### Object-Oriented-Structuring
We don't build Legos by adding just one brick at a time. Instead, we build a bundle of **substructures** and combine them.  

Likewise, convolution models for the same performance are usually built in the homogeneous structure. We can have **compatibility** among them, 1) if the start and end of the algorithm are consistent in every models, 2) consistent in each set of substructures that occupy equivalent position in the model, 3) and if any substructure's end and the following substructure's start are not occluded or departed.  

OOS is TensorBricks code-style to provide compatibility of convolution models and their sub-modules. OOS proposes explicit seperation between modules, and build their structural hierarchy in bottom-up manner. And this is very necessary for our main purposes, **combinational building** and **flexible replacement**.  


### 1. Backbone
Every backbones of OOS consist of stages, stages consist of blocks, and blocks consist of layers. You can access any stage, block or layer of the backbone implemented in OOS, to get features.  

```python
from backbone.efficientnet import *
from backbone.feature_extractor import FeatureExtractor

backbone = efficientnet_b3_backbone(pretrained=True)

del backbone.conv_last.layer[:]
del backbone.widths[-1]

backbone = FeatureExtractor(backbone, ['stage3', 'stage5', 'stage7'])
```

### 2. FPN
OOS proposes dynamic implementations of FPN. They are free from the number of pyramid levels and strides on the size of input image, while strictly following papers and reference codes.

```python
from fpn.pan import PAN

fpn = PAN(3, backbone.widths[3: 8: 2], 256)
```

### 3. Head
Every heads of OOS are consistent in the form of prediction. Classifiers in any performance do not contain possibilities, and dense predictions of detection models are arranged from bottom to top at the pyramid level and from left-top to bottom-right on a feature.  

OOS provides an efficient anchor-maker, synchronized with any detection prediction. It supports the anchor-priors and the box format of every series. And it is fast, since it can generate anchor boxes before forward, given the size of input image.  

```python
from head.detection.yolo import Yolo_V4_Head

head = Yolo_V4_Head(3, [256, 256, 256], [4, 4, 4], 80, [8, 16, 32], Act=nn.SiLU())
```

### 4. Frame
No model is created from nothing. It is derived from previous works and share some conventions along the series of developments. OOS implements the most basic attributes among models in the series as Frame. Therefore, to combine them all, you only need to put the substructures above into the frame.  

```python
from model.detection.yolo import Yolo_Frame

frame = Yolo_Frame

frame.anchor_sizes = [
    [(13, 17), (31, 25), (24, 51), (61, 45)],
    [(48, 102), (119, 96), (97, 189), (217, 184)],
    [(171, 384), (324, 451), (616, 618), (800, 800)]
]
frame.strides = [8, 16, 32]

model = frame(896)

model.backbone = backbone
model.fpn = fpn
model.head = head
```

*Wow! You now have a new model, EfficientNet-PAN-YoloV4.*  

### 5. Upgrade your model through replacement
Also, you can upgrade the model by replacing one of the substructures, instead of building from scratch, within the series.

```python
from fpn.bifpn import BiFPN

fpn = BiFPN(3, 6, backbone.widths[3: 8: 2], 256)
model.fpn = fpn
```

*Holy cow! Now you get EfficientNet-BiFPN-YoloV4.*  

(This is not Efficient-Yolo)

## Now We Are Supporting
Now, we are supporting clasification models and one-stage detection models using anchors.  

### Classification
* ResNet - https://arxiv.org/abs/1512.03385
* Wide-ResNet - https://arxiv.org/abs/1605.07146
* ResNeXt - https://arxiv.org/abs/1611.05431
* Res2Net - https://arxiv.org/abs/1904.01169
    * Res2NeXt
    * Res2Net-v1b
* VGG - https://arxiv.org/abs/1409.1556
* MobileNetV1 - https://arxiv.org/abs/1704.04861
* MobileNetV2 - https://arxiv.org/abs/1801.04381
* MNASNet - https://arxiv.org/abs/1807.11626
* MobileNetV3 - https://arxiv.org/abs/1905.02244
    * MobileNetV3-Large
    * MobileNetV3-Small
* EfficientNet - https://arxiv.org/abs/1905.11946
* ReXNet - https://arxiv.org/abs/2007.00992

### Detection
* YoloV3 - https://arxiv.org/abs/1804.02767
    * YoloV3-SPP
    * YoloV3-Tiny
* YoloV4 - https://arxiv.org/abs/2004.10934
* Scaled-YoloV4 - https://arxiv.org/abs/2011.08036
    * YoloV4-CSP
    * YoloV4-Large-P5/P6/P7
    * YoloV4-Tiny
* SSD - https://arxiv.org/abs/1512.02325
    * SSD300/512
* SSD FPN
* DSSD - https://arxiv.org/abs/1701.06659
    * SSD321/513
    * DSSD321/513
* SSDLite - https://arxiv.org/abs/1801.04381
    * MobileNetV2
    * MobileNetV3-Large
    * MobileNetV3-Small
* RetinaNet - https://arxiv.org/abs/1708.02002
* EfficientDet -https://arxiv.org/abs/1911.09070

### New Model
* Efficient-Yolo
* RetinaNet-Res2Net-PAN
* SSDLite-ReXNet-BiFPN

We are developing training and inference frameworks applicable to every models.  

More image models such as vision transformers, two-stage detection models and non-anchor detection models, segmentation models, will be updated.  

### Checkpoints
The clasification models are provided with checkpoints trained on the ImageNet. For detection, only YOLO and SSD models are provided with checkpoints trained in COCO dataset.  

The checkpoints are borrowed from the official or famously forked repositories with permissive licenses (Apache, BSD, MIT), annotated in each class, and they are fully tested.

TensorBricks' own-trained checkpoints will be released in the near future.  


## What's New
### Ver. 1.0
* Mar 7, 2022
* Start Uploading

### Ver. 1.1
* Mar 23, 2022
* Upload Efficient-Yolo / RetinaNet-Res2Net-PAN / SSDLite-ReXNet-BiFPN


## License
This TensorBricks distribution contains no code licensed under GPL or other kinds of CopyLeft license. It is licensed under BSD-3-Clause which is permissive.
