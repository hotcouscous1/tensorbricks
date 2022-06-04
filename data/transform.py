import albumentations as A
import cv2
import numpy as np
import torch


# Albumentations is an image augmentation library using OpenCV + Numpy, which is faster than PIL + torchvision.transforms.
# documentation : https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation
# github : https://github.com/albumentations-team/albumentations
#
# Install opencv-python-headless version 4.1.2.30 in your virtual environment for stable use of Albimentations;
# pip install opencv-python-headless==4.1.2.30


class BBox_Transformer(object):

    """
    __version__ = 1.2
    __date__ = Jun 3, 2022

    Created by hotcouscous1.

    This module is for flexible composition of Albumentations in object detection.

    T = BBox_Transformer(t_prob=0.5, min_area=2048, min_visibility=0.1)

    T.append(A.RandomResizedCrop(256, 512, (0.1, 1.0), p=1))
    T.append(A.HorizontalFlip(p=0.5))
    T.append(A.RandomBrightnessContrast(p=1))
    T.make_compose()

    t_image, t_bboxes, t_category_ids = T(image, bboxes, category_ids).values()

    if label(bboxes and category_ids) is not given, it returns transformed images and empty lists for label.

    ----------------------------------------------------------------------------------------------------------------

    t_prob : the overall probability that any transformation is applied.
            If the probability of one of transformations, for example, A.HorizontalFlip(p=0.2) is 0.2,
            it will apply with a probability of t_prob * 0.2.

    dataset : the bbox format of each dataset
            pascal_voc : [x_min, y_min, x_max, y_max]
            coco : [x_min, y_min, width, height]
            yolo : [x_center, y_center, width, height], normalized by image size
            albumentation : [x_min, y_min, x_max, y_max], normalized by image size

    min_area : a valule of bbox width * bbox height, which is min area of bbox
            not to be dropped after transformations

    min_visibility : a value between 0-1. If the ratio of the bbox area after augmentation to the area before augmentation
    becomes smaller than this, the bbox will be dropped.

    normalize : bound all pixel values between 0-1

    dataset_stat : normalize pixel values by mean and std along RGB channels

    ToTensor : change from HWC(cv2 or PIL) to CHW(Tensor) format, and Numpy to Tensor
    """

    def __init__(self,
                 t_prob: float = 0.5,
                 dataset: str = 'coco',
                 min_area: float = 0,
                 min_visibility: float = 0,
                 normalize: bool = True,
                 dataset_stat: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ToTensor: bool = True
                 ):

        self.transforms = []
        self.Compose = None

        self.t_prob = t_prob
        self.dataset = dataset
        self.min_area = min_area
        self.min_visibility = min_visibility

        self.normalize = normalize
        self.dataset_stat = dataset_stat
        self.ToTensor = ToTensor

        if dataset_stat:
            self.mean, self.std = dataset_stat

            if self.normalize:
                self.normalizer = A.Normalize(self.mean, self.std, 1.0)
            else:
                self.normalizer = A.Normalize(self.mean, self.std)



    def append(self, albumentation):
        self.transforms.append(albumentation)


    def remove(self, albumentation):
        self.transforms.remove(albumentation)


    def make_compose(self):
        bbox_params = A.BboxParams(self.dataset, label_fields=['category_ids'], min_area=self.min_area, min_visibility=self.min_visibility)
        self.Compose = A.Compose(self.transforms, bbox_params=bbox_params, p=self.t_prob)


    def init_compose(self):
        self.Compose = None


    def __call__(self, image, bboxes, category_ids):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not (bboxes and category_ids):
            bboxes, category_ids = [], []

        if self.Compose:
            output = self.Compose(image=image, bboxes=bboxes, category_ids=category_ids)
        else:
            output = {'image': image, 'bboxes': bboxes, 'category_ids': category_ids}

        if self.normalize:
            output['image'] = output['image'] / 255

        if self.dataset_stat:
            output['image'] = self.normalizer(image=output['image'])['image']

        if self.ToTensor:
            output['image'] = np.transpose(output['image'], (2, 0, 1))
            output['image'] = torch.from_numpy(output['image'])

        if not (bboxes and category_ids):
            del output['bboxes']
            del output['category_ids']

        return output
