from data.utils import *
from torchvision.datasets.vision import VisionDataset
from pycocotools.coco import COCO


class CocoDetection(VisionDataset):
    """
    __version__ = 1.2
    __date__ = Jun 3, 2022

    Created by hotcouscous1.

    The main difference of this dataset from torchvsion's one is that it loads images with OpenCV,
    which is faster than PIL and directly applicable to transformations of Albumentations.

    COCO has 1-91 category_ids, but eleven of them are for stuff tasks only.
    To exclude them with less cost, new category_ids 0-80 are accessed by self.cat_table,
    a dictionary of {old category: new category, ...}.

    For the case that a label is not given, BBox_Transformer can treat is without error.
    """

    num_classes = 80
    coco_cat = (i for i in range(1, 91))
    missing_cat = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)


    def __init__(self,
                 root: str,
                 annFile: str,
                 Transformer=None):

        super(CocoDetection, self).__init__(root)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transformer = Transformer
        self.cat_table = category_filter(self.coco_cat, self.missing_cat)


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]

        img_path = coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, img_path))
        target = coco.loadAnns(coco.getAnnIds(imgIds=img_id))


        bboxes, category_ids = [], []

        for i, t in enumerate(target):
            bboxes.append(t['bbox'])
            category_ids.append(t['category_id'])

        if self.transformer:
            transform = self.transformer(image, bboxes, category_ids)
            image, bboxes, category_ids = transform.values()

        for i, cat_id in enumerate(category_ids):
            new_id = self.cat_table[cat_id]
            category_ids[i] = make_one_hot(self.num_classes, new_id)


        bboxes = torch.from_numpy(np.asarray(bboxes))

        if category_ids:
            category_ids = torch.stack(category_ids)
        else:
            category_ids = torch.tensor([], dtype=torch.int8, device=device)

        return image, bboxes, category_ids