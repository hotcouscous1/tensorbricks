from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_metric(
        resultFile: str,
        annFile: str):

    "It returns the result of all COCO metrics."

    annType = 'bbox'
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resultFile)
    imgIds = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    summary = [round(i.item(), 3) for i in cocoEval.stats]
    summary = {'AP': summary[0],
               'AP50': summary[1],
               'AP75': summary[2],
               'APsmall': summary[3],
               'APmedium': summary[4],
               'APlarge': summary[5],
               'AR1': summary[6],
               'AP10': summary[7],
               'AR': summary[8],
               'ARsmall': summary[9],
               'ARmedium': summary[10],
               'ARlarge': summary[11]}

    return summary
