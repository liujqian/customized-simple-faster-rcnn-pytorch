import json
import os

import numpy as np
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

if __name__ == '__main__':
    img = read_image('datasets/webis-webseg-20/webis-webseg-20-screenshots/009514.png')
    img = t.from_numpy(img)[None]
    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn.rpn.proposal_layer.nms_thresh = 0.3
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('checkpoints-rpn-only/fasterrcnn_epoch_best_best_map_0.13068130595527067')
    opt.caffe_pretrain = False  # this model was trained from caffe-pretrained model
    original_roi,roi_scores = trainer.faster_rcnn.predict(img, visualize=True)
    bbox_polygon_list = []
    for bbox in original_roi[:20]:  # (y_{min}, x_{min}, y_{max}, x_{max})
        bbox_int = bbox.astype(np.int32)
        left = bbox_int[1]
        top = bbox_int[0]
        right = bbox_int[3]
        bottom = bbox_int[2]
        bbox_polygon_list.append(
            [
                [
                    [
                        [left.item(), top.item()],
                        [left.item(), bottom.item()],
                        [right.item(), bottom.item()],
                        [right.item(), top.item()],
                        [left.item(), top.item()],
                    ],
                ],
            ],
        )
    out_obj = dict(
        height=img.shape[2],
        width=img.shape[3],
        id="009514",
        segmentations=dict(
            mmdetection_bboxes=bbox_polygon_list,
        ),
    )
    with open("temp/009514_rois.json", "w") as handle:
        json.dump(out_obj, handle)
