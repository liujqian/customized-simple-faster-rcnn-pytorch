import json
import os
import time

import numpy as np
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

if __name__ == '__main__':
    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn.rpn.proposal_layer.nms_thresh = 0.3
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('checkpoints-rpn-only/fasterrcnn_epoch_best_best_map_0.13068130595527067')
    ids = [d.name.split(".")[0] for d in os.scandir("datasets/webis-webseg-20/webis-webseg-20-screenshots") if
           int(d.name.split(".")[0]) > 9487]
    inference_times = []
    for i, img_id in enumerate(ids):
        img = read_image(f'datasets/webis-webseg-20/webis-webseg-20-screenshots/{img_id}.png')
        img = t.from_numpy(img)[None]
        opt.caffe_pretrain = False  # this model was trained from caffe-pretrained model
        inference_start_time = time.time()
        original_roi, roi_scores = trainer.faster_rcnn.predict(img, visualize=True)
        inference_times.append(time.time() - inference_start_time)
        print(f"Just finished the {i}th inference out of {len(ids)}. This one took {inference_times[-1]}seconds!")
        bbox_polygon_list = []
        for bbox in original_roi[:100]:  # (y_{min}, x_{min}, y_{max}, x_{max})
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
            id=img_id,
            segmentations=dict(
                faster_rcnn_rpn_only_bboxes=bbox_polygon_list,
            ),
        )
        with open(f"inferences/rpn-only/screenshots/{img_id}.json", "w") as handle:
            json.dump(out_obj, handle)
        # The average inference time is 0.07484022870447964
    print(f"The average inference time is {sum(inference_times) / len(inference_times)}")
