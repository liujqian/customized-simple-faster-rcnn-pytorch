import json
import os
import pickle
import time

import numpy as np

from .util import read_image


class CocoBboxDataset:
    def __init__(
            self,
            data_dir="datasets/webis-webseg-20",
            split='train',
            data_pickle="",
            use_difficult=False,
            return_difficult=False,
    ):
        start_time = time.time()
        print("Starting to build a dataset object from the coco annotations!")
        if data_pickle != "":
            with open(data_pickle, "rb") as handle:
                self.data = pickle.load(handle)
        else:
            with open(os.path.join(data_dir, f"coco-formatted-info-{split}.json"), "r") as handle:
                coco = json.load(handle)
            images = coco["images"]
            annos = coco["annotations"]
            self.data = []
            for img in images:
                img_id = img["id"]
                datum = {"image_info": img, "annotations": []}
                for anno in annos:
                    if anno["image_id"] == img_id:
                        original_bbox = anno["bbox"]
                        datum["annotations"].append(
                            {
                                "image_id": img_id,
                                "category": anno["category_id"],
                                "bbox": [
                                    original_bbox[1],
                                    original_bbox[0],
                                    original_bbox[1] + original_bbox[3],
                                    original_bbox[0] + original_bbox[2]
                                ]
                            }
                        )
                self.data.append(datum)
        self.data_dir = data_dir
        self.label_names = BBOX_LABEL_NAMES
        print(f"Finished building a dataset object from the coco annotations! Used {time.time() - start_time} seconds.")

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        datum = self.data[i]
        bbox = [annotation["bbox"] for annotation in datum["annotations"]]
        label = [annotation["category"] for annotation in datum["annotations"]]
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = [False] * len(label)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'webis-webseg-20-screenshots', datum["image_info"]["file_name"])
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example


BBOX_LABEL_NAMES = (
    'webpage-segmentation'
)
