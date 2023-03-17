from __future__ import absolute_import
import os
import time

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, ValidationDataset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader), total=len(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        rois = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    valset = ValidationDataset(opt)
    validation_dataloader = data_.DataLoader(valset,
                                             batch_size=1,
                                             num_workers=opt.test_num_workers,
                                             shuffle=False, \
                                             pin_memory=True
                                             )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    start_time = time.time()
    for epoch in range(opt.epoch):
        print(f"Training for the {epoch + 1}th epoch now!")
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader), total=len(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

        eval_result = eval(validation_dataloader, faster_rcnn, test_num=opt.test_num)
        print(f"The mAP after the {epoch + 1}th epoch is {eval_result['map']}! The mAP so far is {best_map}!")
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        best_path = trainer.save(epoch=epoch, save_optimizer=True)
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map, epoch="best", save_optimizer=True)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            print(f"The total time in seconds used for 14 epochs of training is {time.time() - start_time}")
            break


# one epoch ranges from 8:27-8:34 for the original implementation.
if __name__ == '__main__':
    import fire

    fire.Fire()
