import argparse
import cv2
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import Config

from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('results', help='.pkl results')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/visualizations/',
        help='path to store outputs')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    assert args.results.endswith(
        '.pkl', ), ValueError('The results file should be a pkl file.')
    results = mmcv.load(args.results)

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    assert len(results) == len(dataset), ValueError(
        'The length of results is not equal to that of dataset:'
        f' {len(results)} != {len(dataset)}.')

    for result_idx, _results in enumerate(results):
        assert len(_results) == len(dataset.CLASSES)

        data = dataset[result_idx]
        img_path = data['img_metas'][0].data['filename']
        img = mmcv.imread(img_path)

        for i, part in enumerate(dataset.CLASSES):
            bboxes = _results[i]
            _scores = bboxes[:, 4]
            if _scores.shape[0] == 0:
                continue
            idx = np.argmax(_scores)

            point = dataset.bbox2center(bboxes[idx][:4])
            point = tuple(map(int, point))

            img = cv2.circle(img, point, 2, mmcv.color_val('red'))
            bbox = tuple(map(int, bboxes[idx][:4]))
            # img = cv2.rectangle(img, bbox[:2], bbox[2:],
            #                     mmcv.color_val('green'))
            label_text = f'{dataset.CLASSES[i]}'
            loc = (bbox[2], point[1] + 10)
            img = cv2.putText(img, label_text, loc, cv2.FONT_HERSHEY_COMPLEX,
                              0.5, mmcv.color_val('green'))
            # for bbox_idx in range(bboxes.shape[0]):
            #     # score = bboxes[bbox_idx][4]
            #     point = dataset.bbox2center(bboxes[bbox_idx][:4])
            #     point = tuple(map(int, point))
            #     # print(dataset.CLASSES[i], score, point)
            #     img = cv2.circle(img, point, 2, mmcv.color_val('red'))
            #     bbox = tuple(map(int, bboxes[bbox_idx][:4]))
            #     img = cv2.rectangle(img, bbox[:2], bbox[2:],
            #                         mmcv.color_val('green'))

        mmcv.imwrite(img, osp.join(args.out_dir, f'image{result_idx}.jpg'))


if __name__ == "__main__":
    main()
