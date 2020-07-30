import copy
import time
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np

from .builder import DATASETS
from .coco import CocoDataset


class Tianchi:

    def __init__(self, annotation_file=None):
        # load dataset
        self.dataset = dict()
        self.anns = dict()
        self.cats = dict()
        self.studies, self.dicoms = dict(), dict()

        self.study_uid_id_map = dict()
        self.dicom_uid_id_map = dict()
        self.study_to_anns = defaultdict(list)
        self.study_to_dicoms = defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = mmcv.load(annotation_file)
            assert type(
                dataset
            ) == dict, f'annotation file format {type(dataset)} not supported'
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.creatIndex()

    def creatIndex(self):
        # create index
        print('creating index...')
        anns, cats, studies, dicoms = {}, {}, {}, {}
        study_uid_id_map = {}
        dicom_uid_id_map = {}
        study_to_anns = defaultdict(list)
        study_to_dicoms = defaultdict(list)

        # without ground truths, dataset always have "studies" and "dicoms"
        # but there are no "series_uid" and "instance_uid" in study
        if 'studies' in self.dataset:
            for study in self.dataset['studies']:
                studies[study['id']] = study
                study_uid_id_map[study['study_uid']] = study['id']

        if 'dicoms' in self.dataset:
            for dicom in self.dataset['dicoms']:
                dicoms[dicom['id']] = dicom
                dicom_uid_id_map[dicom['instance_uid']] = dicom['id']

        if 'dicoms' in self.dataset and 'studies' in self.dataset:
            for dicom in self.dataset['dicoms']:
                study_to_dicoms[study_uid_id_map[dicom['study_uid']]].append(
                    dicom)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                study_to_anns[ann['study_id']].append(ann)
                anns[ann['id']] = ann

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        print('index created!')

        # create class members
        self.anns = anns
        self.cats = cats
        self.studies = studies
        self.dicoms = dicoms
        self.study_uid_id_map = study_uid_id_map
        self.dicom_uid_id_map = dicom_uid_id_map
        self.study_to_anns = study_to_anns
        self.study_to_dicoms = study_to_dicoms

    def get_cat_ids(self):
        """Returns all category ids."""
        # return [cat['id'] for cat in self.dataset['categories']]
        return list(self.cats.keys())

    def get_study_ids(self):
        """Returns all study ids."""
        return list(self.studies.keys())

    def get_dicom_ids(self, study_id=None):
        """Get dicom ids that satisfy given filter conditions.

        Args:
            study_id: get dicoms for given study id.

        Returns:
            ids: integer list of dicom ids.
        """
        if study_id is None:
            return list(self.dicoms.keys())
        else:
            dicoms = self.study_to_dicoms[study_id]
            return [dicom['id'] for dicom in dicoms]

    def get_ann_ids(self, study_id):
        anns = self.study_to_anns[study_id]
        ann_ids = [ann['id'] for ann in anns]
        return ann_ids

    def load_study(self, study_id):
        return self.studies[study_id]

    def load_dicoms(self, ids=[]):
        if type(ids) == int:
            ids = [ids]
        return [self.dicoms[_id] for _id in ids]

    def load_anns(self, ids=[]):
        if type(ids) == int:
            ids = [ids]
        return [self.anns[_id] for _id in ids]


@DATASETS.register_module()
class TianchiImageDataset(CocoDataset):
    CLASSES = (('disc', 'v1'), ('disc', 'v2'), ('disc', 'v3'), ('disc', 'v4'),
               ('disc', 'v5'), ('vertebra', 'v1'), ('vertebra', 'v2'))
    PARTS = ('T12-L1', 'L1', 'L1-L2', 'L2', 'L2-L3', 'L3', 'L3-L4', 'L4',
             'L4-L5', 'L5', 'L5-S1')

    def load_annotations(self, ann_file):

        self.tianchi = Tianchi(ann_file)
        self.cat_ids = self.tianchi.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.study_ids = self.tianchi.get_study_ids()

        data_infos = []
        for study_id in self.study_ids:
            info = self.tianchi.load_study(study_id)

            if 'instance_uid' in info:
                dicom_id = self.tianchi.dicom_uid_id_map[info['instance_uid']]
                dicom_info = copy.deepcopy(
                    self.tianchi.load_dicoms(dicom_id)[0])
                dicom_id = dicom_info.pop('id')
                dicom_info['dicom_id'] = dicom_id
                info.updata(dicom_info)

            data_infos.append(info)
        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if min(data_info['width'], data_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        study_id = self.data_infos[idx]['id']
        ann_ids = self.tianchi.get_ann_ids(study_id)
        ann_info = self.tianchi.load_anns(ann_ids)

        return self._parse_ann_info(ann_info)

    def _parse_ann_info(self, ann_info):
        gt_points = []
        gt_bboxes = []
        gt_labels = []
        part_inds = []

        for i, ann in enumerate(ann_info):
            if ann['identification'] not in self.PARTS:
                continue
            part_idx = self.PARTS.index(ann['identification'])
            part_inds.append(part_idx)
            gt_points.append(ann['point'])
            gt_bboxes.append(self.center2bbox(ann['point']))
            gt_labels.append(self.cat2label[ann['category_id']])

        if gt_points:
            gt_points = np.array(gt_points, dtype=np.float32)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            part_inds = np.array(part_inds, dtype=np.int64)
        else:
            gt_points = np.zeros((0, 2), dtype=np.float32)
            gt_bboxes = np.zeros((0, 2), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            part_inds = np.array([], dtype=np.int64)

        ann = dict(
            points=gt_points,
            bboxes=gt_bboxes,
            labels=gt_labels,
            part_inds=part_inds)

        return ann

    def prepare_train_img(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=data_info, ann_info=ann_info)
        self.pre_pipeline(results)

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        data_info = self.data_infos[idx]
        results = dict(img_info=data_info)
        self.pre_pipeline(results)

        return self.pipeline(results)

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        # img prefix
        results['img_prefix'] = osp.join(results['img_prefix'],
                                         results['img_info']['study_idx'])
        results['point_fields'] = []

    def center2bbox(self, point, width=20, height=20):
        x1 = point[0] - width / 2.
        y1 = point[1] - height / 2.
        x2 = point[0] + width / 2.
        y2 = point[1] + width / 2.

        return [x1, y1, x2, y2]
