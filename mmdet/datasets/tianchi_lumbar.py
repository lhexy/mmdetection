import copy
import os.path as osp

import mmcv
import numpy as np

from .builder import DATASETS
from .coco import CocoDataset
from .tianchi import Tianchi


@DATASETS.register_module()
class TianchiImageDataset(CocoDataset):
    CLASSES = ('disc: v1', 'disc: v2', 'disc: v3', 'disc: v4', 'disc: v5',
               'vertebra: v1', 'vertebra: v2')
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
                info.update(dicom_info)

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
        gt_labels = []
        part_inds = []

        for i, ann in enumerate(ann_info):
            if ann['identification'] not in self.PARTS:
                continue
            part_idx = self.PARTS.index(ann['identification'])
            part_inds.append(part_idx)
            gt_points.append(ann['point'])
            gt_labels.append(self.cat2label[ann['category_id']])

        if gt_points:
            gt_points = np.array(gt_points, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            part_inds = np.array(part_inds, dtype=np.int64)
        else:
            gt_points = np.zeros((0, 2), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            part_inds = np.array([], dtype=np.int64)

        ann = dict(points=gt_points, labels=gt_labels, part_inds=part_inds)

        return ann

    def prepare_train_img(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=data_info, ann_info=ann_info)
        self.pre_pipeline(results)

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        # TODO: smart prepare test img without ext info
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

    def bbox2center(self, bbox):
        center_x = bbox[0] + bbox[2] / 2.
        center_y = bbox[1] + bbox[3] / 2.

        return [center_x, center_y]

    def _det2json(self, results):
        json_results = []
        for idx in range(len(self)):
            study_id = self.study_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['study_id'] = study_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['point'] = self.bbox2center(bboxes[i][:4])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _tianchi_format(self, results):
        """

        Output Format:
            [{
                'data': [{
                    'annotation': [{
                        'annotator': 54,
                        'data': {
                            'point': [{
                                'coord': [320, 76],
                                'tag': {'disc': 'v1', 'identification': 'xx'},
                                'zIndex': 6}, ...]}
                    }],
                    'instanceUid': 'xxxxx',
                    'seriesUid': 'xxxxx'
                }],
                'studyUid': 'xxxxx',
                'version': 'v0.1'
            }, ...]
        """
        default_version = 'v0.1'  # version key in tianchi annotation
        default_annotator = 44
        default_zIndex = 4

        json_results = []
        for idx in range(len(self)):
            study_id = self.study_ids[idx]
            study_result = dict(version=default_version)
            study_info = self.tianchi.load_study(study_id)
            study_result['studyUid'] = study_info['study_uid']

            result = results[idx]
            point_list = []
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['coord'] = list(
                        map(int, self.bbox2center(bboxes[i][:4])))
                    data['tag'] = dict()
                    category = self.CLASSES[label]
                    tag = category.split(':')[0].strip()
                    code = category.split(':')[1].strip()
                    data['tag'][tag] = code
                    data['tag']['identification'] = 'L1-L2'
                    data['zIndex'] = default_zIndex
                    data['score'] = float(bboxes[i][4])

                    if data['score'] > 0.7:
                        point_list.append(data)
            annotation = [
                dict(annotator=default_annotator, data=dict(point=point_list))
            ]

            if 'instance_uid' in study_info:
                instance_uid = study_info['instance_uid']
            else:
                instance_uid = study_info['study_uid'] + '.1234567.123'

            if 'series_uid' in study_info:
                series_uid = study_info['series_uid']
            else:
                series_uid = study_info['study_uid'] + '.1234567'

            study_data = [
                dict(
                    annotation=annotation,
                    instanceUid=instance_uid,
                    seriesUid=series_uid)
            ]
            study_result['data'] = study_data
            json_results.append(study_result)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the results to coco and Tianchi-specified style json file.

        This method is only able to deal with bbox (or point) predictions,
        and dump them to json file.

        Args:
            results (list): Testing results of Tianchi dataset.
            outfile_prefix (str): The filename prefix of the json file. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.bbox.json", "somepath/xxx.point.json"

        Returns:
            dict[str: str]: Possible keys are "bbox", "point", and
                values are corresponding filenames.
        """
        result_files = dict()
        json_results = self._det2json(results)
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        mmcv.dump(json_results, result_files['bbox'])

        # write tianchi's result json
        tianchi_json_results = self._tianchi_format(results)
        result_files['tianchi'] = f'{outfile_prefix}.tianchi.json'
        mmcv.dump(tianchi_json_results, result_files['tianchi'])

        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for Tianchi)

        Args:
            results (list): Testing results of Tianchi dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if jsonfile_prefix is None:
            import tempfile
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_file = self.results2json(results, jsonfile_prefix)
        return result_file, tmp_dir
