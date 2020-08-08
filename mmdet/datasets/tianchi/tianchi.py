import copy
import itertools
import time
from collections import defaultdict

import mmcv


class Tianchi:
    """Constructor of Tianchi lumbar class for reading annotations
    like COCO.

    Args:
        annotation_file (str): Path to annotation file.
    """

    def __init__(self, annotation_file=None):
        # load dataset
        self.dataset = dict()
        self.anns = dict()
        self.cats = dict()
        self.studies = dict()
        self.series = dict()
        self.dicoms = dict()

        # mapping from "studyUid" to "study_id"
        self.study_uid_id_map = dict()
        # mapping from "seriesUid" to "series_id"
        self.series_uid_id_map = dict()
        # mapping from "instanceUid" to "dicom_id" of DICOM files
        self.dicom_uid_id_map = dict()
        # mapping from "study_id" to its series
        self.study_to_series = defaultdict(list)
        # mapping from "study_id" to its dicoms
        self.study_to_dicoms = defaultdict(list)
        # mapping from "study_id" to its anns
        self.study_to_anns = defaultdict(list)

        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = mmcv.load(annotation_file)
            assert type(
                dataset
            ) == dict, f'annotation file format {type(dataset)} not supported'
            print(f'Done (t={time.time() - tic:0.2f}s)')
            self.dataset = dataset
            self.creatIndex()

    def creatIndex(self):
        # create index
        print('creating index...')

        anns = dict()
        cats = dict()
        studies = dict()
        series = dict()
        dicoms = dict()

        study_uid_id_map = dict()
        series_uid_id_map = dict()
        dicom_uid_id_map = dict()
        study_to_series = defaultdict(list)
        study_to_dicoms = defaultdict(list)
        study_to_anns = defaultdict(list)

        # Without ground truths, the dataset always have "studies" and "dicoms"
        # but there are no "series_uid" and "instance_uid" in study
        if 'studies' in self.dataset:
            for study in self.dataset['studies']:
                studies[study['id']] = study
                study_uid_id_map[study['study_uid']] = study['id']

        if 'series' in self.dataset:
            for _series in self.dataset['series']:
                series[_series['id']] = _series
                series_uid_id_map[_series['series_uid']] = _series['id']

        if 'dicoms' in self.dataset:
            for dicom in self.dataset['dicoms']:
                dicoms[dicom['id']] = dicom
                dicom_uid_id_map[dicom['instance_uid']] = dicom['id']

        if 'series' in self.dataset and 'studies' in self.dataset:
            for _series in self.dataset['series']:
                # mapping from "study_id" to its series
                study_to_series[study_uid_id_map[_series['study_uid']]].append(
                    _series)

        if 'dicoms' in self.dataset and 'studies' in self.dataset:
            for dicom in self.dataset['dicoms']:
                # mapping from "study_id" to its dicoms
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
        self.series = series
        self.dicoms = dicoms
        self.study_uid_id_map = study_uid_id_map
        self.series_uid_id_map = series_uid_id_map
        self.dicom_uid_id_map = dicom_uid_id_map
        self.study_to_series = study_to_series
        self.study_to_dicoms = study_to_dicoms
        self.study_to_anns = study_to_anns

    def get_study_ids(self):
        """Returns all study ids."""
        return list(self.studies.keys())

    def get_series_ids(self, ids=[]):
        """Get series ids with study ids.

        Args:
            ids (int or list[int]): study ids, get all ids if not given.

        Returns:
            list: series ids of each study.
        """
        ids = [ids] if type(ids) == int else ids
        if len(ids) == 0:
            series = self.dataset['series']
        else:
            series = [self.study_to_series[_id] for _id in ids]
            series = list(itertools.chain.from_iterable(series))

        return [_ser['id'] for _ser in series]

    def get_dicom_ids(self, study_ids=[], series_ids=[]):
        """Get dicom ids with given study and series ids."""
        study_ids = [study_ids] if type(study_ids) == int else study_ids
        series_ids = [series_ids] if type(series_ids) == int else series_ids

        if len(study_ids) == 0 and len(series_ids) == 0:
            # return all dicom ids
            dicoms = self.dataset['dicoms']
        else:
            dicoms = [self.study_to_dicoms[_id] for _id in study_ids]
            dicoms = list(itertools.chain.from_iterable(dicoms))
            if not len(series_ids) == 0:
                uids = [
                    self.series[series_id]['series_uid']
                    for series_id in series_ids
                ]
                dicoms = [
                    dicom for dicom in dicoms if dicom['series_uid'] in uids
                ]

        return [dicom['id'] for dicom in dicoms]

    def get_ann_ids(self, study_ids=[]):
        """Get ann ids that satisfy given filter conditions.
        Default skips that filter.

        Args:
            study_ids (int or list): study ids to get anns.

        Returns:
            ids (list): ann ids
        """
        study_ids = [study_ids] if type(study_ids) == int else study_ids

        if len(study_ids) == 0:
            anns = self.dataset['annotations']
        else:
            lists = [
                self.study_to_anns[_id] for _id in study_ids
                if _id in self.study_to_anns
            ]
            anns = list(itertools.chain.from_iterable(lists))

        return [ann['id'] for ann in anns]

    def get_cat_ids(self):
        """Returns all category ids."""
        return list(self.cats.keys())

    def load_studies(self, ids=[]):
        """Load study infos with given study ids."""
        ids = [ids] if type(ids) == int else ids

        return [self.studies[_id] for _id in ids]

    def load_series(self, ids=[]):
        """Load series infos with given series ids."""
        ids = [ids] if type(ids) == int else ids

        return [self.series[_id] for _id in ids]

    def load_dicoms(self, ids=[]):
        """Load dicom infos with given dicom ids."""
        ids = [ids] if type(ids) == int else ids

        return [self.dicoms[_id] for _id in ids]

    def load_anns(self, ids=[]):
        ids = [ids] if type(ids) == int else ids

        return [self.anns[_id] for _id in ids]

    def load_results(self, result_file):
        """Load result file and return a result api object.
        """
        res = Tianchi()
        res.dataset['studies'] = copy.deepcopy(self.dataset['studies'])
        res.dataset['series'] = copy.deepcopy(self.dataset['series'])
        res.dataset['dicoms'] = copy.deepcopy(self.dataset['dicoms'])

        print('Loading and preparing results...')
        tic = time.time()
        anns = mmcv.load(result_file)
        assert type(anns) == list

        anns_study_ids = [ann['study_id'] for ann in anns]
        assert set(anns_study_ids) == (
            set(anns_study_ids) & set(self.get_study_ids()))

        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for _id, ann in enumerate(anns):
            ann['id'] = id + 1
        print(f'DONE (t={time.time() - tic:0.2f}s)')

        res.dataset['annotations'] = anns
        res.creatIndex()
        return res
