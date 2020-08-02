import copy
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
        self.studies, self.dicoms = dict(), dict()

        # mapping from "studyUid" to "study_id"
        self.study_uid_id_map = dict()
        # mapping from "instanceUid" to "dicom_id" of DICOM files
        self.dicom_uid_id_map = dict()
        # mapping from "study_id" to its anns
        self.study_to_anns = defaultdict(list)
        # mapping from "study_id" to its dicoms
        self.study_to_dicoms = defaultdict(list)

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
        anns, cats, studies, dicoms = {}, {}, {}, {}
        study_uid_id_map = {}
        dicom_uid_id_map = {}
        study_to_anns = defaultdict(list)
        study_to_dicoms = defaultdict(list)

        # Without ground truths, the dataset always have "studies" and "dicoms"
        # but there are no "series_uid" and "instance_uid" in study
        if 'studies' in self.dataset:
            for study in self.dataset['studies']:
                studies[study['id']] = study
                study_uid_id_map[study['study_uid']] = study['id']

        if 'dicoms' in self.dataset:
            for dicom in self.dataset['dicoms']:
                dicoms[dicom['id']] = dicom
                dicom_uid_id_map[dicom['instance_uid']] = dicom['id']

        # always have them both
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
        self.dicoms = dicoms
        self.study_uid_id_map = study_uid_id_map
        self.dicom_uid_id_map = dicom_uid_id_map
        self.study_to_anns = study_to_anns
        self.study_to_dicoms = study_to_dicoms

    def get_cat_ids(self, cat_names=[], cat_ids=[]):
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

    def load_result(self, result_file):
        """Load result file and return a result api object.

        Args:
            result_file (str): Filename of result file.

        Returns:
            result (self)
        """
        res = Tianchi()
        res.dataset['studies'] = [study for study in self.dataset['stuies']]

        print('loading and preparing results...')
        tic = time.time()
        assert type(result_file) == str, TypeError(
            f'result file must be str type path, but got {type(result_file)}')
        anns = mmcv.load(result_file)
        assert type(anns) == list, TypeError(
            f'results must be list type, but got {type(anns)}')
        anns_study_ids = [ann['study_id'] for ann in anns]
        assert set(anns_study_ids) == (
            set(anns_study_ids) & set(self.get_study_ids())
        ), ValueError('Results do not correspond to current dataset')

        if 'point' in anns[0] and not anns[0]['point'] == []:
            res.dataset['categories'] = copy.deepcopy(
                self.dataset['categories'])
            for id, ann in enumerate(anns):
                point = ann['point']

        print(f'Done (t={time.time() - tic:0.2f}s)')
        res.dataset['annotations'] = anns
        res.creatIndex()
        return res
