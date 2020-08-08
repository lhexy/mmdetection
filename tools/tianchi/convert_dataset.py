import argparse
import glob
import os
import os.path as osp
import SimpleITK as sitk
from collections import defaultdict, Counter

import mmcv
import numpy as np

DISC_PARTS = ('T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1')
VERTEBRA_PARTS = ('L1', 'L2', 'L3', 'L4', 'L5')

CLASSES = ('T12-L1', 'L1', 'L1-L2', 'L2', 'L2-L3', 'L3', 'L3-L4', 'L4',
           'L4-L5', 'L5', 'L5-S1')
TAGS = ('disc: v1', 'disc: v2', 'disc: v3', 'disc: v4', 'disc: v5',
        'vertebra: v1', 'vertebra: v2')

OFFICIAL_ANNS = {
    'lumbar_train150': 'annotations/lumbar_train150_annotation.json',
    'lumbar_train51': 'annotations/lumbar_train51_annotation.json',
}

META_KEYS = {
    'study_uid': '0020|000d',
    'series_uid': '0020|000e',
    'instance_uid': '0008|0018',
    'series_description': '0008|103e',
    'image_position': '0020|0032',
    'image_orientation': '0020|0037',
    'slice_thickness': '0018|0050',
    'pixel_spacing': '0028|0030',
}


class Series:
    """Series object for handling series data info
    and some processing methods.

    Args:
        series_uid (str): series Uid
        dicoms (list[dict]): a list of dicom info dicts
    """

    def __init__(self, series_uid, dicoms):
        self._series_uid = series_uid
        self._dicoms = self._filter_dicoms(dicoms)

        if len(self._dicoms) == 0:
            self.valid = False
        else:
            self.valid = True

        if self.valid:
            # set the plane of the series
            planes = Counter([dicom['plane'] for dicom in self._dicoms])
            self._plane = planes.most_common(1)[0][0]

            # filter with specified plane
            self._dicoms = [
                dicom for dicom in self._dicoms
                if dicom['plane'] == self._plane
            ]
            # series description
            for dicom in self._dicoms:
                if 'series_description' in dicom:
                    _description = dicom['series_description']
                    break
                else:
                    _description = 'none'
            self._description = _description

            # positions
            positions = np.stack(
                [dicom['image_position'] for dicom in self._dicoms])
            if self._plane == 'sagittal':
                dim = 0  # x
            elif self._plane == 'coronal':
                dim = 1  # y
            elif self._plane == 'transverse':
                dim = 2  # z
            else:
                raise ValueError(f'Get wrong plane {self._plane}.')
            inds = np.argsort(positions[:, dim])
            self._positions = positions[inds]
            # sort dicoms with positions
            self._dicoms = [self._dicoms[idx] for idx in inds]

    @property
    def series_uid(self):
        return self._series_uid

    @property
    def plane(self):
        return self._plane

    @property
    def series_description(self):
        return self._description

    @property
    def positions(self):
        return self._positions

    def _filter_dicoms(self, dicoms):
        valids = []
        for i, dicom in enumerate(dicoms):
            if 'image_position' not in dicom:
                continue
            if 'plane' not in dicom:
                continue
            valids.append(dicom)

        return valids

    def anns(self, series_id, study_id, study_idx, study_uid):
        """Generate annotation for series."""
        data = dict(
            id=series_id,
            series_uid=self._series_uid,
            plane=self._plane,
            series_description=self._description,
            # study info
            study_id=study_id,
            study_idx=study_idx,
            study_uid=study_uid,
            # dicom info
            instance_uids=[dicom['instance_uid'] for dicom in self._dicoms])

        return data


class Study:

    @staticmethod
    def _parse_meta(meta):
        _meta = {}
        for keyname, meta_key in META_KEYS.items():
            if meta_key not in meta:
                continue
            _meta[keyname] = meta[meta_key]

        processing_items = [
            'image_position', 'image_orientation', 'slice_thickness',
            'pixel_spacing'
        ]
        for key in processing_items:
            if key not in _meta:
                continue
            _meta[key] = list(map(float, _meta[key].strip().split('\\')))

        if 'image_orientation' in _meta:
            # process meta info
            rd = np.array(_meta['image_orientation'][:3], dtype=np.float32)
            cd = np.array(_meta['image_orientation'][3:], dtype=np.float32)
            normal = np.cross(rd, cd)
            normal = np.abs(normal / np.linalg.norm(normal))

            thr = 0.8660254037844387  # pi / 6
            xd = np.array([1, 0, 0], dtype=np.float32)
            yd = np.array([0, 1, 0], dtype=np.float32)
            zd = np.array([0, 0, 1], dtype=np.float32)
            if np.abs(np.matmul(normal, xd)) > thr:
                plane = 'sagittal'
            elif np.abs(np.matmul(normal, yd)) > thr:
                plane = 'coronal'
            elif np.abs(np.matmul(normal, zd)) > thr:
                plane = 'transverse'
            else:
                plane = 'unclear'
            _meta['normal'] = normal.tolist()
            _meta['plane'] = plane

        return _meta

    @staticmethod
    def _load_all_dicoms(study_id, study_idx, dicom_filenames):
        # set DICOM file reader
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetImageIO('GDCMImageIO')

        dicoms_list = []
        for dicom_filename in dicom_filenames:
            # TODO: multi-processes
            reader.SetFileName(dicom_filename)
            try:
                reader.ReadImageInformation()
            except RuntimeError:
                # some DICOM files are unreadable
                continue

            _dicom_dict = dict(
                study_id=study_id,
                study_idx=study_idx,
                filename=osp.basename(dicom_filename))
            meta = {
                key: reader.GetMetaData(key)
                for key in reader.GetMetaDataKeys()
            }
            _dicom_dict.update(Study._parse_meta(meta))
            dicoms_list.append(_dicom_dict)

        return dicoms_list

    @staticmethod
    def fromfiles(study_id, path):
        """
        Args:
            study_id (int): study_id
            path (str): path of study dir, e.g., "somepath/stduy*/"

        Returns:
            Study
        """
        study_idx = osp.basename(path)

        # get all DICOM files in the study dir (path)
        dicom_filenames = glob.glob(osp.join(path, '*.dcm'))
        # dicoms_list: list[dict]
        dicoms_list = Study._load_all_dicoms(study_id, study_idx,
                                             dicom_filenames)

        return Study(study_id, study_idx, dicoms_list, path)

    def __init__(self, study_id, study_idx, dicoms, dicom_prefix=None):
        """
        Args:
            study_id (int): study id
            study_idx (str): "study*"
            dicoms (list[dict]): a list of dicom info dicts
            dicom_prefix(str, optional): prefix of the DICOM files
        """
        self._study_id = study_id
        self._study_idx = study_idx
        self._dicom_prefix = dicom_prefix
        self.dicoms = dicoms

        # for handling more fields
        self._fields = {}

        # series
        series_dict = defaultdict(list)
        for dicom in self.dicoms:
            series_dict[dicom['series_uid']].append(dicom)
        self.series_list = []
        self.series_planes = []
        self.series_descriptions = []
        for series_uid, _dicoms_list in series_dict.items():
            series = Series(series_uid, _dicoms_list)
            if series.valid:
                self.series_list.append(series)
                self.series_planes.append(series.plane)
                self.series_descriptions.append(series.series_description)
        self.num_series = len(self.series_list)

    @property
    def study_id(self):
        return self._study_id

    @property
    def study_idx(self):
        return self._study_idx

    @property
    def study_uid(self):
        assert hasattr(self, 'dicoms')
        return self.dicoms[-1]['study_uid']

    def _convert_to_jpg(self, dicom_path, img_path):
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetImageIO('GDCMImageIO')
        img = reader.Execute()

        img = sitk.GetArrayFromImage(img)[0]
        img = img.astype(np.float32)
        # output_pixel = (input_pixel - input_min) * \
        #   ((output_max - output_min) / (input_max - input_min)) + \
        #   output_min
        img = (img - img.min()) * (255.0 / (img.max() - img.min()))
        if reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            img = 255.0 - img

        img = np.ascontiguousarray(img)
        mmcv.imwrite(img, img_path)

        return img

    def read_or_convert_images(self, img_prefix):
        """
        Read images of study, or convert to JPG images if not exists.

        Args:
            img_prefix (str): path of stored JPG images
        """
        # imgs will be stored in ``{img_prefix}/{study_idx}``
        img_prefix = osp.join(img_prefix, self._study_idx)

        for dicom in self.dicoms:
            filename = dicom['filename']
            filename = filename.replace('.dcm', '.jpg')

            img_path = osp.join(img_prefix, filename)
            if osp.exists(img_path):
                img = mmcv.imread(img_path)
            else:
                # no JPG image, convert DICOM to JPG
                img = self._convert_to_jpg(
                    osp.join(self._dicom_prefix, dicom['filename']), img_path)
            height, width = img.shape[:2]
            # update dicom info
            dicom['filename'] = filename
            dicom['height'] = height
            dicom['width'] = width

    def update(self, key, value):
        self._fields[key] = value

    def anns(self):
        """Generate annotation for study."""
        data = dict(
            id=self._study_id,
            study_idx=self._study_idx,
            study_uid=self.study_uid)

        for key in ['series_uid', 'instance_uid']:
            if key in self._fields:
                data[key] = self._fields[key]

        return data


def get_cat_id(tag, code):
    """Get category id.

    Args:
        tag (str): Spine part in ["disc", "vertebra"].
        code (str): Disease code in ["v1", "v2", "v3", "v4", "v5"].

    Returns:
        category_id (int): 1-based category id in categories.
    """
    return CLASSES.index(': '.join((tag, code))) + 1


def load_official_anns(points, study_id, ann_id):
    """Parse official annotations and convert to coco-style
    annotations.

    Args:
        points (list[dict]): Annotations of each point on spine.
        study_id (int): 1-based study_id.
        ann_id (int): 1-based ann_id.

    Returns:
        anns_list (list[dict]): A list of annottions (dict) of
            each point.
        ann_id (int): increased ann_id.
    """
    anns_list = []
    for point in points:
        part = point['tag']['identification']
        if part in DISC_PARTS:
            tag = 'disc'
            assert tag in point['tag']
            code = point['tag'][tag]
            if code == '':
                code = 'v1'  # labeled as normal if missed
            if ',' in code:
                # some disc have multi labels
                for _code in code.split(','):
                    cat_id = CLASSES.index(part) + 1
                    _ann = dict(
                        id=ann_id,
                        study_id=study_id,
                        category_id=cat_id,
                        point=point['coord'],
                        identification=part,
                        tag=tag,
                        code=_code)
                    anns_list.append(_ann)
                    ann_id += 1
            else:
                cat_id = CLASSES.index(part) + 1
                _ann = dict(
                    id=ann_id,
                    study_id=study_id,
                    category_id=cat_id,
                    point=point['coord'],
                    identification=part,
                    tag=tag,
                    code=code)
                anns_list.append(_ann)
                ann_id += 1
        elif part in VERTEBRA_PARTS:
            tag = 'vertebra'
            assert tag in point['tag']
            code = point['tag'][tag]
            if code == '':
                code = 'v1'  # labeled as normal if missed
            cat_id = CLASSES.index(part) + 1
            _ann = dict(
                id=ann_id,
                study_id=study_id,
                category_id=cat_id,
                point=point['coord'],
                identification=part,
                tag=tag,
                code=code)
            anns_list.append(_ann)
            ann_id += 1

    return anns_list, ann_id


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to generate annotation file.')
    parser.add_argument(
        '--data-root',
        default='/data/tianchi',
        help='Path to Tianchi lumbar dataset.')
    parser.add_argument(
        '--dicom-prefix',
        default='lumbar_train150',
        help='Prefix of subset DICOM files (dirname).')
    parser.add_argument(
        '--img-prefix',
        type=str,
        default=None,
        help='Prefix of converted JPEG images.')
    parser.add_argument(
        '--official-ann',
        type=str,
        default=None,
        help='Path to official annotation file.')
    parser.add_argument(
        '--ann-file',
        type=str,
        default=None,
        help='Path of converted annotation file to be stored.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_root = args.data_root
    dicom_prefix = osp.join(data_root, args.dicom_prefix)

    # default: without official annotation file
    with_anns = False
    if args.official_ann is None:
        if args.dicom_prefix in OFFICIAL_ANNS:
            # train subset, use pre-defined annotation file
            official_ann = OFFICIAL_ANNS[args.dicom_prefix]
            official_ann = osp.join(data_root, official_ann)
            print(f'Using pre-defined annotation file path: {official_ann}')
            with_anns = True
        else:
            print('Having no specified annotation file, the script will '
                  'generate annotation file without ground truth.')
    else:
        official_ann = osp.join(data_root, args.official_ann)
        print(f'Using specified annotation file {official_ann}')
        with_anns = True

    # outputed ann_file
    if args.ann_file is None:
        # default: use ``annotations/{dicom_prefix}.json``
        ann_file = f'annotations/{args.dicom_prefix}.json'
    else:
        ann_file = args.ann_file
    # ann_file abspath
    ann_file = osp.join(data_root, ann_file)
    # remove existing ann_file
    if osp.exists(ann_file):
        print(f'WARNING: {ann_file} exists, it will be removed.')
        os.remove(ann_file)

    # outputed imgs
    if args.img_prefix is None:
        # default: use ``images/{dicom_prefix}/``
        img_prefix = f'images/{args.dicom_prefix}'
    else:
        img_prefix = args.img_prefix
    # img_prefix abspath
    img_prefix = osp.join(data_root, img_prefix)
    print(f'Set image prefix to {img_prefix}')
    # check the img_prefix dir; the script will convert DICOM
    # to JPEG if not exist.
    mmcv.mkdir_or_exist(img_prefix)

    # Step. 1: get all studies.
    studies = glob.glob(osp.join(dicom_prefix, 'study*'))
    print(f'There are {len(studies)} study instances in {dicom_prefix}')

    # Step. 2: get all DICOMs for each study instance
    studies_dict = defaultdict(dict)
    for i, study in enumerate(studies):
        study_id = i + 1  # study_id starts from 1
        # init Study instance
        study = Study.fromfiles(study_id, study)
        # read or convert images, update info about images
        study.read_or_convert_images(img_prefix)
        studies_dict[study.study_uid] = study

    # Step. 3: series
    series_id = 1
    series_list = []
    for study_uid, study in studies_dict.items():
        for _series in study.series_list:
            _series_anns = _series.anns(
                series_id=series_id,
                study_id=study.study_id,
                study_idx=study.study_idx,
                study_uid=study.study_uid)

            series_list.append(_series_anns)
            series_id += 1

    # Step. 4: dicoms
    dicom_id = 1
    dicoms_list = []
    for _, study in studies_dict.items():
        for dicom in study.dicoms:
            dicom['id'] = dicom_id
            dicoms_list.append(dicom)
            dicom_id += 1

    # Step. 5: set all categories
    categories = []
    for i, category in enumerate(CLASSES):
        cat = dict(id=i + 1)  # cat_id starts from 1
        cat['name'] = category
        categories.append(cat)

    # Step. 6 (optional): load official annotations
    if with_anns:
        print(f'Loading offiicial annotation file {official_ann}')
        ori_anns = mmcv.load(official_ann)

        ann_id = 1  # ann_id starts from 1
        anns_list = []
        for i, _ann in enumerate(ori_anns):
            # study info in official annotion file, e.g.,
            # seriesUid, instanceUid
            study_uid = _ann['studyUid']
            studies_dict[study_uid].update('series_uid',
                                           _ann['data'][0]['seriesUid'])
            studies_dict[study_uid].update('instance_uid',
                                           _ann['data'][0]['instanceUid'])

            study_id = studies_dict[study_uid].study_id
            points = _ann['data'][0]['annotation'][0]['data']['point']
            _anns_list, ann_id = load_official_anns(points, study_id, ann_id)
            anns_list.extend(_anns_list)

    # Step. 7: output some info
    print(f'There are {len(dicoms_list)} DICOMs in total.')
    print(f'There are {len(series_list)} series in total.')
    print(f'There are {len(categories)} categories in total. They are: ')
    for cat in categories:
        print(f'  {cat}')
    if with_anns:
        print(f'There are {len(anns_list)} converted annotations in total.')

    # Step. 8: write ann_file
    # convert studies_dict to studies_list
    studies_list = [study.anns() for _, study in studies_dict.items()]
    anns = dict(
        studies=studies_list,
        series=series_list,
        dicoms=dicoms_list,
        categories=categories)
    if with_anns:
        anns['annotations'] = anns_list
    print(f'Writing {ann_file}...')
    mmcv.dump(anns, ann_file)
    print('Done.')


if __name__ == '__main__':
    main()
