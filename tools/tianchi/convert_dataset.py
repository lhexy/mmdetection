import argparse
import glob
import os
import os.path as osp
import SimpleITK as sitk
from collections import defaultdict

import mmcv
import numpy as np

PARTS = ('T12-L1', 'L1', 'L1-L2', 'L2', 'L2-L3', 'L3', 'L3-L4', 'L4', 'L4-L5',
         'L5', 'L5-S1')
DISC_PARTS = ('T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1')
VERTEBRA_PARTS = ('L1', 'L2', 'L3', 'L4', 'L5')

CLASSES = ('disc: v1', 'disc: v2', 'disc: v3', 'disc: v4', 'disc: v5',
           'vertebra: v1', 'vertebra: v2')

OFFICIAL_ANNS = {
    'lumbar_train150': 'annotations/lumbar_train150_annotation.json',
    'lumbar_train51': 'annotations/lumbar_train51_annotation.json',
}


class Study(object):
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

    def __init__(self, study_id, study_path):
        self._study_id = study_id
        # study_idx is the dirname of study in official dataset
        self._study_idx = osp.basename(study_path)

        # get all DICOM files in the study dir
        self._dicom_filenames = glob.glob(osp.join(study_path, '*.dcm'))

    @property
    def study_id(self):
        return self._study_id

    @property
    def study_idx(self):
        return self._study_idx

    @property
    def study_uid(self):
        assert hasattr(self, 'dicoms'), 'Please run load_all_dicoms first'
        return self.dicoms[-1]['study_uid']

    def info(self):
        _info = dict(
            id=self._study_id,
            study_idx=self._study_idx,
            study_uid=self.study_uid)

        return _info

    def _convert_to_img(self, reader, img_path):
        """Convert DICOM to JPEG image.

        Args:
            reader (sitk.Reader): DICOM file reader.
            img_path (str): Path to store converted image.

        Returns:
            img (np.ndarray): Converted image.
        """
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

    def _parse_meta(self, meta):
        _meta = {}
        for keyname, meta_key in self.META_KEYS.items():
            if meta_key not in meta:
                continue
            _meta[keyname] = meta[meta_key]

        process_items = [
            'image_position', 'image_orientation', 'slice_thickness',
            'pixel_spacing'
        ]
        for key in process_items:
            if key not in _meta:
                continue
            _meta[key] = list(map(float, _meta[key].strip().split('\\')))

        if 'image_orientation' in _meta:
            # process meta info
            rd = np.array(_meta['image_orientation'][:3], dtype=np.float32)
            cd = np.array(_meta['image_orientation'][3:], dtype=np.float32)
            normal = np.cross(rd, cd)
            normal = normal / np.linalg.norm(normal)

            thr = 0.86
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

    def load_all_dicoms(self, reader, dicom_id, img_prefix):
        """Get all DICOM files of the study instance.

        Args:
            reader (sitk.Reader): SimpleITK DICOM file reader.
            dicom_id (int): Global dicom_id, starting from 1.
            img_prefix (str): Path to converted images.

        Returns:
            dicoms_list (list[dict]): A list of each dicom avaiable meta info.
            dicom_id (int): increased dicom_id.
        """
        # imgs will be stored in ``{img_prefix}/{study_idx}``
        img_prefix = osp.join(img_prefix, self._study_idx)

        dicoms_list = []
        for dicom_filename in self._dicom_filenames:
            # read DICOM file
            reader.SetFileName(dicom_filename)
            try:
                reader.ReadImageInformation()
            except RuntimeError:
                # some DICOM files are unreadable
                continue

            _dicom_dict = dict(id=dicom_id, study_id=self._study_id)

            filename = osp.basename(dicom_filename)
            filename = filename.replace('.dcm', '.jpg')
            _dicom_dict['filename'] = filename

            meta = {
                key: reader.GetMetaData(key)
                for key in reader.GetMetaDataKeys()
            }
            _dicom_dict.update(self._parse_meta(meta))

            # image height and width
            img_path = osp.join(img_prefix, filename)
            if osp.exists(img_path):
                img = mmcv.imread(img_path)
            else:
                # no JPEG image, convert DICOM to JPEG
                img = self._convert_to_img(reader, img_path)
            height, width = img.shape[:2]
            _dicom_dict['height'] = height
            _dicom_dict['width'] = width

            dicoms_list.append(_dicom_dict)
            dicom_id += 1

        # set Study attributes
        self.dicoms = dicoms_list
        return dicoms_list, dicom_id


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
            assert 'disc' in point['tag']
            code = point['tag']['disc']
            if code == '':
                code = 'v1'  # labeled as normal if missed
            if ',' in code:
                # some disc have multi labels
                for _code in code.split(','):
                    cat_id = get_cat_id('disc', _code)
                    _ann = dict(
                        id=ann_id,
                        study_id=study_id,
                        category_id=cat_id,
                        point=point['coord'],
                        identification=part)
                    anns_list.append(_ann)
                    ann_id += 1
            else:
                cat_id = get_cat_id('disc', code)
                _ann = dict(
                    id=ann_id,
                    study_id=study_id,
                    category_id=cat_id,
                    point=point['coord'],
                    identification=part)
                anns_list.append(_ann)
                ann_id += 1
        elif part in VERTEBRA_PARTS:
            assert 'vertebra' in point['tag']
            code = point['tag']['vertebra']
            if code == '':
                code = 'v1'  # labeled as normal if missed
            cat_id = get_cat_id('vertebra', code)
            _ann = dict(
                id=ann_id,
                study_id=study_id,
                category_id=cat_id,
                point=point['coord'],
                identification=part)
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

    # Step. 2: set DICOM file reader
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetImageIO('GDCMImageIO')

    # Step. 3: get all DICOMs for each study instance
    dicom_id = 1  # starts from 1, global dicom id
    dicoms_list = []  # all dicoms will be stored in the list
    # keep a dict of studies (using study_uid as key) for
    # adding official annotations in the following
    studies_dict = defaultdict(dict)
    for i, study in enumerate(studies):
        study_id = i + 1  # study_id starts from 1
        study = Study(study_id, study)
        _dicoms_list, dicom_id = study.load_all_dicoms(reader, dicom_id,
                                                       img_prefix)
        dicoms_list.extend(_dicoms_list)
        studies_dict[study.study_uid] = study.info()

    # Step. 4: set all categories
    categories = []
    for i, category in enumerate(CLASSES):
        cat = dict(id=i + 1)  # cat_id starts from 1
        cat['tag'] = category.split(':')[0].strip()  # disc, vertebra
        cat['code'] = category.split(':')[1].strip()  # v1, v2, v3, v4, v5
        categories.append(cat)

    # Step. 5 (optional): load official annotations
    if with_anns:
        print(f'Loading offiicial annotation file {official_ann}')
        ori_anns = mmcv.load(official_ann)

        ann_id = 1  # ann_id starts from 1
        anns_list = []
        for i, _ann in enumerate(ori_anns):
            # study info in official annotion file, e.g.,
            # seriesUid, instanceUid
            study_uid = _ann['studyUid']
            studies_dict[study_uid]['series_uid'] = _ann['data'][0][
                'seriesUid']
            studies_dict[study_uid]['instance_uid'] = _ann['data'][0][
                'instanceUid']

            study_id = studies_dict[study_uid]['id']
            points = _ann['data'][0]['annotation'][0]['data']['point']
            _anns_list, ann_id = load_official_anns(points, study_id, ann_id)
            anns_list.extend(_anns_list)

    # Step. 6: output some info
    print(f'There are {len(dicoms_list)} DICOMs in total.')
    print(f'There are {len(categories)} categories in total. They are: ')
    for cat in categories:
        print(f'  {cat}')
    if with_anns:
        print(f'There are {len(anns_list)} converted annotations in total.')

    # Step. 7: write ann_file
    # convert studies_dict to studies_list
    studies_list = [study for _, study in studies_dict.items()]
    anns = dict(
        studies=studies_list, dicoms=dicoms_list, categories=categories)
    if with_anns:
        anns['annotations'] = anns_list
    print(f'Writing {ann_file}...')
    mmcv.dump(anns, ann_file)
    print('Done.')


if __name__ == '__main__':
    main()
