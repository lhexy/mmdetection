import argparse
import os.path as osp
from collections import defaultdict

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to specify the instance uid of test studies.')
    parser.add_argument(
        '--data-root',
        default='/data/tianchi',
        help='Path to Tianchi lumbar dataset.')
    parser.add_argument(
        '--img-prefix',
        default='images/lumbar_testA50',
        help='Prefix of converted JPEG images.')
    parser.add_argument(
        '--ann-file',
        default='annotations/lumbar_testA50.json',
        help='Path of converted annotation file.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_root = args.data_root
    ann_file = osp.join(data_root, args.ann_file)

    ann_infos = mmcv.load(ann_file)
    studies = ann_infos['studies']
    dicoms = ann_infos['dicoms']

    # use study_uid as key
    dicoms_dict = defaultdict(list)
    for dicom in dicoms:
        dicoms_dict[dicom['study_uid']].append(dicom)

    specified_dict = {
        'study201': 'image14.jpg',
        'study202': 'image39.jpg',
        'study203': 'image12.jpg',
        'study204': 'image16.jpg',
        'study205': 'image16.jpg',
        'study206': 'image15.jpg',
        'study207': 'image12.jpg',
        'study208': 'image20.jpg',
        'study209': 'image17.jpg',
        'study211': 'image6.jpg',
        'study212': 'image30.jpg',
        'study213': 'image14.jpg',
        'study214': 'image16.jpg',
        'study215': 'image34.jpg',
        'study216': 'image26.jpg',
        'study217': 'image17.jpg',
        'study218': 'image14.jpg',
        'study219': 'image18.jpg',
        'study220': 'image15.jpg',
        'study221': 'image28.jpg',
        'study222': 'image37.jpg',
        'study223': 'image34.jpg',
        'study224': 'image34.jpg',
        'study225': 'image14.jpg',
        'study226': 'image14.jpg',
        'study227': 'image17.jpg',
        'study228': 'image14.jpg',
        'study229': 'image17.jpg',
        'study230': 'image6.jpg',
        'study231': 'image17.jpg',
        'study232': 'image23.jpg',
        'study233': 'image30.jpg',
        'study234': 'image9.jpg',
        'study235': 'image8.jpg',
        'study236': 'image17.jpg',
        'study237': 'image6.jpg',
        'study238': 'image15.jpg',
        'study239': 'image44.jpg',
        'study240': 'image16.jpg',
        'study241': 'image25.jpg',
        'study242': 'image12.jpg',
        'study243': 'image15.jpg',
        'study244': 'image6.jpg',
        'study245': 'image17.jpg',
        'study246': 'image127.jpg',
        'study247': 'image17.jpg',
        'study248': 'image22.jpg',
        'study249': 'image14.jpg',
        'study250': 'image17.jpg',
        'study297': 'image22.jpg'
    }

    new_studies_list = []
    for study in studies:
        _dicoms = dicoms_dict[study['study_uid']]
        filename = specified_dict[study['study_idx']]

        for _dicom in _dicoms:
            if _dicom['filename'] == filename:
                study['series_uid'] = _dicom['series_uid']
                study['instance_uid'] = _dicom['instance_uid']
                break
        new_studies_list.append(study)

    ann_infos['studies'] = new_studies_list
    out_file = ann_file.replace('.json', '.ext.json')
    print(f'Writing new annotations to {out_file}')
    mmcv.dump(ann_infos, out_file)


if __name__ == '__main__':
    main()
