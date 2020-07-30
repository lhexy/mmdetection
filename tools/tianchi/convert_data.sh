#!/usr/bin/env bash

DATAROOT=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/convert_dataset.py --data-root $DATAROOT \
    --dicom-prefix lumbar_train150

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/convert_dataset.py --data-root $DATAROOT \
    --dicom-prefix lumbar_train51

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/convert_dataset.py --data-root $DATAROOT \
    --dicom-prefix lumbar_testA50
