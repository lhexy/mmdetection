import numpy as np
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import DefaultFormatBundle, to_tensor
from .loading import LoadAnnotations
from .transforms import RandomFlip, Resize


@PIPELINES.register_module()
class TianchiLoadAnnotations(LoadAnnotations):

    def __init__(self,
                 with_point=True,
                 with_bbox=True,
                 bbox_size=(40, 30),
                 with_label=True,
                 with_part_inds=True):
        super().__init__(with_bbox, with_label)
        self.with_point = with_point
        self.bbox_size = bbox_size
        self.with_part_inds = with_part_inds

    def _load_bboxes(self, results):
        """Convert points to bboxes.
        """
        points = results['gt_points'].copy()
        bboxes = np.zeros((points.shape[0], 4), dtype=np.float32)

        bboxes[:, 0] = points[:, 0] - self.bbox_size[0] / 2.
        bboxes[:, 1] = points[:, 1] - self.bbox_size[1] / 2.
        bboxes[:, 2] = points[:, 0] + self.bbox_size[0] / 2.
        bboxes[:, 3] = points[:, 1] + self.bbox_size[1] / 2.

        results['gt_bboxes'] = bboxes
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_points(self, results):
        """Private function to load point annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_points'] = ann_info['points'].copy()
        results['point_fields'].append('gt_points')
        if self.with_part_inds:
            results['part_inds'] = results['ann_info']['part_inds'].copy()
        return results

    def __call__(self, results):
        if self.with_point:
            results = self._load_points(results)
        results = super().__call__(results)
        return results


@PIPELINES.register_module()
class TianchiFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        results = super().__call__(results)
        for key in ['gt_points', 'part_inds']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))

        return results


@PIPELINES.register_module()
class TianchiResize(Resize):

    def _resize_points(self, results):
        """Resize points with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('point_fields', []):
            points = results[key] * results['scale_factor'][:2]
            points[:, 0] = np.clip(points[:, 0], 0, img_shape[1])
            points[:, 1] = np.clip(points[:, 1], 0, img_shape[0])
            results[key] = points

        return results

    def __call__(self, results):
        results = super().__call__(results)
        results = self._resize_points(results)

        return results


@PIPELINES.register_module()
class TianchiRandomFlip(RandomFlip):
    """Flip the image & point.
    """

    def point_flip(self, points, img_shape, direction):
        """Flip points.
        """

        assert points.shape[-1] == 2
        flipped = points.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0] = w - points[..., 0]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1] = h - points[..., 1]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")

        return flipped

    def __call__(self, results):
        results = super().__call__(results)
        if results['flip']:
            # flip points
            for key in results.get('point_fields', []):
                results[key] = self.point_flip(results[key],
                                               results['img_shape'],
                                               results['flip_direction'])
        return results
