import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class TianchiBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 with_tag=True,
                 num_classes=11,
                 num_tags=7,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 num_tag_convs=0,
                 num_tag_fcs=0,
                 fc_out_channels=1024,
                 loss_tag=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super().__init__(
            num_shared_convs,
            num_shared_fcs,
            num_cls_convs,
            num_cls_fcs,
            num_reg_convs,
            num_reg_fcs,
            fc_out_channels=fc_out_channels,
            num_classes=num_classes,
            *args,
            **kwargs)
        self.with_tag = with_tag
        self.num_tags = num_tags
        self.num_tag_convs = num_tag_convs
        self.num_tag_fcs = num_tag_fcs

        self.loss_tag = build_loss(loss_tag)

        # add part specific branch
        self.tag_convs, self.tag_fcs, self.tag_last_dim = \
            self._add_conv_fc_branch(
                self.num_tag_convs, self.num_tag_fcs, self.shared_out_channels)

        if self.with_tag:
            self.fc_tag = nn.Linear(self.tag_last_dim, self.num_tags)

    def init_weights(self):
        super().init_weights()
        for m in self.tag_fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        if self.with_tag:
            nn.init.normal_(self.fc_tag.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_tag = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.tag_convs:
            x_tag = conv(x_tag)
        if x_tag.dim() > 2:
            if self.with_avg_pool:
                x_tag = self.avg_pool(x_tag)
            x_tag = x_tag.flatten(1)
        for fc in self.tag_fcs:
            x_tag = self.rele(fc(x_tag))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        tag_score = self.fc_tag(x_tag) if self.with_tag else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, tag_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_gt_tags, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        tags = pos_bboxes.new_full((num_samples, ),
                                   self.num_tags,
                                   dtype=torch.long)
        tag_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            tags[:num_pos] = pos_gt_tags
            tag_weights[:num_pos] = pos_weight

            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
            tag_weights[-num_neg:] = 1.0

        return (labels, label_weights, tags, tag_weights, bbox_targets,
                bbox_weights)

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    gt_tags,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]

        pos_gt_tags_list = []
        for i, res in enumerate(sampling_results):
            pos_gt_tags = gt_tags[i][res.pos_assigned_gt_inds]
            pos_gt_tags_list.append(pos_gt_tags)

        (labels, label_weights, tags, tag_weights, bbox_targets,
         bbox_weights) = multi_apply(
             self._get_target_single,
             pos_bboxes_list,
             neg_bboxes_list,
             pos_gt_bboxes_list,
             pos_gt_labels_list,
             pos_gt_tags_list,
             cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            tags = torch.cat(tags, 0)
            tag_weights = torch.cat(tag_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return (labels, label_weights, tags, tag_weights, bbox_targets,
                bbox_weights)

    def loss(self,
             cls_score,
             tag_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             tags,
             tag_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

        if tag_score is not None:
            avg_factor = max(torch.sum(tag_weights > 0).float().item(), 1.)
            if tag_score.numel() > 0:
                losses['loss_tag'] = self.loss_tag(
                    tag_score,
                    tags,
                    tag_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses

    def _nms(self,
             bboxes,
             cls_scores,
             tag_scores,
             score_thr,
             nms_cfg,
             max_num=-1,
             top_k=3,
             score_factors=None):

        num_classes = cls_scores.size(1) - 1
        if bboxes.shape[1] > 4:
            bboxes = bboxes.view(cls_scores.size(0), -1, 4)
        else:
            bboxes = bboxes[:, None].expand(-1, num_classes, 4)
        tag_scores = tag_scores[:, None].expand(-1, num_classes,
                                                tag_scores.size(-1))

        scores = cls_scores[:, :-1]

        # filter out bboxes with low scores
        valid_mask = scores > score_thr
        bboxes = bboxes[valid_mask]
        tag_scores = tag_scores[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        bboxes_results = []
        for i in range(num_classes):
            inds = (labels == i).nonzero().squeeze(1)
            if inds.numel() == 0:
                _bboxes = torch.zeros((0, 6),
                                      dtype=torch.float,
                                      device=scores.device)
            else:
                if inds.numel() < top_k:
                    topk_inds = inds
                else:
                    _, _inds = scores[inds].topk(top_k)
                    topk_inds = inds[_inds]

                _bboxes = bboxes[topk_inds]
                _scores = scores[topk_inds]
                _tag_scores = tag_scores[topk_inds]
                _tag_scores, _tag_labels = _tag_scores.topk(1, dim=1)
                _tag_labels = _tag_labels.type_as(_tag_scores)
                _bboxes = torch.cat(
                    [_bboxes, _scores[:, None], _tag_labels, _tag_scores], -1)

            bboxes_results.append(_bboxes.cpu().numpy())

        return bboxes_results

    def get_bboxes(self,
                   rois,
                   cls_score,
                   tag_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        assert torch.is_tensor(cls_score) and torch.is_tensor(tag_score)
        cls_score = F.softmax(cls_score, dim=1)
        tag_score = F.softmax(tag_score, dim=1)

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, cls_score, tag_score
        else:
            bboxes_results = self._nms(bboxes, cls_score, tag_score,
                                       cfg.score_thr, cfg.nms, cfg.max_per_img)
            return bboxes_results
