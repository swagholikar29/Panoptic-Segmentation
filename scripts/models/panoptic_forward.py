import time
import numpy as np
import tensorflow as tf
import tf.nn as nn
from models.config.config import config
from models.operators.modules.mask_roi import MaskROI
from models.operators.modules.unary_logits import MaskTerm, SegTerm
from models.operators.modules.mask_removal import MaskRemoval
from models.operators.modules.mask_matching import MaskMatching
import models.panoptic_model

class panoptic_forward(y_sem,y_inst)
    def __init__(self, backbone_depth):
            
        #param for training
        self.box_keep_fraction = config.train.panoptic_box_keep_fraction #(change as per file structure)
        self.enable_void = 0 #config.train.panoptic_box_keep_fraction < 1 #(change as per file structutre)

        self.mask_roi_panoptic = MaskROI(clip_boxes=True, bbox_class_agnostic=False, top_n=config.test.max_det, 
                                        num_classes=self.num_classes, nms_thresh=0.5, class_agnostic=True, score_thresh=config.test.panoptic_score_thresh) #finds out gt ROI regions (change as per file structure)
        self.mask_removal = MaskRemoval(fraction_threshold=0.3) #remove the masks in those regions 
        self.seg_term = SegTerm(config.dataset.num_seg_classes) #keeps only stuff segmentation from  semantic (change as per file structure)
        self.mask_term = MaskTerm(config.dataset.num_seg_classes, box_scale=1/4.0) #(change as per file structure)
        self.mask_matching = MaskMatching(config.dataset.num_seg_classes, enable_void=self.enable_void) #add instance (Z(stuff+i))=X(maski)+Y(maski) #(change as per file structure)

        # # Loss layer
        ##check what to do for nn
        self.panoptic_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=False, name='categorical_crossentropy') #pixel-wise cross entropy loss, ignore index=255
        self.initialize()

    def initialize(self):
        pass

    def forward(self, data, label=None):

        if label is not None:
            # extract gt rois for panoptic head
            gt_rois, cls_idx = self.get_gt_rois(label['roidb'], data['im_info']) 
            if self.enable_void:
                keep_inds = np.random.choice(gt_rois.shape[0], max(int(gt_rois.shape[0] * self.box_keep_fraction), 1), replace=False)
                gt_rois = gt_rois[keep_inds]
                cls_idx = cls_idx[keep_inds]
            #gt_rois, cls_idx = gt_rois.to(device(rois)), cls_idx.to(device(rois)) ####change to tensorflow 2.0
            gt_rois = tf.device(rois) ##may need to change
            cls_idx = tf.device(rois) ##may need to change

            # Calc mask logits with gt rois
            mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], gt_rois) ##do we need to extract anything from encoder-decoder ?
            mask_score = mask_score.gather(1, cls_idx.view(-1, 1, 1, 1).expand(-1, -1, config.network.mask_size, config.network.mask_size)) ##we may need to import config to our file directory

            # Calc panoptic logits
            seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_score'], gt_rois)
            mask_logits = self.mask_term(mask_score, gt_rois, cls_idx, fcn_output['fcn_score'])

            if self.enable_void:
                #void_logits = torch.max(fcn_output['fcn_score'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
                void_logits = tf.math.reduce_max(fcn_output['fcn_score'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], axis=1, keepdim=True)[0] - tf.math.reduce_max(seg_inst_logits, axis=1, keepdim=True)[0]
                inst_logits = seg_inst_logits + mask_logits
                #panoptic_logits = torch.cat([seg_logits, inst_logits, void_logits], dim=1)
                panoptic_logits = tf.concat([seg_logits, inst_logits, void_logits], axis=1)
            else:
                #panoptic_logits = torch.cat([seg_logits, (seg_inst_logits + mask_logits)], dim=1)
                panoptic_logits = tf.concat([seg_logits, (seg_inst_logits + mask_logits)], axis=1)



            # generate gt for panoptic head 
            #with torch.no_grad():
            with tf.stop_gradient():
                if self.enable_void:
                    panoptic_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], keep_inds=keep_inds)
                else:
                    panoptic_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'])

            # Panoptic head loss
            panoptic_acc = self.calc_panoptic_acc(panoptic_logits, panoptic_gt)
            panoptic_loss = self.panoptic_loss(panoptic_logits, panoptic_gt)
            panoptic_loss = panoptic_loss.mean()

            ###edit as per requirement
            
            output = {
                #'rpn_cls_loss': rpn_cls_loss.unsqueeze(0),
                #'rpn_bbox_loss': rpn_bbox_loss.unsqueeze(0),
                #'cls_loss': cls_loss.unsqueeze(0),
                #'bbox_loss': bbox_loss.unsqueeze(0),
                #'mask_loss': mask_loss.unsqueeze(0),
                #'fcn_loss': fcn_loss.unsqueeze(0),
                'panoptic_loss': panoptic_loss.unsqueeze(0),
                #'rcnn_accuracy': rcnn_acc.unsqueeze(0),
                'panoptic_accuracy': panoptic_acc.unsqueeze(0),
            }
            #if config.train.fcn_with_roi_loss:
            #   output.update({'fcn_roi_loss': fcn_roi_loss}) ##change as per file structure

            return output

        else:

            # get panoptic logits
            keep_inds, mask_logits = self.mask_removal(mask_rois[:, 1:], cls_prob, mask_score, cls_idx, fcn_output['fcn_output'].shape[2:])
            mask_rois = mask_rois[keep_inds]
            cls_idx = cls_idx[keep_inds]
            cls_prob = cls_prob[keep_inds]
            seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_output'], mask_rois * 4.0)

            results.update({
                'panoptic_cls_inds': cls_idx, 
                'panoptic_cls_probs': cls_prob
            })

            if self.enable_void:
                #void_logits = torch.max(fcn_output['fcn_output'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
                void_logits = tf.math.reduce_max(fcn_output['fcn_output'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], axis=1, keepdim=True)[0] - tf.math.reduce_max(seg_inst_logits, axis=1, keepdim=True)[0]
                inst_logits = (seg_inst_logits + mask_logits)
                #panoptic_logits = torch.cat([seg_logits, inst_logits, void_logits], dim=1)
                panoptic_logits = tf.concat([seg_logits, inst_logits, void_logits], axis=1)
                void_id = panoptic_logits.shape[1] - 1
                #panoptic_output = torch.max(panoptic_logits, dim=1)[1]
                panoptic_output = tf.concat(panoptic_logits, axis=1)[1]
                panoptic_output[panoptic_output == void_id] = 255
            else:
                #panoptic_logits = torch.cat([seg_logits, (seg_inst_logits + mask_logits)], dim=1)
                panoptic_logits = tf.concat([seg_logits, (seg_inst_logits + mask_logits)], axis=1)
                #panoptic_output = torch.max(F.softmax(panoptic_logits, dim=1), dim=1)[1]
                panoptic_output = tf.math.reduce_max(tf.nn.softmax(panoptic_logits, axis=1), axis=1)[1]

            results.update({
            'panoptic_outputs': panoptic_output,
            })
            return results
        

    def calc_panoptic_acc(self, panoptic_logits, gt):
        #_, output_cls = torch.max(panoptic_logits.data, 1, keepdim=True)
        _, output_cls = tf.math.reduce_max(panoptic_logits.data, 1, keepdim=True)
        ignore = (gt == 255).long().sum()
        correct = (output_cls.view(-1) == gt.data.view(-1)).long().sum()
        total = (gt.view(-1).shape[0]) - ignore
        assert total != 0
        panoptic_acc = correct.float() / total.float()
        return panoptic_acc










