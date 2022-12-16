import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import cv2


class MaskRemoval(Model):
    def __init__(self, fraction_threshold=0.3):
        super(MaskRemoval, self).__init__()
        self.fraction_threshold = fraction_threshold

    def forward(self, mask_rois, cls_prob, mask_prob, cls_idx, im_shape):
        mask_logit_gpu = mask_prob
        mask_energy = mask_rois.tf.zeros(1, mask_rois.size(0), (im_shape[0], im_shape[1]))
        frame_id = 0
        context = device(mask_rois)
        mask_rois = tf.make_ndarray(mask_rois)
        cls_prob = tf.make_ndarray(cls_prob)
        mask_logit = tf.make_ndarray(mask_logit_gpu)
        cls_idx = tf.make_ndarray(cls_idx)

        mask_image = np.zeros((np.max(cls_idx),) + im_shape, dtype=np.uint8)

        sorted_inds = np.argsort(cls_prob)[::-1]
        mask_rois = mask_rois[sorted_inds]
        cls_prob[sorted_inds]
        mask_logit = mask_logit[sorted_inds]
        cls_idx = cls_idx[sorted_inds] - 1
        if len(cls_idx) == 1 and cls_idx[0] == -1:
            mask_energy = mask_logit_gpu.tf.zeros(1, 1, (im_shape[0], im_shape[1]))
            return tf.convert_to_tensor(np.array([0], dtype=np.int64)), mask_energy

        keep_inds = []
        ref_boxes = mask_rois.astype(np.int32)

        for i in range(sorted_inds.shape[0]):
            ref_box = ref_boxes[i, :].astype(np.int32)
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = max(w, 1)
            h = max(h, 1)
            logit = cv2.resize(mask_logit[i].squeeze(), (w, h))
            logit_tensor = tf.convert_to_tensor(logit)  # check here once .cuda() not written
            mask = np.array(logit > 0, dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_shape[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_shape[0])

            crop_mask = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            mask_sum = crop_mask.sum()

            mask_image_crop = mask_image[cls_idx[i]][y_0:y_1, x_0:x_1]
            if mask_sum == 0 or (
                    np.logical_and(mask_image_crop >= 1, crop_mask == 1).sum() / mask_sum > self.fraction_threshold):
                continue
            keep_inds.append(sorted_inds[i])
            mask_image[cls_idx[i]][y_0:y_1, x_0:x_1] += crop_mask
            mask_energy[0, frame_id, y_0: y_1, x_0: x_1] = logit_tensor[(y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                           (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            frame_id += 1

        mask_energy = mask_energy[:, :len(keep_inds)]
        if len(keep_inds) == 0:
            mask_energy = mask_logit_gpu.tf.zeros(1, mask_rois.size(0), (im_shape[0], im_shape[1]))
            return tf.convert_to_tensor(np.array([0], dtype=np.int64)), mask_energy

        return tf.convert_to_tensor(np.array(keep_inds)), mask_energy
