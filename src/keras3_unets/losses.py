from __future__ import absolute_import

import numpy as np
from keras import backend as K, ops


def _crps_tf(y_true, y_pred, factor=0.05):
    """
    core of (pseudo) CRPS loss.

    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    """

    # mean absolute error
    mae = K.mean(ops.abs(y_pred - y_true))

    dist = ops.std(y_pred)

    return mae - factor * dist


def crps2d_tf(y_true, y_pred, factor=0.05):
    """
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:

        CRPS = mean_abs_err - factor * std

    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff

     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN)

    ** factor > 0.1 may yield negative loss values.

    Compatible with high-level Keras training methods

    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.

    """

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    batch_num = y_pred.shape.as_list()[0]

    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_tf(y_true[i, ...], y_pred[i, ...], factor=factor)

    return crps_out / batch_num


def _crps_np(y_true, y_pred, factor=0.05):
    """
    Numpy version of _crps_tf.
    """

    # mean absolute error
    mae = np.nanmean(np.abs(y_pred - y_true))
    dist = np.nanstd(y_pred)

    return mae - factor * dist


def crps2d_np(y_true, y_pred, factor=0.05):
    """
    (Experimental)
    Nunpy version of `crps2d_tf`.

    Documentation refers to `crps2d_tf`.
    """

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    batch_num = len(y_pred)

    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_np(y_true[i, ...], y_pred[i, ...], factor=factor)

    return crps_out / batch_num


# ========================= #
# Dice loss and variants


def dice_coef(y_true, y_pred, const=K.epsilon()):
    """
    Sørensen–Dice coefficient for 2-d samples.

    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    """

    # flatten 2-d tensors
    y_true_pos = ops.reshape(y_true, [-1])
    y_pred_pos = ops.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = ops.sum(y_true_pos * y_pred_pos)
    false_neg = ops.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = ops.sum((1 - y_true_pos) * y_pred_pos)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg)

    return coef_val


def dice(y_true, y_pred, const=K.epsilon()):
    """
    Sørensen–Dice Loss.

    dice(y_true, y_pred, const=K.epsilon())

    Input
    ----------
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    """
    # tf tensor casting
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    loss_val = 1 - dice_coef(y_true, y_pred, const=const)

    return loss_val


# ========================= #
# Tversky loss and variants


def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    """
    Weighted Sørensen–Dice coefficient.

    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    """

    # flatten 2-d tensors
    y_true_pos = ops.reshape(y_true, [-1])
    y_pred_pos = ops.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = ops.sum(y_true_pos * y_pred_pos)
    false_neg = ops.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = ops.sum((1 - y_true_pos) * y_pred_pos)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + const
    )

    return coef_val


def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    """
    Tversky Loss.

    tversky(y_true, y_pred, alpha=0.5, const=K.epsilon())

    ----------
    Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018.
    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks.
    arXiv preprint arXiv:1803.11078.

    Input
    ----------
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    """
    # tf tensor casting
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)

    return loss_val


def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3, const=K.epsilon()):
    """
    Focal Tversky Loss (FTL)

    focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)

    ----------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved
    attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging
    (ISBI 2019) (pp. 683-687). IEEE.

    ----------
    Input
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases
        gamma: tunable parameter within [1, 3].
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    """
    # tf tensor casting
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    # (Tversky loss)**(1/gamma)
    loss_val = ops.pow(
        (1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1 / gamma
    )

    return loss_val


# ========================= #
# MS-SSIM


def ms_ssim(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss.

    ms_ssim(y_true, y_pred, **tf_ssim_kw)

    ----------
    Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. Multiscale structural similarity for image quality assessment.
    In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

    ----------
    Input
        kwargs: keywords of `tf.image.ssim_multiscale`
                https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale

        *Issues of `tf.image.ssim_multiscale`refers to:
                https://stackoverflow.com/questions/57127626/error-in-calculation-of-inbuilt-ms-ssim-function-in-tensorflow

    """

    raise NotImplementedError("Not implemented in backend agnostic keras 3")
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    tf_ms_ssim = ops.image.ssim_multiscale(y_true, y_pred, **kwargs)

    return 1 - tf_ms_ssim


# ======================== #


def iou_box_coef(y_true, y_pred, mode="giou", dtype=K.floatx()):
    """
    Inersection over Union (IoU) and generalized IoU coefficients for bounding boxes.

    iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32)

    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019.
    Generalized intersection over union: A metric and a loss for bounding box regression.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).

    ----------
    Input
        y_true: the target bounding box.
        y_pred: the predicted bounding box.

        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.

        dtype: the data type of input tensors.
               Default is tf.float32.

    """

    zero = ops.convert_to_tensor(0.0, dtype)

    # subtrack bounding box coords
    ymin_true, xmin_true, ymax_true, xmax_true = ops.unstack(y_true, 4, axis=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = ops.unstack(y_pred, 4, axis=-1)

    # true area
    w_true = ops.maximum(zero, xmax_true - xmin_true)
    h_true = ops.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true

    # pred area
    w_pred = ops.maximum(zero, xmax_pred - xmin_pred)
    h_pred = ops.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred

    # intersections
    intersect_ymin = ops.maximum(ymin_true, ymin_pred)
    intersect_xmin = ops.maximum(xmin_true, xmin_pred)
    intersect_ymax = ops.minimum(ymax_true, ymax_pred)
    intersect_xmax = ops.minimum(xmax_true, xmax_pred)

    w_intersect = ops.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = ops.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect

    # IoU
    area_union = area_true + area_pred - area_intersect
    iou = ops.divide_no_nan(area_intersect, area_union)

    if mode == "iou":
        return iou

    else:
        # encolsed coords
        enclose_ymin = ops.minimum(ymin_true, ymin_pred)
        enclose_xmin = ops.minimum(xmin_true, xmin_pred)
        enclose_ymax = ops.maximum(ymax_true, ymax_pred)
        enclose_xmax = ops.maximum(xmax_true, xmax_pred)

        # enclosed area
        w_enclose = ops.maximum(zero, enclose_xmax - enclose_xmin)
        h_enclose = ops.maximum(zero, enclose_ymax - enclose_ymin)
        area_enclose = w_enclose * h_enclose

        # generalized IoU
        giou = iou - ops.divide_no_nan((area_enclose - area_union), area_enclose)

        return giou


def iou_box(y_true, y_pred, mode="giou", dtype=K.floatx()):
    """
    Inersection over Union (IoU) and generalized IoU losses for bounding boxes.

    iou_box(y_true, y_pred, mode='giou', dtype=tf.float32)

    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019.
    Generalized intersection over union: A metric and a loss for bounding box regression.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).

    ----------
    Input
        y_true: the target bounding box.
        y_pred: the predicted bounding box.

        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.

        dtype: the data type of input tensors.
               Default is tf.float32.

    """

    y_pred = ops.convert_to_tensor(y_pred)
    y_pred = ops.cast(y_pred, dtype)

    y_true = ops.cast(y_true, dtype)

    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)


def iou_seg(y_true, y_pred, dtype=K.floatx()):
    """
    Inersection over Union (IoU) loss for segmentation maps.

    iou_seg(y_true, y_pred, dtype=tf.float32)

    ----------
    Rahman, M.A. and Wang, Y., 2016, December. Optimizing intersection-over-union in deep neural networks for
    image segmentation. In International symposium on visual computing (pp. 234-244). Springer, Cham.

    ----------
    Input
        y_true: segmentation targets, c.f. `keras.losses.categorical_crossentropy`
        y_pred: segmentation predictions.

        dtype: the data type of input tensors.
               Default is tf.float32.

    """

    # tf tensor casting
    y_pred = ops.convert_to_tensor(y_pred)
    y_pred = ops.cast(y_pred, dtype)
    y_true = ops.cast(y_true, y_pred.dtype)

    y_pred = ops.squeeze(y_pred)
    y_true = ops.squeeze(y_true)

    y_true_pos = ops.reshape(y_true, [-1])
    y_pred_pos = ops.reshape(y_pred, [-1])

    area_intersect = ops.sum(ops.multiply(y_true_pos, y_pred_pos))

    area_true = ops.sum(y_true_pos)
    area_pred = ops.sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect

    return 1 - ops.divide_no_nan(area_intersect, area_union)


# ========================= #
# Semi-hard triplet


def triplet_1d(y_true, y_pred, N, margin=5.0):
    """
    (Experimental)
    Semi-hard triplet loss with one-dimensional vectors of anchor, positive, and negative.

    triplet_1d(y_true, y_pred, N, margin=5.0)

    Input
    ----------
        y_true: a dummy input, not used within this function. Appeared as a requirment of tf.keras.loss function format.
        y_pred: a single pass of triplet training, with `shape=(batch_num, 3*embeded_vector_size)`.
                i.e., `y_pred` is the ordered and concatenated anchor, positive, and negative embeddings.
        N: Size (dimensions) of embedded vectors
        margin: a positive number that prevents negative loss.

    """

    # anchor sample pair separations.
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N : 2 * N]
    Embd_neg = y_pred[:, 2 * N :]

    # squared distance measures
    d_pos = ops.sum(ops.square(Embd_anchor - Embd_pos), 1)
    d_neg = ops.sum(ops.square(Embd_anchor - Embd_neg), 1)
    loss_val = ops.maximum(0.0, margin + d_pos - d_neg)
    loss_val = ops.mean(loss_val)

    return loss_val
