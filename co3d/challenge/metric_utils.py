# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import Optional
from .data_types import RGBDAFrame


EVAL_METRIC_NAMES = ["psnr", "psnr_fg", "depth_abs_fg", "iou"]


def eval_one(
    pred: RGBDAFrame,
    target: RGBDAFrame,
):
    return eval_one_rgbda(
        pred.image,
        pred.depth,
        pred.mask,
        target.image,
        target.depth,
        target.mask,
    )


def eval_one_rgbda(
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    fg_mask: np.ndarray,
    gt_image_rgb: np.ndarray,
    gt_depth_map: np.ndarray,
    gt_fg_mask: np.ndarray,
):
    """
    image_rgb: 3xHxW, black background
    depth_map: 1xHxW
    fg_mask: 1xHxW in {0, 1}
    gt_image_rgb: 3xHxW, black background
    gt_depth_map: 1xHxW
    gt_fg_mask: 1xHxW in {0, 1}
    """

    gt_image_rgb_masked = gt_image_rgb * gt_fg_mask
    psnr = calc_psnr(image_rgb, gt_image_rgb_masked)
    psnr_fg = calc_psnr(image_rgb, gt_image_rgb_masked, mask=gt_fg_mask)
    mse_depth, abs_depth = calc_mse_abs_depth(
        depth_map,
        gt_depth_map,
        gt_fg_mask,
        crop=5,
    )
    iou = calc_iou(fg_mask, gt_fg_mask)

    return {
        "psnr": psnr,
        "psnr_fg": psnr_fg,
        "depth_abs_fg": abs_depth,
        "iou": iou,
    }


def calc_psnr(
    x: np.ndarray,
    y: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.float32:
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y, mask=mask)
    psnr = np.log10(np.clip(mse, 1e-10, None)) * (-10.0)
    return psnr


def calc_mse(
    x: np.ndarray,
    y: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.float32:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return np.mean((x - y) ** 2)
    else:
        mask_expand = np.broadcast_to(mask, x.shape)
        return (((x - y) ** 2) * mask_expand).sum() / np.clip(
            mask_expand.sum(), 1e-5, None
        )


def rgb_l1(
    pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.float32:
    """
    Calculates the mean absolute error between the predicted colors `pred`
    and ground truth colors `target`.
    """
    if mask is None:
        mask = np.ones_like(pred[:1])
    return (np.abs(pred - target) * mask).sum() / np.clip(mask.sum(), 1, None)


def calc_mse_abs_depth(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    crop: int,
    get_best_scale: bool = True,
    best_scale_clamp_thr: float = 1e-4,
) -> np.float32:

    # crop
    if crop > 0:
        target = target[:, crop:-crop, crop:-crop]
        pred = pred[:, crop:-crop, crop:-crop]
        mask = mask[:, crop:-crop, crop:-crop]

    target = target * mask
    dmask = (target > 0.0).astype(np.float32)
    dmask_mass = np.clip(dmask.sum(), 1e-4, None)

    if get_best_scale:
        # mult preds by a scalar "scale_best"
        # 	s.t. we get best possible mse error
        scale_best = estimate_depth_scale_factor(
            pred, target, dmask, best_scale_clamp_thr
        )
        pred = pred * scale_best

    df = target - pred
    mse_depth = (dmask * (df ** 2)).sum() / dmask_mass
    abs_depth = (dmask * np.abs(df)).sum() / dmask_mass
    return mse_depth, abs_depth


def estimate_depth_scale_factor(pred, gt, mask, clamp_thr):
    xy = pred * gt * mask
    xx = pred * pred * mask
    scale_best = xy.mean() / np.clip(xx.mean(), clamp_thr, None)
    return scale_best


def calc_iou(
    predict: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.float32:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    if mask is not None:
        predict = predict * mask
        target = target * mask
    intersect = (predict * target).sum()
    union = (predict + target - predict * target).sum() + 1e-4
    return intersect / union
