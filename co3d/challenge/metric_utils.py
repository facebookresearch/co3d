# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import Optional
from typing import Tuple
from .data_types import RGBDAFrame


EVAL_METRIC_NAMES = ["psnr", "psnr_fg", "depth_abs_fg", "iou", "psnr_full_image"]


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
        gt_depth_mask=target.depth_mask,
    )


def eval_one_rgbda(
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    fg_mask: np.ndarray,
    gt_image_rgb: np.ndarray,
    gt_depth_map: np.ndarray,
    gt_fg_mask: np.ndarray,
    gt_depth_mask: Optional[np.ndarray] = None,
    crop_around_fg_mask: bool = False,
):
    """
    Args:
        image_rgb: 3xHxW, black background
        depth_map: 1xHxW
        fg_mask: 1xHxW in {0, 1}
        gt_image_rgb: 3xHxW, black background
        gt_depth_map: 1xHxW
        gt_fg_mask: 1xHxW in {0, 1}
        gt_depth_mask: 1xHxW in {0, 1}

    Returns:
        eval_result: a dictionary {metric_name: str: metric_value: float}
    """

    # chuck non-finite depth
    gt_depth_map[~np.isfinite(gt_depth_map)] = 0

    if gt_depth_mask is not None:
        gt_depth_map = gt_depth_map * gt_depth_mask
    
    if crop_around_fg_mask:
        import pdb; pdb.set_trace()
        fg_mask_box_xxyy = _get_bbox_from_mask(gt_fg_mask[0])
        [
            image_rgb,
            depth_map,
            fg_mask,
            gt_image_rgb,
            gt_depth_map,
            gt_fg_mask,
            gt_depth_mask,
        ] = [
            x[
                :, 
                fg_mask_box_xxyy[2]:fg_mask_box_xxyy[3],
                fg_mask_box_xxyy[0]:fg_mask_box_xxyy[1],
            ] for x in [
                image_rgb,
                depth_map,
                fg_mask,
                gt_image_rgb,
                gt_depth_map,
                gt_fg_mask,
                gt_depth_mask,
            ]
        ]

    gt_image_rgb_masked = gt_image_rgb * gt_fg_mask
    psnr = calc_psnr(image_rgb, gt_image_rgb_masked)
    psnr_full_image = calc_psnr(image_rgb, gt_image_rgb)
    psnr_fg = calc_psnr(image_rgb, gt_image_rgb_masked, mask=gt_fg_mask)
    mse_depth, abs_depth = calc_mse_abs_depth(
        depth_map,
        gt_depth_map,
        gt_fg_mask,
        crop=5,
    )
    iou = calc_iou(fg_mask, gt_fg_mask)

    if not np.isfinite(abs_depth):
        import pdb; pdb.set_trace()
        pass

    return {
        "psnr": psnr,
        "psnr_fg": psnr_fg,
        "psnr_full_image": psnr_full_image,
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


def _get_bbox_from_mask(
    mask: np.ndarray,
    box_crop_context: float = 0.1,
    thr: float = 0.5,
    decrease_quant: float = 0.05,
) -> Tuple[int, int, int, int]:
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    assert thr > 0.0
    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))
    h, w = y1 - y0 + 1, x1 - x0 + 1
    if box_crop_context > 0.0:
        c = box_crop_context
        x0 -= w * c / 2
        y0 -= h * c / 2
        h += h * c
        w += w * c
        x1 = x0 + w
        y1 = y0 + h
    x0, x1 = [np.clip(x_, 0, mask.shape[1]) for x_ in [x0, x1]]
    y0, y1 = [np.clip(y_, 0, mask.shape[0]) for y_ in [y0, y1]]
    return np.round(np.array(x0, x1, y0, y1)).astype(int).tolist()


def _get_1d_bounds(arr: np.ndarray) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]