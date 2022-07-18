# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import logging
import time
from typing import Optional
from typing import Tuple
from .data_types import RGBDAFrame


EVAL_METRIC_NAMES = ["psnr_masked", "psnr_fg", "psnr_full_image", "depth_abs_fg", "iou"]
EVAL_METRIC_MISSING_VALUE = {
    "psnr_masked": 0.0,
    "psnr_fg": 0.0,
    "psnr_full_image": 0.0,
    "depth_abs_fg": 100000.0,
    "iou": 0.0,
}


logger = logging.getLogger(__file__)


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
    gt_fg_mask_threshold: Optional[float] = 0.5,
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

    # with Timer("start"):
    for xn, x in zip(
        ("image_rgb", "fg_mask", "depth_map"),
        (image_rgb, fg_mask, depth_map),
    ):
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite element in {xn}")

    if gt_fg_mask_threshold is not None:
        # threshold the gt mask if note done before
        gt_fg_mask = (gt_fg_mask > gt_fg_mask_threshold).astype(np.float32)

    # chuck non-finite depth
    gt_depth_map[~np.isfinite(gt_depth_map)] = 0

    if gt_depth_mask is not None:
        gt_depth_map = gt_depth_map * gt_depth_mask
    
    if crop_around_fg_mask:
        raise NotImplementedError("")
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
        
    # with Timer("psnrs"):
    psnr_masked = calc_psnr(image_rgb, gt_image_rgb_masked)
    
    psnr_full_image = calc_psnr(image_rgb, gt_image_rgb)
    psnr_fg = calc_psnr(image_rgb, gt_image_rgb_masked, mask=gt_fg_mask)
    
    # with Timer("depth"):
    mse_depth, abs_depth, aux_depth = calc_mse_abs_depth(
        depth_map,
        gt_depth_map,
        gt_fg_mask,
        crop=5,
    )
    
    # with Timer("iou"):
    iou = calc_iou(fg_mask, gt_fg_mask)

    return {
        "psnr_masked": psnr_masked,
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

    scale_l1 = scale_l2 = None
    for l_norm in ["l1", "l2"]:     
        if get_best_scale:
            # mult preds by a scalar "scale_best"
            # 	s.t. we get best possible mse error
            _optimal_scale = {
                "l1": _optimal_l1_scale,
                "l2": _optimal_l2_scale,
            }[l_norm]
            scale_best = _optimal_scale(
                pred * dmask, target * dmask, best_scale_clamp_thr
            )
            pred_scaled = pred * scale_best
            if l_norm=="l1":
                scale_l1 = scale_best 
            elif l_norm=="l2":
                scale_l2 = scale_best 
            else:
                raise ValueError(l_norm)
        else:
            pred_scaled = pred

        df = target - pred_scaled
        
        if l_norm=="l1":
            abs_depth = (dmask * np.abs(df)).sum() / dmask_mass
        elif l_norm=="l2":
            mse_depth = (dmask * (df ** 2)).sum() / dmask_mass
        else:
            raise ValueError(l_norm)
    
    return mse_depth, abs_depth, {"scale_l1": scale_l1, "scale_l2": scale_l2}


def _optimal_l2_scale(pred, gt, clamp_thr):
    """
    Return the scale s that minimizes ||gt - s pred||^2.
    The inverse scale is clamped to [eps, Inf]
    """
    xy = pred * gt
    xx = pred * pred
    scale_best = xy.mean() / np.clip(xx.mean(), clamp_thr, None)
    return scale_best


def _optimal_l1_scale(pred, gt, clamp_thr):
    """
    Return the scale s that minimizes |gt - s pred|_1.
    The scale is clamped in [-max_scale, max_scale].
    This function operates along the specified axis.
    """
    max_scale = 1 / clamp_thr
    x, y = pred.reshape(-1), gt.reshape(-1)
    pivots = y / np.clip(x, 1e-10, None)
    perm = np.argsort(pivots)
    pivots = pivots[perm]
    x_sorted = x[perm]
    score = -np.abs(x).sum() + 2 * np.cumsum(np.abs(x_sorted))
    # find the index of first positive score
    i = (score <= 0).astype(np.float32).sum().astype(np.int64)
    # i = torch.unsqueeze(i, dim)
    if i >= len(pivots.reshape(-1)):
        # logger.warning("Scale outside of bounds!")
        return 1.0
    else:
        scale = pivots[i]
        scale = np.clip(scale, -max_scale, max_scale)
    # scale = torch.take_along_dim(pivots, i, dim=dim)
    # scale = torch.clip(scale, min=-max_scale, max=max_scale)
    # outshape = [s for si, s in enumerate(y.shape) if si != dim]
    # scale = scale.view(outshape)
    return float(scale)



def calc_iou(
    predict: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: Optional[float] = 0.5,
) -> np.float32:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    if threshold is not None:
        predict = (predict >= threshold).astype(np.float32)
        target = (target >= threshold).astype(np.float32)
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


class Timer:
    def __init__(self, name=None):
        self.name = name if name is not None else "timer"

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info(f"{self.name} - {time.time() - self.start:.3e} sec")