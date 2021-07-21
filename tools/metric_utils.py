# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch.nn import functional as F
from typing import Optional, Tuple


def eval_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    crop: int = 1,
    mask: Optional[torch.Tensor] = None,
    get_best_scale: bool = True,
    mask_thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the depth error between the prediction `pred` and the ground
    truth `gt`.

    Args:
        pred: A tensor of shape (N, 1, H, W) denoting the predicted depth maps.
        gt: A tensor of shape (N, 1, H, W) denoting the ground truth depth maps.
        crop: The number of pixels to crop from the border.
        mask: A mask denoting the valid regions of the gt depth.
        get_best_scale: If `True`, estimates a scaling factor of the predicted depth
            that yields the best mean squared error between `pred` and `gt`.
            This is typically enabled for cases where predicted reconstructions
            are inherently defined up to an arbitrary scaling factor.
        mask_thr: A constant used to threshold the `mask` to specify the valid
            regions.

    Returns:
        mse_depth: Mean squared error between `pred` and `gt`.
        abs_depth: Mean absolute difference between `pred` and `gt`.
    """

    # chuck out the border
    if crop > 0:
        gt = gt[:, :, crop:-crop, crop:-crop]
        pred = pred[:, :, crop:-crop, crop:-crop]

    if mask is not None:
        # mult gt by mask
        if crop > 0:
            mask = mask[:, :, crop:-crop, crop:-crop]
        gt = gt * (mask > mask_thr).float()

    dmask = (gt > 0.0).float()
    dmask_mass = torch.clamp(dmask.sum((1, 2, 3)), 1e-4)

    if get_best_scale:
        # mult preds by a scalar "scale_best"
        # 	s.t. we get best possible mse error
        xy = pred * gt * dmask
        xx = pred * pred * dmask
        scale_best = xy.mean((1, 2, 3)) / torch.clamp(xx.mean((1, 2, 3)), 1e-4)
        pred = pred * scale_best[:, None, None, None]

    df = gt - pred

    mse_depth = (dmask * (df ** 2)).sum((1, 2, 3)) / dmask_mass
    abs_depth = (dmask * df.abs()).sum((1, 2, 3)) / dmask_mass

    return mse_depth, abs_depth


def calc_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y, mask=mask)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return (((x - y) ** 2) * mask).sum() / mask.expand_as(x).sum().clamp(1e-5)


def calc_bce(
    pred: torch.Tensor,
    gt: torch.Tensor,
    equal_w: bool = True,
    pred_eps: float = 0.01,
) -> torch.Tensor:
    """
    Calculates the binary cross entropy.
    """
    if pred_eps > 0.0:
        # up/low bound the predictions
        pred = torch.clamp(pred, pred_eps, 1.0 - pred_eps)

    if equal_w:
        mask = (gt > 0.5).float()
        weight = mask / mask.sum().clamp(1.0) + (1 - mask) / (1 - mask).sum().clamp(1.0)
        # weight sum should be at this point ~2
        weight = weight * (weight.numel() / weight.sum().clamp(1.0))
    else:
        weight = torch.ones_like(gt)

    return F.binary_cross_entropy(pred, gt, reduction="mean", weight=weight)


def rgb_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculates the mean absolute error between the predicted colors `pred`
    and ground truth colors `target`.
    """
    if mask is None:
        mask = torch.ones_like(pred[:, :1])
    return ((pred - target).abs() * mask).sum(dim=(1, 2, 3)) / mask.sum(
        dim=(1, 2, 3)
    ).clamp(1)


def huber(dfsq: torch.Tensor, scaling: float = 0.03) -> torch.Tensor:
    """
    Calculates the huber function of the input squared error `dfsq`.
    The function smoothly transitions from a region with unit gradient
    to a hyperbolic function at `dfsq=scaling`.
    """
    loss = (safe_sqrt(1 + dfsq / (scaling * scaling), eps=1e-4) - 1) * scaling
    return loss


def neg_iou_loss(
    predict: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    return 1.0 - iou(predict, target, mask=mask)


def safe_sqrt(A: torch.Tensor, eps: float = float(1e-4)) -> torch.Tensor:
    """
    performs safe differentiable sqrt
    """
    return (torch.clamp(A, float(0)) + eps).sqrt()


def iou(
    predict: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    dims = tuple(range(predict.dim())[1:])
    if mask is not None:
        predict = predict * mask
        target = target * mask
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-4
    return (intersect / union).sum() / intersect.numel()


def beta_prior(pred: torch.Tensor, cap: float = 0.1) -> torch.Tensor:
    if cap <= 0.0:
        raise ValueError("capping should be positive to avoid unbound loss")

    min_value = math.log(cap) + math.log(cap + 1.0)
    return (torch.log(pred + cap) + torch.log(1.0 - pred + cap)).mean() - min_value
