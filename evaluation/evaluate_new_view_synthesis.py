# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import warnings
import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np
from pytorch3d.renderer.cameras import CamerasBase
from tabulate import tabulate
import torch

from dataset.co3d_dataset import FrameData
from dataset.utils import is_known_frame, is_train_frame
from tools.camera_utils import (
    select_cameras,
    concatenate_cameras,
    volumetric_camera_overlaps,
)
from tools.image_utils import mask_background
from tools.metric_utils import iou, calc_psnr, rgb_l1, eval_depth

EVAL_N_SRC_VIEWS = [1, 3, 5, 7, 9]


@dataclass
class NewViewSynthesisPrediction:
    """
    Holds the tensors that describe a result of synthesizing new views.
    """

    depth_render: Optional[torch.Tensor] = None
    image_render: Optional[torch.Tensor] = None
    mask_render: Optional[torch.Tensor] = None
    camera_distance: Optional[torch.Tensor] = None


def eval_batch(
    frame_data: FrameData,
    nvs_prediction: NewViewSynthesisPrediction,
    bg_color: Union[torch.Tensor, str, float] = "black",
    mask_thr: float = 0.5,
    lpips_model=None,
    visualize: bool = False,
    source_cameras: Optional[List[CamerasBase]] = None,
):
    """
    Produce performance metrics for a single batch of new-view synthesis
    predictions.

    Given a set of known views (for which frame_data.frame_type.endswith('known')
    is True), a new-view synthesis method (NVS) is tasked to generate new views
    of the scene from the viewpoint of the target views (for which
    frame_data.frame_type.endswith('known') is False). The resulting
    synthesized new views, stored in `nvs_prediction`, are compared to the
    target ground truth in `frame_data` in terms of geometry and appearance
    resulting in a dictionary of metrics returned by the `eval_batch` function.

    Args:
        frame_data: A FrameData object containing the input to the new view
            synthesis method.
        nvs_prediction: The data describing the synthesized new views.
        bg_color: The background color of the generated new views and the
            ground truth.
        lpips_model: A pre-trained model for evaluating the LPIPS metric.
        visualize: If True, visualizes the results to Visdom.
        source_cameras: A list of all training cameras for evaluating the
            difficulty of the target views.

    Returns:
        results: A dictionary holding evaluation metrics.
    """

    # obtain copies to make sure we dont edit the original data
    nvs_prediction = copy.deepcopy(nvs_prediction)
    frame_data = copy.deepcopy(frame_data)

    # mask the ground truth depth in case frame_data contains the depth mask
    if frame_data.depth_mask is not None:
        frame_data.depth_map *= frame_data.depth_mask

    is_train = is_train_frame(frame_data.frame_type)
    if not (is_train[0] == is_train).all():
        raise ValueError("All frames in the eval batch have to be either train/test.")

    is_known = is_known_frame(
        frame_data.frame_type,
        device=frame_data.image_rgb.device,
    )

    if not ((is_known[1:] == 1).all() and (is_known[0] == 0).all()):
        raise ValueError(
            "For evaluation the first element of the batch has to be"
            + " a target view while the rest should be source views."
        )  # TODO: do we need to enforce this?

    # take only the first (target image)
    for k in [
        "mask_render",
        "image_render",
        "depth_render",
    ]:
        setattr(nvs_prediction, k, getattr(nvs_prediction, k)[:1])
    for k in [
        "depth_map",
        "image_rgb",
        "fg_probability",
        "mask_crop",
    ]:
        if not hasattr(frame_data, k) or getattr(frame_data, k) is None:
            continue
        setattr(frame_data, k, getattr(frame_data, k)[:1])

    if frame_data.depth_map.sum() <= 0:
        warnings.warn("Empty depth map in evaluation!")

    # eval all results in the resolution of the frame_data image
    image_resol = list(frame_data.image_rgb.shape[2:])

    # threshold the masks to make ground truth binary masks
    mask_fg, mask_crop = [
        (getattr(frame_data, k) >= mask_thr) for k in ("fg_probability", "mask_crop")
    ]
    image_rgb_masked = mask_background(
        frame_data.image_rgb,
        mask_fg,
        bg_color=bg_color,
    )

    # resize to the target resolution
    for k in ("mask_render", "image_render", "depth_render"):
        imode = "bilinear" if k == "image_render" else "nearest"
        val = getattr(nvs_prediction, k)
        setattr(
            nvs_prediction,
            k,
            torch.nn.functional.interpolate(val, size=image_resol, mode=imode),
        )

    # clamp predicted images
    image_render = nvs_prediction.image_render.clamp(0.0, 1.0)

    if visualize:
        from visdom import Visdom

        viz = Visdom()

    results = {}

    results["iou"] = iou(
        nvs_prediction.mask_render,
        mask_fg,
        mask=mask_crop,
    )

    for loss_fg_mask, name_postfix in zip((mask_crop, mask_fg), ("", "_fg")):

        loss_mask_now = mask_crop * loss_fg_mask

        for rgb_metric_name, rgb_metric_fun in zip(
            ("psnr", "rgb_l1"), (calc_psnr, rgb_l1)
        ):
            results[rgb_metric_name + name_postfix] = rgb_metric_fun(
                image_render,
                image_rgb_masked,
                mask=loss_mask_now,
            )

            if visualize:
                loss_val = float(results[rgb_metric_name + name_postfix])
                viz.images(
                    torch.cat(
                        (
                            image_render,
                            image_rgb_masked,
                            loss_mask_now.repeat(1, 3, 1, 1),
                        ),
                        dim=3,
                    ),
                    env="eval_debug",
                    win=rgb_metric_name + name_postfix,
                    opts={"title": f"{rgb_metric_name}{name_postfix}_{loss_val:1.2f}"},
                )

        if name_postfix == "_fg":
            # only record depth metrics for the foreground
            _, abs_ = eval_depth(
                nvs_prediction.depth_render,
                frame_data.depth_map,
                get_best_scale=True,
                mask=loss_mask_now,
                crop=5,
            )
            results["depth_abs" + name_postfix] = abs_.mean()

            if visualize:
                from tools.vis_utils import make_depth_image
                from pytorch3d.vis.plotly_vis import plot_scene
                from tools.point_cloud_utils import get_rgbd_point_cloud
                from pytorch3d.renderer import PerspectiveCameras

                loss_val = float(abs_.mean())
                viz.images(
                    torch.cat(
                        (
                            make_depth_image(
                                nvs_prediction.depth_render, loss_mask_now
                            ),
                            make_depth_image(frame_data.depth_map, loss_mask_now),
                        ),
                        dim=3,
                    ),
                    env="eval_debug",
                    win="depth_abs" + name_postfix,
                    opts={"title": f"depth_abs_{name_postfix}_{loss_val:1.2f}"},
                )
                viz.images(
                    loss_mask_now,
                    env="eval_debug",
                    win="depth_abs" + name_postfix + "_mask",
                    opts={"title": f"depth_abs_{name_postfix}_{loss_val:1.2f}_mask"},
                )
                viz.images(
                    frame_data.depth_mask[:1],
                    env="eval_debug",
                    win="depth_abs" + name_postfix + "_maskd",
                    opts={"title": f"depth_abs_{name_postfix}_{loss_val:1.2f}_maskd"},
                )

                # show the 3D plot
                viewpoint_trivial = PerspectiveCameras().to(image_render.device)
                pcl_pred = get_rgbd_point_cloud(
                    viewpoint_trivial,
                    image_render,
                    nvs_prediction.depth_render,
                    # mask_crop,
                    torch.ones_like(nvs_prediction.depth_render),
                    # loss_mask_now,
                )
                pcl_gt = get_rgbd_point_cloud(
                    viewpoint_trivial,
                    image_rgb_masked,
                    frame_data.depth_map,
                    # mask_crop,
                    torch.ones_like(frame_data.depth_map),
                    # loss_mask_now,
                )
                _pcls = {
                    pn: p
                    for pn, p in zip(("pred_depth", "gt_depth"), (pcl_pred, pcl_gt))
                    if int(p.num_points_per_cloud()) > 0
                }
                plotlyplot = plot_scene(
                    {f"pcl{name_postfix}": _pcls},
                    camera_scale=1.0,
                    pointcloud_max_points=10000,
                    pointcloud_marker_size=1.0,
                )
                viz.plotlyplot(
                    plotlyplot,
                    env="eval_debug",
                    win=f"pcl{name_postfix}",
                )
                import pdb

                pdb.set_trace()

    if lpips_model is not None:
        im1, im2 = [
            2.0 * im.cuda().clamp(0.0, 1.0) - 1.0
            for im in (image_rgb_masked, nvs_prediction.image_render)
        ]
        results["lpips"] = lpips_model.forward(im1, im2).item()

    # convert all metrics to floats
    results = {k: float(v) for k, v in results.items()}

    if source_cameras is None:
        source_cameras = select_cameras(frame_data.camera, torch.where(is_known)[0])

    results["meta"] = {
        # calculate the camera difficulties and add to results
        "camera_difficulty": float(
            calculate_camera_difficulties(
                select_cameras(frame_data.camera, [0]),
                source_cameras,
            )
        ),
        # store the size of the batch (corresponds to n_src_views+1)
        "batch_size": int(is_known.numel()),
        # store the type of the target frame
        "frame_type": str(frame_data.frame_type[0]),
    }

    return results


def average_per_batch_results(
    results_per_batch: List[dict],
    idx: Optional[List[int]] = None,
) -> dict:
    """
    Average a list of per-batch metrics `results_per_batch`.
    Optionally, if `idx` is given, only a subset of the per-batch
    metrics, indexed by `idx`, is averaged.
    """
    result_keys = list(results_per_batch[0].keys())
    result_keys.remove("meta")
    if idx is not None:
        results_per_batch = [results_per_batch[i] for i in idx]
    if len(results_per_batch) == 0:
        return {k: float("NaN") for k in result_keys}
    return {
        k: float(np.array([r[k] for r in results_per_batch]).mean())
        for k in result_keys
    }


def calculate_camera_difficulties(
    cameras_target: CamerasBase,
    cameras_source: CamerasBase,
) -> torch.Tensor:
    """
    Calculate the difficulties of the target cameras, given a set of known
    cameras `cameras_source`.
    """
    ious = [
        volumetric_camera_overlaps(
            concatenate_cameras(
                [
                    select_cameras(cameras_target, [cami]),
                    cameras_source.to(cameras_target.device),
                ]
            )
        )[0, :]
        for cami in range(cameras_target.R.shape[0])
    ]
    camera_difficulties = torch.stack(
        [_reduce_camera_iou_overlap(iou[1:]) for iou in ious]
    )
    return camera_difficulties


def _reduce_camera_iou_overlap(ious: torch.Tensor, topk: int = 2):
    """
    Calculate the final camera difficulty by computing the average of the
    ious of the two most similar cameras.
    """
    return ious.topk(k=min(topk, len(ious) - 1)).values.mean()


def get_camera_difficulty_bin_edges(task: str):
    """
    Get the edges of camera difficulty bins.
    """
    _eps = 1e-5
    if task == "multisequence":
        # TODO: extract those to constants
        diff_bin_edges = torch.linspace(0.5, 1.0 + _eps, 4)
        diff_bin_edges[0] = 0.0 - _eps
    elif task == "singlesequence":
        diff_bin_edges = torch.tensor([0.0 - _eps, 0.97, 0.98, 1.0 + _eps]).float()
    else:
        raise ValueError(f"No such eval task {task}.")
    diff_bin_names = ["hard", "medium", "easy"]
    return diff_bin_edges, diff_bin_names


def summarize_nvs_eval_results(
    per_batch_eval_results: List[NewViewSynthesisPrediction],
    task: str = "singlesequence",
):
    """
    Compile the per-batch evaluation results `per_batch_eval_results` into
    a set of aggregate metrics. The produced metrics depend on the task.

    Args:
        per_batch_eval_results: Metrics of each per-batch evaluation.
        task: The type of the new-view synthesis task.
            Either 'singlesequence' or 'multisequence'.

    Returns:
        nvs_results_flat: A flattened dict of all aggregate metrics.
        aux_out: A dictionary holding a set of auxiliary results.
    """
    n_batches = len(per_batch_eval_results)
    if task == "singlesequence":
        eval_sets = (None,)
        # assert n_batches==100
    elif task == "multisequence":
        eval_sets = ("train", "test")
        # assert n_batches==1000
    else:
        raise ValueError(task)
    batch_sizes = torch.tensor(
        [r["meta"]["batch_size"] for r in per_batch_eval_results]
    ).long()
    camera_difficulty = torch.tensor(
        [r["meta"]["camera_difficulty"] for r in per_batch_eval_results]
    ).float()
    is_train = is_train_frame([r["meta"]["frame_type"] for r in per_batch_eval_results])

    # init the result database dict
    results = []

    diff_bin_edges, diff_bin_names = get_camera_difficulty_bin_edges(task)
    n_diff_edges = diff_bin_edges.numel()

    # add per set averages
    for SET in eval_sets:
        if SET is None:
            # task=='singlesequence'
            ok_set = torch.ones(n_batches, dtype=torch.bool)
            set_name = "test"
        else:
            # task=='multisequence'
            ok_set = is_train == int(SET == "train")
            set_name = SET

        # eval each difficulty bin, including a full average result (diff_bin=None)
        for diff_bin in [None, *list(range(n_diff_edges - 1))]:
            if diff_bin is None:
                # average over all results
                in_bin = ok_set
                diff_bin_name = "all"
            else:
                b1, b2 = diff_bin_edges[diff_bin : (diff_bin + 2)]
                in_bin = ok_set & (camera_difficulty > b1) & (camera_difficulty <= b2)
                diff_bin_name = diff_bin_names[diff_bin]
            bin_results = average_per_batch_results(
                per_batch_eval_results, idx=torch.where(in_bin)[0]
            )
            results.append(
                {
                    "subset": set_name,
                    "subsubset": f"diff={diff_bin_name}",
                    "metrics": bin_results,
                }
            )

        if task == "multisequence":
            # split based on n_src_views
            n_src_views = batch_sizes - 1
            for n_src in EVAL_N_SRC_VIEWS:
                ok_src = ok_set & (n_src_views == n_src)
                n_src_results = average_per_batch_results(
                    per_batch_eval_results,
                    idx=torch.where(ok_src)[0],
                )
                results.append(
                    {
                        "subset": set_name,
                        "subsubset": f"n_src={int(n_src)}",
                        "metrics": n_src_results,
                    }
                )

    aux_out = {"results": results}
    return flatten_nvs_results(results), aux_out


def _get_flat_nvs_metric_key(result, metric_name):
    metric_key_postfix = f"|subset={result['subset']}|{result['subsubset']}"
    metric_key = f"{metric_name}{metric_key_postfix}"
    return metric_key


def flatten_nvs_results(results):
    """
    Takes input `results` list of dicts of the form:
    ```
        [
            {
                'subset':'train/test/...',
                'subsubset': 'src=1/src=2/...',
                'metrics': nvs_eval_metrics}
            },
            ...
        ]
    ```
    And converts to a flat dict as follows:
    {
        'subset=train/test/...|subsubset=src=1/src=2/...': nvs_eval_metrics,
        ...
    }
    """
    results_flat = {}
    for result in results:
        for metric_name, metric_val in result["metrics"].items():
            metric_key = _get_flat_nvs_metric_key(result, metric_name)
            assert metric_key not in results_flat
            results_flat[metric_key] = metric_val
    return results_flat


def pretty_print_nvs_metrics(results):
    subsets, subsubsets = [
        _ordered_set([r[k] for r in results]) for k in ("subset", "subsubset")
    ]
    metrics = _ordered_set([metric for r in results for metric in r["metrics"]])

    for subset in subsets:
        tab = {}
        for metric in metrics:
            tab[metric] = []
            header = ["metric"]
            for subsubset in subsubsets:
                metric_vals = [
                    r["metrics"][metric]
                    for r in results
                    if r["subsubset"] == subsubset and r["subset"] == subset
                ]
                if len(metric_vals) > 0:
                    tab[metric].extend(metric_vals)
                    header.extend(subsubsets)

        if any(len(v) > 0 for v in tab.values()):
            print(f"===== NVS results; subset={subset} =====")
            print(
                tabulate(
                    [[metric, *v] for metric, v in tab.items()],
                    headers=header,
                )
            )


def _ordered_set(list_):
    return list(OrderedDict((i, 0) for i in list_).keys())


def aggregate_nvs_results(task_results):
    """
    Aggregate nvs results.
    For singlescene, this averages over all categories and scenes,
    for multiscene, the average is over all per-category results.
    """
    task_results_cat = [r_ for r in task_results for r_ in r]
    subsets, subsubsets = [
        _ordered_set([r[k] for r in task_results_cat]) for k in ("subset", "subsubset")
    ]
    metrics = _ordered_set(
        [metric for r in task_results_cat for metric in r["metrics"]]
    )
    average_results = []
    for subset in subsets:
        for subsubset in subsubsets:
            metrics_lists = [
                r["metrics"]
                for r in task_results_cat
                if r["subsubset"] == subsubset and r["subset"] == subset
            ]
            avg_metrics = {}
            for metric in metrics:
                avg_metrics[metric] = float(
                    np.nanmean(np.array([l[metric] for l in metrics_lists]))
                )
            average_results.append(
                {
                    "subset": subset,
                    "subsubset": subsubset,
                    "metrics": avg_metrics,
                }
            )
    return average_results
