# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import torch
from typing import Tuple
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.structures import Pointclouds
from pytorch3d.implicitron.dataset.json_index_dataset import _get_clamp_bbox
from pytorch3d.implicitron.models.base_model import ImplicitronRender
from pytorch3d.implicitron.dataset.visualize import get_implicitron_sequence_pointcloud
from pytorch3d.implicitron.tools.point_cloud_utils import (
    render_point_cloud_pytorch3d,
    get_rgbd_point_cloud,
)


def render_point_cloud(
    camera: CamerasBase,
    render_size: Tuple[int, int],
    pointcloud: Pointclouds,
    point_radius: float = 0.03,
) -> ImplicitronRender:
    """
    Render the point cloud `pointcloud` to the camera `camera` using the 
    PyTorch3D point cloud renderer.

    Args:
        camera: Rendering camera.
        render_size: 2-tuple of integers denoting the render size (HxW)
        pointcloud: The point cloud to render.
        point_radius: Radius of the rendered points.
    """
    # render the sequence point cloud to each evaluation view
    data_rendered, render_mask, depth_rendered = render_point_cloud_pytorch3d(
        camera,
        pointcloud,
        render_size=render_size,
        point_radius=point_radius,
        topk=10,
        eps=1e-2,
        bin_size=0,
    )

    # cast to the implicitron render
    return ImplicitronRender(
        depth_render=depth_rendered,
        image_render=data_rendered,
        mask_render=render_mask,
    )


def paste_render_to_original_image(
    frame_data: FrameData,
    render: ImplicitronRender,
) -> ImplicitronRender:
    """
    Paste a rendering result `render` into the original image coordinate frame.

    Args:
        frame_data: The `FrameData` object as returned by the `JsonIndexDataset`.
        render: A render to be pasted into the original image coordinates.
    """
    # size of the render
    render_size = render.image_render.shape[2:]
    
    # estimate render scale w.r.t. the frame_data images
    render_scale_factors = [
        sr / s for sr, s in zip(render_size, frame_data.image_rgb.shape[2:])
    ]
    assert abs(render_scale_factors[0]-render_scale_factors[1]) <= 1e-2, (
        "non-isotropic render rescale"
    )

    # original image size
    orig_size = frame_data.image_size_hw[0].tolist()

    # bounding box of the crop in the original image
    if frame_data.crop_bbox_xywh is not None:
        bbox_xywh = frame_data.crop_bbox_xywh[0]
    else:
        bbox_xywh = torch.LongTensor([0, 0, orig_size[1], orig_size[0]])
    
    # get the valid part of the render
    render_bounds_wh = [None, None]
    for axis in [0, 1]:
        # resize the mask crop to the size of the render
        if render_size != frame_data.mask_crop.shape[2:]:
            mask_crop_render_size = torch.nn.functional.interpolate(
                frame_data.mask_crop, size=render_size, mode="nearest"
            )
        else:
            mask_crop_render_size = frame_data.mask_crop
        # get the bounds of the mask_crop along dimemsion = 1-axis
        valid_dim_pix = mask_crop_render_size[0, 0].sum(dim=axis).reshape(-1).nonzero()
        assert valid_dim_pix.min()==0
        render_bounds_wh[axis] = valid_dim_pix.max().item() + 1
    
    render_out = {}
    for render_type, render_val in dataclasses.asdict(render).items():
        if render_val is None:
            continue
        # get the valid part of the render
        render_valid_ = render_val[..., :render_bounds_wh[1], :render_bounds_wh[0]]

        # resize the valid part to the original size
        render_resize_ = torch.nn.functional.interpolate(
            render_valid_,
            size=tuple(reversed(bbox_xywh[2:].tolist())),
            mode="bilinear" if render_type=="image_render" else "nearest",
            align_corners=False if render_type=="image_render" else None,
        )

        # paste the original-sized crop to the original image
        render_pasted_ = render_resize_.new_zeros(1, render_resize_.shape[1], *orig_size)
        render_pasted_[
            ...,
            bbox_xywh[1]:(bbox_xywh[1]+render_resize_.shape[2]),
            bbox_xywh[0]:(bbox_xywh[0]+render_resize_.shape[3]),
        ] = render_resize_
        render_out[render_type] = render_pasted_
        
    # if True:
    #     # debug visualize
    #     from visdom import Visdom
    #     viz = Visdom()
    #     visdom_env = "debug_paste_render_to_original_image"
    #     viz.image(
    #         render.image_render[0],
    #         env=visdom_env,
    #         win="original",
    #     )
    #     viz.image(
    #         render_out["image_render"][0],
    #         env=visdom_env,
    #         win="pasted",
    #     )
    #     import pdb; pdb.set_trace()
    #     pass

    return ImplicitronRender(**render_out)


def get_sequence_pointcloud(
    dataset: JsonIndexDataset,
    sequence_name: str,
    num_workers: int = 12,
    max_loaded_frames: int = 50,
    max_n_points: int = int(1e5),
    seed: int = 42,
    load_dataset_pointcloud: bool = False,
) -> Pointclouds:
    """
    Given a `dataset` object and the name of a sequence in it (`sequence_name`),
    generate a 3D pointcloud containing the main foreground object of the scene.

    Args:
        dataset: A dataset of containing sequence annotations.
        sequence_name: The name of the sequence to reconstruct.
        num_workers: Number of cores to use for loading the sequence data.
        max_n_points: Maximum number of points to keep in the point cloud.
        seed: Random seed for reproducibility.
        load_dataset_pointcloud: If `True` uses the CO3D ground truth dataset
            point cloud, otherwise generates the point cloud by unprojecting
            the depth maps of known frames.
    """
    with torch.random.fork_rng():  # fork rng for reproducibility
        torch.manual_seed(seed)
        sequence_pointcloud, _ = get_implicitron_sequence_pointcloud(
            dataset,
            sequence_name,
            mask_points=True,
            max_frames=max_loaded_frames,
            num_workers=num_workers,
            load_dataset_point_cloud=load_dataset_pointcloud,
        )
        sequence_pointcloud = _subsample_pointcloud(sequence_pointcloud, max_n_points)
    return sequence_pointcloud


def get_eval_frame_data_pointcloud(
    eval_frame_data: FrameData,
    max_n_points: int = int(3e4),
):
    """
    Generate a pointcloud by unprojecting the known depth maps of a `FrameData` object
    `eval_frame_data`. 

    Args:
        eval_frame_data: `FrameData` to unproject.
        max_n_points: Maximum number of points to keep in the point cloud.
    """
    batch_size = eval_frame_data.image_rgb.shape[0]
    pointcloud = get_rgbd_point_cloud(
        eval_frame_data.camera[list(range(1, batch_size))],
        eval_frame_data.image_rgb[1:],
        eval_frame_data.depth_map[1:],
        (eval_frame_data.fg_probability[1:] > 0.5).float(),
        mask_points=True,
    )
    return _subsample_pointcloud(pointcloud, max_n_points)


def _subsample_pointcloud(p: Pointclouds, n: int):
    n_points = p.num_points_per_cloud().item()
    if n_points > n:
        # subsample the point cloud in case it is bigger than max_n_points
        subsample_idx = torch.randperm(
            n_points,
            device=p.points_padded().device,
        )[:n]
        p = Pointclouds(
            points=p.points_padded()[:, subsample_idx],
            features=p.features_padded()[:, subsample_idx],
        )
    return p
