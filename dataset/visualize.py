# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import pytorch3d
from pytorch3d.vis import plotly_vis

from typing import Optional, Tuple
from visdom import Visdom
from .co3d_dataset import Co3dDataset, FrameData

from pytorch3d.structures import Pointclouds
from tools.point_cloud_utils import get_rgbd_point_cloud


def get_co3d_sequence_pointcloud(
    dataset: Co3dDataset,
    sequence_name: str = Optional[None],
    mask_points: bool = True,
    max_frames: int = -1,
    num_workers: int = 0,
    load_dataset_point_cloud: bool = False,
) -> Tuple[Pointclouds, FrameData]:
    """
    Make a point cloud by sampling random points from each frame the dataset.
    """

    if len(dataset) == 0:
        raise ValueError("The dataset is empty.")

    if not dataset.load_depths:
        raise ValueError(
            "The Co3D dataset has to load depths (dataset.load_depths=True)."
        )

    if mask_points and not dataset.load_masks:
        raise ValueError(
            "For mask_points=True, the Co3D dataset has to load masks"
            + " (dataset.load_masks=True)."
        )

    # setup the indices of frames loaded from the dataset db
    sequence_entries = list(range(len(dataset)))
    if sequence_name is not None:
        sequence_entries = [
            ei
            for ei in sequence_entries
            if dataset.frame_annots[ei]["frame_annotation"].sequence_name
            == sequence_name
        ]
        if len(sequence_entries) == 0:
            raise ValueError(
                f'There are no dataset entries for sequence name "{sequence_name}".'
            )

    # subsample loaded frames if needed
    if (max_frames > 0) and (len(sequence_entries) > max_frames):
        sequence_entries = [
            sequence_entries[i]
            for i in torch.randperm(len(sequence_entries))[:max_frames].sort().values
        ]

    # take only the part of the dataset corresponding to the sequence entries
    sequence_dataset = torch.utils.data.Subset(dataset, sequence_entries)

    # load the required part of the dataset
    loader = torch.utils.data.DataLoader(
        sequence_dataset,
        batch_size=len(sequence_dataset),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=FrameData.collate,
    )

    for frame_data in loader:  # there's only one batch
        pass

    # scene point cloud
    if load_dataset_point_cloud:
        if not dataset.load_point_clouds:
            raise ValueError(
                "For load_dataset_point_cloud=True, the Co3D dataset has to"
                + " load point clouds (dataset.load_point_clouds=True)."
            )
        point_cloud = frame_data.sequence_point_cloud

    else:
        point_cloud = get_rgbd_point_cloud(
            frame_data.camera,
            frame_data.image_rgb,
            frame_data.depth_map,
            (frame_data.fg_probability > 0.5).float(),
            mask_points=mask_points,
        )

    return point_cloud, frame_data
