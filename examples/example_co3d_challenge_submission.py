# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import torch
import dataclasses
from tqdm import tqdm
from pytorch3d.structures import Pointclouds
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset import _get_clamp_bbox
from pytorch3d.implicitron.models.base_model import ImplicitronRender
from pytorch3d.implicitron.dataset.visualize import get_implicitron_sequence_pointcloud
from pytorch3d.implicitron.tools.point_cloud_utils import render_point_cloud_pytorch3d
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from co3d.challenge.io import get_category_to_subset_name_list
from co3d.challenge.co3d_submission import CO3DSubmission
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet
from co3d.challenge.utils import (
    get_co3d_sequence_set_from_subset_name,
    get_co3d_task_from_subset_name,
)
from co3d.dataset.co3d_dataset_v2 import (
    CO3DV2DatasetMapProvider,
)


DATASET_ROOT = "/large_experiments/p3/replay/datasets/co3d/co3d45k_220512/export_v10/"


logger = logging.getLogger(__name__)


def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
):
    task = get_co3d_task_from_subset_name(subset_name)
    expand_args_fields(CO3DV2DatasetMapProvider)
    dataset_map = CO3DV2DatasetMapProvider(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        task_str=co3d_task_to_task_str(task),
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
    )
    return dataset_map.get_dataset_map()


def co3d_task_to_task_str(co3d_task: CO3DTask) -> str:
    return {CO3DTask.MANY_VIEW: "singlesequence", CO3DTask.FEW_VIEW: "multisequence"}[
        co3d_task
    ]  # this is the old co3d naming of the task


@torch.no_grad()
def update_dbir_submission_with_category_and_subset(
    submission: CO3DSubmission,
    dataset_root: str,
    category: str,
    subset_name: str,
    num_workers: int = 12,
    # max_n_points: int = int(1e5),
    max_n_points: int = int(1e4),
):
    logger.info(f"Evaluating category '{category}' subset '{subset_name}'")

    # Get the evaluation device.
    device = torch.device("cuda") if torch.cuda.is_available() else device("cpu")

    # Determine the sequence set and the task we are solving
    sequence_set = submission.sequence_set
    task = submission.task

    # Obtain the CO3Dv2 dataset map
    dataset_map = get_dataset_map(dataset_root, category, subset_name)

    # Many-view
    train_dataset = dataset_map["train"]
    
    # dbir.build_sequence_pointcloud(
    #     train_dataset,
    #     train_dataset[0].sequence_name,
    #     num_workers=num_workers,
    #     max_frames=50,
    # )
    
    # Obtain the colored sequence pointcloud using the depth maps and images
    # in the training set.
    sequence_pointcloud, _ = get_implicitron_sequence_pointcloud(
        train_dataset,
        train_dataset[0].sequence_name,
        mask_points=True,
        max_frames=50,
        num_workers=num_workers,
        load_dataset_point_cloud=False,
    )
    n_points = sequence_pointcloud.num_points_per_cloud().item()
    if n_points > max_n_points:
        # subsample the point cloud in case it is bigger than max_n_points
        subsample_idx = torch.randperm(n_points, device=device)[:max_n_points]
        sequence_pointcloud = Pointclouds(
            points=sequence_pointcloud.points_padded()[:, subsample_idx],
            features=sequence_pointcloud.features_padded()[:, subsample_idx],
        )

    sequence_pointcloud = sequence_pointcloud.to(device)

    # The test dataloader simply iterates over test_dataset.eval_batches
    # this is done by setting test_dataset.eval_batches as the batch sampler
    test_dataset = dataset_map["test"]
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=1,
        batch_sampler=test_dataset.eval_batches,
        num_workers=0,
        collate_fn=FrameData.collate,
    )

    # loop over eval examples
    logger.info(
        f"Rendering {len(test_dataloader)} test views for {category}/{subset_name}"
    )
    for eval_frame_data in tqdm(test_dataloader):

        # move the data to the requested device
        eval_frame_data = eval_frame_data.to(device)

        # render the sequence point cloud to each evaluation view
        data_rendered, render_mask, depth_rendered = render_point_cloud_pytorch3d(
            eval_frame_data.camera[[0]],
            sequence_pointcloud,
            render_size=eval_frame_data.image_rgb.shape[-2:],
            point_radius=0.03,
            topk=10,
            eps=1e-2,
            bin_size=0,
        )

        # cast to the implicitron render
        render = ImplicitronRender(
            depth_render=depth_rendered,
            image_render=data_rendered,
            mask_render=render_mask,
        )

        # cut the valid part of the render and resize to orig image size
        render_full_image = _paste_render_to_original_image(eval_frame_data, render)

        # get the image, mask, depth as numpy arrays for the challenge submission
        image, mask, depth = [
            getattr(render_full_image, f"{data_type}_render").cpu().numpy()[0]
            for data_type in ["image", "mask", "depth"]
        ]

        # add the results to the submission object
        submission.add_result(
            category=category,
            subset_name=subset_name,
            sequence_name=eval_frame_data.sequence_name[0],
            frame_number=int(eval_frame_data.frame_number[0]),
            image=image,
            mask=mask,
            depth=depth,
        )


def make_dbir_submission():

    dataset_root = DATASET_ROOT
    task = CO3DTask.MANY_VIEW
    sequence_set = CO3DSequenceSet.DEV
    output_folder = os.path.join(os.path.split(__file__)[0], "dbir_submission_files")

    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=output_folder,
        dataset_root=DATASET_ROOT,
    )

    category_to_subset_name_list = get_category_to_subset_name_list(
        dataset_root,
        task,
        sequence_set,
    )

    category = "toytrain"
    subset_name = "manyview_dev_1"
    update_dbir_submission_with_category_and_subset(
        submission=submission,
        dataset_root=dataset_root,
        category=category,
        subset_name=subset_name,
    )


    for category, category_subset_name_list in category_to_subset_name_list.items():
        for subset_name in category_subset_name_list:
            update_dbir_submission_with_category_and_subset(
                submission=submission,
                dataset_root=dataset_root,
                category=category,
                subset_name=subset_name,
            )
        submission.evaluate()

    submission.evaluate()

    submission.export_results(validate_results=True)


def _paste_render_to_original_image(
    frame_data: FrameData,
    render: ImplicitronRender,
) -> ImplicitronRender:
    # size of the render
    render_size = render.image_render.shape[2:]
    # bounding box of the crop in the original image
    bbox_xywh = frame_data.bbox_xywh[0]
    
    # original image size
    orig_size = frame_data.image_size_hw[0].tolist()
    # scale of the render w.r.t. the original crop
    render_scale = min(render_size[1] / bbox_xywh[3], render_size[0] / bbox_xywh[2])
    # valid part of the render
    render_bounds_wh = (bbox_xywh[2:] * render_scale).round().long()

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

    return ImplicitronRender(**render_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_dbir_submission()
