# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import torch
import math
import sys
import json
import random

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple

from co3d.utils import dbir_utils 
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.models.visualization.render_flyaround import render_flyaround
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.implicitron.tools.vis_utils import (
    get_visdom_connection,
    make_depth_image,
)
from pytorch3d.implicitron.tools.point_cloud_utils import (
    get_rgbd_point_cloud,
)


DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")


logger = logging.getLogger(__file__)
        

def main(
    output_dir: str = os.path.join(os.path.dirname(__file__), "show_co3d_dataset_files"),
    n_show_sequences_per_category: int = 2,
    visdom_env: str = "show_co3d_dataset",
    visualize_point_clouds: bool = False,
    visualize_3d_scene: bool = True,
    n_frames_show: int = 20,
):
    """
    Visualizes object point clouds from the CO3D dataset.

    Note that the code iterates over all CO3D categories and (by default) exports
    2 videos per a category subset. Hence, the whole loop will run for
    a long time (3-4 hours).
    """

    # make the script reproducible
    random.seed(30)

    # log info messages
    logging.basicConfig(level=logging.INFO)

    # make the output dir
    os.makedirs(output_dir, exist_ok=True)

    # get the category list
    if DATASET_ROOT is None:
        raise ValueError(
            "Please set the CO3DV2_DATASET_ROOT environment variable to a valid"
            " CO3Dv2 dataset root folder."
        )
    with open(os.path.join(DATASET_ROOT, "category_to_subset_name_list.json"), "r") as f:
        category_to_subset_name_list = json.load(f)
    
    # get the visdom connection
    viz = get_visdom_connection()

    # iterate over the co3d categories
    categories = sorted(list(category_to_subset_name_list.keys()))
    for category in tqdm(categories):

        subset_name_list = category_to_subset_name_list[category]

        for subset_name in subset_name_list:

            # obtain the dataset
            expand_args_fields(JsonIndexDatasetMapProviderV2)
            dataset_map = JsonIndexDatasetMapProviderV2(
                category=category,
                subset_name=subset_name,
                test_on_train=False,
                only_test_set=False,
                load_eval_batches=True,
                dataset_JsonIndexDataset_args=DictConfig(
                    {"remove_empty_masks": False, "load_point_clouds": True}
                ),
            ).get_dataset_map()

            train_dataset = dataset_map["train"]

            # select few sequences to visualize
            sequence_names = list(train_dataset.seq_annots.keys())

            # select few sequence names
            show_sequence_names = random.sample(
                sequence_names,
                k=min(n_show_sequences_per_category, len(sequence_names)),
            )
            
            for sequence_name in show_sequence_names:

                # load up a bunch of frames
                show_dataset_idx = [
                    x[2] for x in list(train_dataset.sequence_frames_in_order(sequence_name))
                ]
                random.shuffle(show_dataset_idx)
                show_dataset_idx = show_dataset_idx[:n_frames_show]
                data_to_show = [train_dataset[i] for i in show_dataset_idx]
                data_to_show_collated = data_to_show[0].collate(data_to_show)
                
                # show individual frames
                all_ims = []
                for k in ["image_rgb", "depth_map", "depth_mask", "fg_probability"]:
                    # all_ims_now = torch.stack([d[k] for d in data_to_show])
                    all_ims_now = getattr(data_to_show_collated, k)
                    if k=="depth_map":
                        all_ims_now = make_depth_image(
                            all_ims_now, torch.ones_like(all_ims_now)
                        )
                    if k in ["depth_mask", "fg_probability", "depth_map"]:
                        all_ims_now = all_ims_now.repeat(1, 3, 1, 1)
                    all_ims.append(all_ims_now.clamp(0.0, 1.0))
                all_ims = torch.cat(all_ims, dim=2)
                title = f"random_frames"
                viz.images(
                    all_ims, nrow=all_ims.shape[-1], env=visdom_env,
                    win=title, opts={"title": title},
                )
                
                if visualize_3d_scene:
                    # visualize a 3d plotly plot of the scene
                    camera_show = data_to_show_collated.camera
                    pointcloud_show = get_rgbd_point_cloud(
                        data_to_show_collated.camera,
                        data_to_show_collated.image_rgb,
                        data_to_show_collated.depth_map,
                        (data_to_show_collated.fg_probability > 0.5).float(),
                        mask_points=True,
                    )
                    viz.plotlyplot(
                        plot_scene(
                            {
                                sequence_name: {
                                    "camera":camera_show,
                                    "point_cloud": pointcloud_show
                                }
                            }
                        ),
                        env=visdom_env,
                        win="3d_scene",
                    )

                if not visualize_point_clouds:
                    continue

                for load_dataset_pointcloud in [True, False]:

                    model = PointcloudRenderingModel(
                        train_dataset,
                        sequence_name,
                        device="cuda:0",
                        load_dataset_pointcloud=load_dataset_pointcloud,
                    )

                    video_path = os.path.join(
                        output_dir,
                        category,
                        f"{subset_name}_l{load_dataset_pointcloud}",
                    )

                    os.makedirs(os.path.dirname(video_path), exist_ok=True)

                    logger.info(f"Rendering rotating video {video_path}")
                    
                    render_flyaround(
                        train_dataset,
                        sequence_name,
                        model,
                        video_path,
                        n_flyaround_poses=40,
                        fps=20,
                        trajectory_type="circular_lsq_fit",
                        max_angle=2 * math.pi,
                        trajectory_scale=1.5,
                        scene_center=(0.0, 0.0, 0.0),
                        up=(0.0, -1.0, 0.0),
                        traj_offset=1.0,
                        n_source_views=1,
                        visdom_show_preds=True,
                        visdom_environment=visdom_env,
                        visualize_preds_keys=(
                            "images_render",
                            "masks_render",
                            "depths_render",
                        ),
                    )


class PointcloudRenderingModel(torch.nn.Module):
    def __init__(
        self,
        train_dataset: JsonIndexDataset,
        sequence_name: str,
        render_size: Tuple[int, int] = [400, 400],
        device = None,
        load_dataset_pointcloud: bool = False,
    ):
        super().__init__()
        self._render_size = render_size
        self._pointcloud = dbir_utils.get_sequence_pointcloud(
            train_dataset,
            sequence_name,
            load_dataset_pointcloud=load_dataset_pointcloud,
        ).to(device)
        
    def forward(
        self,
        camera: CamerasBase,
        **kwargs,
    ):
        render = dbir_utils.render_point_cloud(
            camera[[0]],
            self._render_size,
            self._pointcloud,
            point_radius=0.01,
        )
        return {
            "images_render": render.image_render,
            "masks_render": render.mask_render,
            "depths_render": render.depth_render,
        }


if __name__=="__main__":
    main()
