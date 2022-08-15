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

import pytorch3d
from co3d.utils import dbir_utils 
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2
)
from pytorch3d.implicitron.tools.config import expand_args_fields

# implicitron trainer is not a part of pytorch3d so we include it here
_pytorch3d_root = os.path.dirname(os.path.dirname(pytorch3d.__file__))
implicitron_trainer_dir = os.path.join(_pytorch3d_root, "projects", "implicitron_trainer")
sys.path.insert(0, implicitron_trainer_dir)
from visualize_reconstruction import render_sequence


DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")


logger = logging.getLogger(__file__)
        

def main(
    output_dir: str = os.path.join(os.path.dirname(__file__), "show_co3d_dataset_files"),
    n_show_sequences_per_category: int = 2,
):
    # make the output dir
    os.makedirs(output_dir, exist_ok=True)

    # get the category list
    with open(os.path.join(DATASET_ROOT, "category_to_subset_name_list.json"), "r") as f:
        category_to_subset_name_list = json.load(f)
    
    categories = sorted(list(category_to_subset_name_list.keys()))

    for category in tqdm(categories):

        subset_name_list = category_to_subset_name_list[category]

        for subset_name in [
            "fewview_test",
            "manyview_test",
            "fewview_dev",
            "manview_dev_0",
        ]:

            if subset_name not in subset_name_list:
                continue

            dataset_map = JsonIndexDatasetMapProviderV2(
                category=category,
                subset_name=subset_name,
                dataset_root=DATASET_ROOT,
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

                    render_sequence(
                        train_dataset,
                        sequence_name,
                        model,
                        video_path,
                        n_eval_cameras=40,
                        fps=20,
                        max_angle=2 * math.pi,
                        trajectory_type="circular_lsq_fit",
                        trajectory_scale=1.1,
                        scene_center=(0.0, 0.0, 0.0),
                        up=(0.0, -1.0, 0.0),
                        traj_offset=1.0,
                        n_source_views=9,
                        viz_env="show_co3d_dataset",
                        visdom_show_preds=False,
                        visdom_server="http://127.0.0.1",
                        visdom_port=8097,
                        num_workers=10,
                        seed=None,
                        video_resize=None,
                        visualize_preds_keys=("images_render",),
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
        )
        return {
            "images_render": render.image_render,
            "masks_render": render.mask_render,
            "depths_render": render.depth_render,
        }


if __name__=="__main__":
    main()
