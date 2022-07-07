# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import logging
import numpy as np
from PIL import Image
from typing import Optional

from .data_types import CO3DSequenceSet, CO3DTask, RGBDAFrame


logging.getLogger(__file__)


def store_rgbda_frame(rgbda_frame: RGBDAFrame, fl: str):
    store_mask(rgbda_frame.mask[0], fl + "_mask.png")
    store_depth(rgbda_frame.depth[0], fl + "_depth.png")
    store_image(rgbda_frame.image, fl + "_image.png")
    if rgbda_frame.depth_mask is not None:
        store_1bit_png_mask(rgbda_frame.depth_mask[0], fl + "depth_mask.png")


def load_rgbda_frame(fl: str, check_for_depth_mask: bool = False) -> RGBDAFrame:
    f = RGBDAFrame(
        mask=load_mask(os.path.realpath(fl + "_mask.png"))[None],
        depth=load_depth(os.path.realpath(fl + "_depth.png"))[None],
        image=load_image(os.path.realpath(fl + "_image.png")),
    )
    if check_for_depth_mask:
        depth_mask_path = fl + "_depth_mask.png"
        if os.path.isfile(depth_mask_path):
            f.depth_mask = load_1bit_png_mask(depth_mask_path)[None]
    return f


def store_1bit_png_mask(mask: np.ndarray, fl: str):
    """
    mask: HxW
    """
    Image.fromarray((mask*255).astype('u1'), mode='L').convert('1').save(fl, "PNG")


def load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def load_mask(fl: str):
    return np.array(Image.open(fl)).astype(np.float32) / 255.0


def store_mask(mask: np.ndarray, fl: str, mode: str = "L"):
    """
    mask: HxW
    """
    assert mask.ndim == 2
    if mode == "L":
        mpil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").convert("L")
    elif mode == "I;16":
        mpil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="I;16").convert(
            "I;16"
        )
    else:
        raise ValueError(mode)
    mpil.save(fl, "PNG")


def load_depth(fl: str):
    depth_pil = Image.open(fl)
    depth = (
        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((depth_pil.size[1], depth_pil.size[0]))
    )
    assert depth.ndim == 2
    return depth


def store_depth(depth: np.ndarray, fl: str):
    assert depth.ndim == 2
    depth_uint16 = np.frombuffer(depth.astype(np.float16), dtype=np.uint16).reshape(
        depth.shape
    )
    Image.fromarray(depth_uint16).save(fl)


def load_image(fl: str):
    return np.array(Image.open(fl)).astype(np.float32).transpose(2, 0, 1) / 255.0


def store_image(image: np.ndarray, fl: str):
    assert image.ndim == 3
    Image.fromarray((image.transpose(1, 2, 0) * 255.0).astype(np.uint8)).save(fl)


def get_category_to_subset_name_list(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
):
    json_file = os.path.join(dataset_root, "category_to_subset_name_list.json")
    with open(json_file, "r") as f:
        category_to_subset_name_list = json.load(f)

    # filter per-category subset lists by the selected task
    if task is not None:
        category_to_subset_name_list = {
            category: [
                subset_name
                for subset_name in subset_name_list
                if subset_name.startswith(task.value)
            ]
            for category, subset_name_list in category_to_subset_name_list.items()
        }

    # filter per-category subset lists by the selected sequence set
    if sequence_set is not None:
        category_to_subset_name_list = {
            category: [
                subset_name
                for subset_name in subset_name_list
                if f"_{sequence_set.value}" in subset_name
            ]
            for category, subset_name_list in category_to_subset_name_list.items()
        }

    # remove the categories with completely empty subset_name_lists
    category_to_subset_name_list = {
        c: l for c, l in category_to_subset_name_list.items() if len(l) > 0
    }

    return category_to_subset_name_list


def load_all_eval_batches(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
    remove_frame_paths: bool = False,
):

    category_to_subset_name_list = get_category_to_subset_name_list(
        dataset_root,
        task=task,
        sequence_set=sequence_set,
    )

    eval_batches = {}
    for category, subset_name_list in category_to_subset_name_list.items():
        for subset_name in subset_name_list:
            # load the subset eval batches
            eval_batches[(category, subset_name)] = _load_eval_batches_file(
                dataset_root,
                category,
                subset_name,
                remove_frame_paths=remove_frame_paths,
            )
    return eval_batches


def _load_eval_batches_file(
    dataset_root: str,
    category: str,
    subset_name: str,
    remove_frame_paths: bool = True,
):
    eval_batches_fl = os.path.join(
        dataset_root,
        category,
        "eval_batches",
        f"eval_batches_{subset_name}.json",
    )
    with open(eval_batches_fl, "r") as f:
        eval_batches = json.load(f)
    eval_batches = [
        b[0] for b in eval_batches
    ]  # take only the first (target evaluation) frame
    if remove_frame_paths:
        eval_batches = [b[:2] for b in eval_batches]
    return eval_batches
