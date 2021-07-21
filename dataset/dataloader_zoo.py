# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch
import numpy as np

from .co3d_dataset import FrameData
from .scene_batch_sampler import SceneBatchSampler


def dataloader_zoo(
    datasets: typing.Dict[str, torch.utils.data.Dataset],
    dataset_name="co3d_singlesequence",
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_len: int = 1000,
    dataset_len_val: int = 1,
    images_per_seq_options: typing.List[int] = [2],
):
    """
    Returns a set of dataloaders for a given set of datasets.

    Args:
        datasets: A dictionary containing the
            `"dataset_subset_name": torch_dataset_object` key, value pairs.
        dataset_name: The name of the returned dataset.
        batch_size: The size of the batch of the dataloader.
        num_workers: Number data-loading threads.
        dataset_len: The number of batches in a training epoch.
        dataset_len_val: The number of batches in a validation epoch.
        images_per_seq_options: Possible numbers of images sampled per sequence.

    Returns:
        dataloaders: A dictionary containing the
            `"dataset_subset_name": torch_dataloader_object` key, value pairs.
    """

    if dataset_name not in ["co3d_singlesequence", "co3d_multisequence"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloaders = {}

    if dataset_name in ["co3d_singlesequence", "co3d_multisequence"]:
        for dataset_set, dataset in datasets.items():
            num_samples = {
                "train": dataset_len,
                "val": dataset_len_val,
                "test": None,
            }[dataset_set]

            if dataset_set == "test":
                batch_sampler = dataset.eval_batches
            else:
                num_samples = len(dataset) if num_samples <= 0 else num_samples
                batch_sampler = SceneBatchSampler(
                    dataset,
                    batch_size,
                    num_batches=num_samples,
                    images_per_seq_options=images_per_seq_options,
                )

            dataloaders[dataset_set] = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=FrameData.collate,
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataloaders
