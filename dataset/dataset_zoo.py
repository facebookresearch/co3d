# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import os
import json
import typing

import torch

from .co3d_dataset import Co3dDataset
from .utils import (
    DATASET_TYPE_TRAIN,
    DATASET_TYPE_TEST,
    DATASET_TYPE_KNOWN,
    DATASET_TYPE_UNKNOWN,
)


# TODO from dataset.dataset_configs import DATASET_CONFIGS
DATASET_CONFIGS = {
    "default": dict(
        box_crop=True,
        box_crop_context=0.3,
        image_width=800,
        image_height=800,
        remove_empty_masks=True,
    ),
}

DATASET_ROOT = "specify_DATASET_ROOT_FOLDER_here"

# fmt: off
CO3D_CATEGORIES = list(reversed([
    "baseballbat", "banana",  "bicycle", "microwave", "tv", 
    "cellphone", "toilet", "hairdryer", "couch", "kite", "pizza", 
    "umbrella", "wineglass", "laptop",
    "hotdog", "stopsign", "frisbee", "baseballglove", 
    "cup", "parkingmeter", "backpack", "toyplane", "toybus", 
    "handbag", "chair", "keyboard", "car", "motorcycle", 
    "carrot", "bottle", "sandwich", "remote", "bowl", "skateboard",
    "toaster", "mouse", "toytrain", "book",  "toytruck",
    "orange", "broccoli", "plant", "teddybear", 
    "suitcase", "bench", "ball", "cake", 
    "vase", "hydrant", "apple", "donut", 
]))
# fmt: on


def dataset_zoo(
    dataset_name: str = "co3d_singlesequence",
    dataset_root: str = DATASET_ROOT,
    category: typing.Optional[str] = None,
    limit_to: int = -1,
    limit_sequences_to: int = -1,
    n_frames_per_sequence: int = -1,
    test_on_train: bool = False,
    load_point_clouds: bool = False,
    mask_images: bool = False,
    mask_depths: bool = False,
    restrict_sequence_name: typing.List[str] = [],
    test_restrict_sequence_id: int = -1,
    assert_single_seq: bool = True,
) -> typing.Dict[str, torch.utils.data.Dataset]:
    """
    Generates the training / validation and testing dataset objects.

    Args:
        dataset_name: The name of the returned dataset.
        dataset_root: The root folder of the dataset.
        category: The object category of the dataset.
        limit_to: Limit the dataset to the first #limit_to frames.
        limit_sequences_to: Limit the dataset to the first
            #limit_sequences_to sequences.
        n_frames_per_sequence: Randomly sample #n_frames_per_sequence frames
            in each sequence.
        test_on_train: Construct validation and test datasets from
            the training subset.
        load_point_clouds: Enable returning scene point clouds from the dataset.
        mask_images: Mask the loaded images with segmentation masks.
        mask_depths: Mask the loaded depths with segmentation masks.
        restrict_sequence_name: Restrict the dataset sequences to the ones
            present in the given list of names.
        test_restrict_sequence_id: The ID of the loaded sequence.
            Active for dataset_name='co3d_singlesequence'.
        assert_single_seq: Assert that only frames from a single sequence
            are present in all generated datasets.

    Returns:
        datasets: A dictionary containing the
            `"dataset_subset_name": torch_dataset_object` key, value pairs.
    """

    datasets = {}

    # TODO:
    # - implement loading multiple categories

    if dataset_name in ["co3d_singlesequence", "co3d_multisequence"]:
        # This maps the common names of the dataset subsets ("train"/"val"/"test")
        # to the names of the subsets in the CO3D dataset.
        set_names_mapping = _get_co3d_set_names_mapping(dataset_name, test_on_train)

        # load the evaluation batches
        task = dataset_name.split("_")[-1]
        batch_indices_path = os.path.join(
            dataset_root,
            category,
            f"eval_batches_{task}.json",
        )
        if not os.path.isfile(batch_indices_path) and dataset_root == DATASET_ROOT:
            # The batch indices file does not exist and dataset_root is at its default
            # value, i.e. most probably the user has not specified the root folder.
            raise ValueError(
                "Please specify a correct dataset_root folder as a value of the"
                " DATASET_ROOT variable in dataset_zoo.py."
            )

        with open(batch_indices_path, "r") as f:
            eval_batch_index = json.load(f)

        if task == "singlesequence":
            assert (
                test_restrict_sequence_id is not None and test_restrict_sequence_id >= 0
            ), (
                "Please specify an integer id 'test_restrict_sequence_id'"
                + " of the sequence considered for 'singlesequence'"
                + " training and evaluation."
            )
            assert len(restrict_sequence_name) == 0, (
                "For the 'singlesequence' task, the restrict_sequence_name has"
                " to be unset while test_restrict_sequence_id has to be set to an"
                " integer defining the order of the evaluation sequence."
            )
            # a sort-stable set() equivalent:
            eval_batches_sequence_names = list(
                {b[0][0]: None for b in eval_batch_index}.keys()
            )
            eval_sequence_name = eval_batches_sequence_names[test_restrict_sequence_id]
            eval_batch_index = [
                b for b in eval_batch_index if b[0][0] == eval_sequence_name
            ]
            # overwrite the restrict_sequence_name
            restrict_sequence_name = eval_sequence_name

        for dataset, subsets in set_names_mapping.items():
            frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
            assert os.path.isfile(frame_file)

            sequence_file = os.path.join(
                dataset_root, category, "sequence_annotations.jgz"
            )
            assert os.path.isfile(sequence_file)

            subset_lists_file = os.path.join(dataset_root, category, "set_lists.json")
            assert os.path.isfile(subset_lists_file)

            # TODO: maybe directly in param list
            params = {
                **copy.deepcopy(DATASET_CONFIGS["default"]),
                "frame_annotations_file": frame_file,
                "sequence_annotations_file": sequence_file,
                "subset_lists_file": subset_lists_file,
                "dataset_root": dataset_root,
                "limit_to": limit_to,
                "limit_sequences_to": limit_sequences_to,
                "n_frames_per_sequence": n_frames_per_sequence
                if dataset == "train"
                else -1,
                "subsets": subsets,
                "load_point_clouds": load_point_clouds,
                "mask_images": mask_images,
                "mask_depths": mask_depths,
                "pick_sequence": restrict_sequence_name and [restrict_sequence_name],
            }

            datasets[dataset] = Co3dDataset(**params)
            if dataset == "test":
                datasets[dataset].eval_batches = datasets[
                    dataset
                ].seq_frame_index_to_dataset_index(eval_batch_index)

        if assert_single_seq:
            # check theres only one sequence in all datasets
            assert (
                len(
                    set(
                        e["frame_annotation"].sequence_name
                        for dset in datasets.values()
                        for e in dset.frame_annots
                    )
                )
                <= 1
            ), "Multiple sequences loaded but expected one"

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if test_on_train:
        datasets["val"] = datasets["train"]
        datasets["test"] = datasets["train"]

    return datasets


def _get_co3d_set_names_mapping(
    dataset_name: str, test_on_train: bool
) -> typing.Dict[str, typing.List[str]]:
    """
    Returns the mapping of the common dataset subset names ("train"/"val"/"test")
    to the names of the corresponding subsets in the CO3D dataset
    ("test_known"/"test_unseen"/"train_known"/"train_unseen").
    """
    single_seq = dataset_name == "co3d_singlesequence"

    set_names_mapping = {
        "train": [
            (DATASET_TYPE_TEST if single_seq else DATASET_TYPE_TRAIN)
            + "_"
            + DATASET_TYPE_KNOWN
        ]
    }
    if not test_on_train:
        prefixes = [DATASET_TYPE_TEST]
        if not single_seq:
            prefixes.append(DATASET_TYPE_TRAIN)
        set_names_mapping.update(
            {
                dset: [
                    p + "_" + t
                    for p in prefixes
                    for t in [DATASET_TYPE_KNOWN, DATASET_TYPE_UNKNOWN]
                ]
                for dset in ["val", "test"]
            }
        )

    return set_names_mapping
