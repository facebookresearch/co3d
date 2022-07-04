# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#TODO: Move this to Implicitron!!!!!


import copy
import json
import os
import glob
from dataclasses import field
from typing import Any, Dict, List, Sequence, Tuple, Union
from omegaconf import DictConfig

from pytorch3d.implicitron.dataset.dataset_map_provider import (
    DatasetMap,
    DatasetMapProviderBase,
    Task,
    PathManagerFactory,
)
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import (
    registry, run_auto_creation, registry, get_default_args_field
)


CO3D_V2_CATEGORIES = [
    "apple","backpack","ball","banana","baseballbat","baseballglove",
    "bench","bicycle","book","bottle","bowl","broccoli","cake","car",
    "carrot","cellphone","chair","couch","cup","donut","frisbee","hairdryer",
    "handbag","hotdog","hydrant","keyboard","kite","laptop",# "microwave",
    "motorcycle","mouse","orange","parkingmeter","pizza","plant","remote",
    "sandwich","skateboard","stopsign","suitcase","teddybear","toaster",
    "toilet","toybus","toyplane","toytrain","toytruck","tv",
    "umbrella","vase","wineglass",
]


@registry.register
class CO3DV2DatasetMapProvider(DatasetMapProviderBase):  # pyre-ignore [13]
    """
    Generates the training / validation and testing dataset objects for
    a dataset laid out on disk like Co3D, with annotations in json files.

    Args:
        category: The object category of the dataset.
        subset: The dataset subset. One of:
            "multisequence"
            "singlesequence_<sequence_id>"
            "challenge_multisequence"
            "challenge_singlesequence"
        dataset_root: The root folder of the dataset.
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
            Active for task_str='singlesequence'.
        assert_single_seq: Assert that only frames from a single sequence
            are present in all generated datasets.
        only_test_set: Load only the test set.
        aux_dataset_kwargs: Specifies additional arguments to the
            JsonIndexDataset constructor call.
        path_manager: Optional[PathManager] for interpreting paths
    """

    category: str
    subset_name: str
    dataset_root: str = ""
    task_str: str = "singlesequence"

    test_on_train: bool = False
    only_test_set: bool = False
    load_eval_batches: bool = True

    dataset_args: DictConfig = get_default_args_field(JsonIndexDataset)
    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"


    def __post_init__(self):
        super().__init__()
        run_auto_creation(self)


    def _load_annotation_json(self, json_filename: str):
        full_path = os.path.join(
            self.dataset_root,
            self.category,
            json_filename,
        )
        path_manager = self.path_manager_factory.get()
        if path_manager is not None:
            full_path = self.path_manager.get_local_path(full_path)
        if not os.path.isfile(full_path):
            # The batch indices file does not exist.
            # Most probably the user has not specified the root folder.
            raise ValueError(
                f"Looking for dataset json file in {full_path}. "
                + "Please specify a correct dataset_root folder."
            )
        with open(full_path, "r") as f:
            data = json.load(f)
        return data


    def get_dataset_map(self) -> DatasetMap:
        if self.only_test_set and self.test_on_train:
            raise ValueError("Cannot have only_test_set and test_on_train")

        # TODO:
        # - implement loading multiple categories

        frame_file = os.path.join(
            self.dataset_root, self.category, "frame_annotations.jgz"
        )
        sequence_file = os.path.join(
            self.dataset_root, self.category, "sequence_annotations.jgz"
        )

        dataset_args = copy.deepcopy(self.dataset_args)
        dataset_args.update(dict(
            dataset_root=self.dataset_root,
            frame_annotations_file=frame_file,
            sequence_annotations_file=sequence_file,
            subset_lists_file="",
        ))

        dataset = JsonIndexDataset(**dataset_args)

        # load the list of train/val/test frames
        subset_mapping = self._load_annotation_json(
            os.path.join("set_lists", f"set_lists_{self.subset_name}.json")
        )

        # load the evaluation batches
        if self.load_eval_batches:
            eval_batch_index = self._load_annotation_json(
                os.path.join("eval_batches", f"eval_batches_{self.subset_name}.json")
            )

        train_dataset = None
        if not self.only_test_set:
            train_dataset = json_index_dataset_from_frame_index(dataset, subset_mapping["train"])

        if self.test_on_train:
            assert train_dataset is not None
            val_dataset = test_dataset = train_dataset
        else:
            val_dataset = json_index_dataset_from_frame_index(
                dataset, subset_mapping["val"]
            )
            test_dataset = json_index_dataset_from_frame_index(
                dataset, subset_mapping["test"]
            )
            if self.load_eval_batches:
                test_dataset.eval_batches = test_dataset.seq_frame_index_to_dataset_index(
                    eval_batch_index
                )
        
        datasets = DatasetMap(train=train_dataset, val=val_dataset, test=test_dataset)

        return datasets

    def get_task(self) -> Task:
        return Task(self.task_str)


def json_index_dataset_from_frame_index(
    dataset: JsonIndexDataset,
    frame_index: List[List[Union[Tuple[str, str], Tuple[str, str, str]]]],
):
    dataset_new = copy.deepcopy(dataset)
    dataset_indices = dataset_new.seq_frame_index_to_dataset_index([frame_index])[0]
    dataset_new.frame_annots = [dataset_new.frame_annots[i] for i in dataset_indices]
    dataset_new._invalidate_indexes(filter_seq_annots=True)
    for fa in dataset_new.frame_annots:
        if fa.meta is not None:
            fa.subset = fa.meta.get("frame_type", None)
    return dataset_new


def get_available_categories_and_subset_names(dataset_root: str) -> Dict[str, List[str]]:
    categories_to_subset_names = {}
    category_dirs = [
        d for d in glob.glob(os.path.join(dataset_root, "*"))
        if (os.path.isdir(d) and not d.startswith("_"))
    ]
    for category_dir in category_dirs:
        category = os.path.split(category_dir)[-1]
        categories_to_subset_names[category] = [
            os.path.splitext(os.path.split(f)[-1])[0].replace("eval_batches_", "")
            for f in glob(os.path.join(category_dir, "eval_batches", "*.json"))
        ]
    return categories_to_subset_names
