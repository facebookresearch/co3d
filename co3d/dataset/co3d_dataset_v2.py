# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# TODO: Move this to Implicitron!!!!!


import copy
import json
import os
import glob
import logging
import warnings
from typing import List, Tuple, Union
from omegaconf import DictConfig

from pytorch3d.implicitron.dataset.dataset_map_provider import (
    DatasetMap,
    DatasetMapProviderBase,
    Task,
    PathManagerFactory,
)
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import (
    registry,
    run_auto_creation,
    registry,
    get_default_args_field,
    expand_args_fields,
)


logger = logging.getLogger(__name__)


@registry.register
class CO3DV2DatasetMapProvider(DatasetMapProviderBase):  # pyre-ignore [13]
    """
    Generates the training / validation and testing dataset objects for
    a dataset laid out on disk like Co3D, with annotations in json files.

    Args:
        category: The object category of the dataset.
        subset: The dataset subset.
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

    category: str = ""
    subset_name: str = ""
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


    def get_dataset_map(self) -> DatasetMap:
        if self.only_test_set and self.test_on_train:
            raise ValueError("Cannot have only_test_set and test_on_train")

        frame_file = os.path.join(
            self.dataset_root, self.category, "frame_annotations.jgz"
        )
        sequence_file = os.path.join(
            self.dataset_root, self.category, "sequence_annotations.jgz"
        )

        path_manager = self.path_manager_factory.get()

        common_dataset_kwargs = dict(
            **self.dataset_args,
            dataset_root=self.dataset_root,
            frame_annotations_file=frame_file,
            sequence_annotations_file=sequence_file,
            subsets=None,
            subset_lists_file="",
            path_manager=path_manager,
        )

        expand_args_fields(JsonIndexDataset)
        dataset =  JsonIndexDataset(**common_dataset_kwargs)

        available_subset_names = self._get_available_subset_names()
        logger.debug(f"Available subset names: {str(available_subset_names)}.")
        if self.subset_name not in available_subset_names:
            raise ValueError(
                f"Unknown subset name {self.subset_name}."
                + f" Choose one of available subsets: {str(available_subset_names)}."
            )

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
            # load the training set
            logger.debug("Loading train dataset.")
            train_dataset = json_index_dataset_from_frame_index(dataset, subset_mapping["train"])
            logger.info(f"Train dataset: {str(train_dataset)}")

        if self.test_on_train:
            assert train_dataset is not None
            val_dataset = test_dataset = train_dataset
        else:
            # load the val and test sets
            logger.debug("Loading val dataset.")
            val_dataset = json_index_dataset_from_frame_index(
                dataset, subset_mapping["val"]
            )
            logger.info(f"Val dataset: {str(val_dataset)}")
            logger.debug("Loading test dataset.")
            test_dataset = json_index_dataset_from_frame_index(
                dataset, subset_mapping["test"]
            )
            logger.info(f"Test dataset: {str(test_dataset)}")
            if self.load_eval_batches:
                # load the eval batches
                logger.debug("Loading eval batches.")
                try:
                    test_dataset.eval_batches = test_dataset.seq_frame_index_to_dataset_index(
                        eval_batch_index,
                    )
                except IndexError:
                    warnings.warn(
                        "Some eval batches are missing from the test dataset."
                        + " The evaluation results will be incomparable to the"
                        + " evaluation results calculated on the original dataset."
                    )
                    test_dataset.eval_batches = test_dataset.seq_frame_index_to_dataset_index(
                        eval_batch_index,
                        allow_missing_indices=True,
                    )
                logger.info(f"# eval batches: {len(test_dataset.eval_batches)}")
        
        return DatasetMap(train=train_dataset, val=val_dataset, test=test_dataset)


    def get_task(self) -> Task:
        return Task(self.task_str)


    def _load_annotation_json(self, json_filename: str):
        full_path = os.path.join(
            self.dataset_root,
            self.category,
            json_filename,
        )
        logger.info(f"Loading frame index json from {full_path}.")
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


    def _get_available_subset_names(self):
        path_manager = self.path_manager_factory.get()
        if path_manager is not None:
            dataset_root = self.path_manager.get_local_path(self.dataset_root)
        else:
            dataset_root = self.dataset_root
        return get_available_subset_names(dataset_root, self.category)
        

def json_index_dataset_from_frame_index(
    dataset: JsonIndexDataset,
    frame_index: List[List[Union[Tuple[str, str], Tuple[str, str, str]]]],
    allow_missing_indices: bool = True,
) -> JsonIndexDataset:
    # Get the indices into the frame annots.
    dataset_indices = dataset.seq_frame_index_to_dataset_index(
        [frame_index],
        allow_missing_indices=dataset.is_filtered() and allow_missing_indices,
    )[0]
    valid_dataset_indices = [i for i in dataset_indices if i is not None]

    # Deep copy the whole dataset except frame_annots, which are large so we
    # deep copy only the requested subset of frame_annots.
    memo = {id(dataset.frame_annots): None}  # ignores frame_annots during deepcopy
    dataset_new = copy.deepcopy(dataset, memo)
    dataset_new.frame_annots = copy.deepcopy(
        [dataset.frame_annots[i] for i in valid_dataset_indices]
    )
    
    # This will kill all unneeded sequence annotations.
    dataset_new._invalidate_indexes(filter_seq_annots=True)
    
    # Finally annotate the frame annotations with the name of the subset
    # stored in meta.
    for frame_annot in dataset_new.frame_annots:
        frame_annotation = frame_annot["frame_annotation"]
        if frame_annotation.meta is not None:
            frame_annot["subset"] = frame_annotation.meta.get("frame_type", None)
    
    # A sanity check - this will crash in case some entries from frame_index are missing
    # in dataset_new.
    valid_frame_index = [
        fi for fi, di in zip(frame_index, dataset_indices) if di is not None
    ]
    _ = dataset_new.seq_frame_index_to_dataset_index(
        [valid_frame_index], allow_missing_indices=False
    )[0]

    return dataset_new


def get_available_subset_names(dataset_root: str, category: str) -> List[str]:
    """
    Get the available subset names for a given category folder inside a root dataset
    folder `dataset_root`.
    """
    category_dir = os.path.join(dataset_root, category)
    if not os.path.isdir(category_dir):
        raise ValueError(
            f"Looking for dataset files in {category_dir}. "
            + "Please specify a correct dataset_root folder."
        )
    set_list_jsons = os.listdir(os.path.join(category_dir , "set_lists"))
    return [
        json_file.replace("set_lists_", "").replace(".json", "")
        for json_file in set_list_jsons
    ]
