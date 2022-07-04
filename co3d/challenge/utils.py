# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import zipfile
import glob

from collections import defaultdict
from typing import List, Dict
from .data_types import CO3DSequenceSet, CO3DTask


def get_co3d_task_from_subset_name(subset_name: str) -> CO3DTask:
    if subset_name.startswith("manyview"):
        return CO3DTask.MANY_VIEW
    elif subset_name.startswith("fewview"):
        return CO3DTask.FEW_VIEW
    else:
        raise ValueError(f"Invalid subset name {subset_name}!")


def get_co3d_sequence_set_from_subset_name(subset_name: str) -> CO3DSequenceSet:
    return CO3DSequenceSet(subset_name.split("_")[1])
    

def unzip(file_path: str, output_dir: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def check_user_submission_file_paths(
    ground_truth_files: Dict[str, str],
    user_submission_files: Dict[str, str],
):
    missing_gt_examples = [
        gt_example_name for gt_example_name in ground_truth_files 
        if gt_example_name not in user_submission_files
    ]
    if len(missing_gt_examples) > 0:
        raise ValueError(
            f"There are missing evaluation examples: {str(missing_gt_examples)}"
        )

    additional_user_examples = [
        user_example for user_example in user_submission_files 
        if user_example not in ground_truth_files
    ]
    if len(additional_user_examples) > 0:
        raise ValueError(
            f"Unexpected submitted evaluation examples {str(additional_user_examples)}"
        )


def get_result_directory_file_names(result_dir: str) -> Dict[str, str]:
    """
    Result directory structure:
        <test_example_name>_image.png
        <test_example_name>_mask.png
        <test_example_name>_depth.png
        ...

    Returns:
        result_files: dict {test_example_name_i: root_path_i}
    """

    result_type_files = {}
    for result_type in ("image", "mask", "depth"):
        postfix = f"_{result_type}.png"
        matching_files = sorted(glob.glob(os.path.join(result_dir, f"*{postfix}")))
        result_type_files[result_type] = {
            os.path.split(f)[-1][:-len(postfix)]: f
            for f in matching_files
        }

    example_names = sorted(list(set([
        n for t in ("image", "mask", "depth") for n in result_type_files[t].keys()
    ])))
    
    missing_examples = defaultdict(list)
    for example_name in example_names:
        for result_type in ("image", "mask", "depth"):
            if example_name not in result_type_files[result_type]:
                missing_examples[example_name].append(result_type)
    
    if len(missing_examples) > 0:
        msg = "\n".join([f"   {k} missing {str(v)}" for k, v in missing_examples.items()])
        raise ValueError("Some evaluation examples are incomplete:\n" + msg)
        
    result_files = {
        example_name: result_type_files["image"][example_name][:-len("_image.png")]
        for example_name in example_names
    }

    return result_files


# def list_present_categories_and_subset_names_and_images(submission_dir: str):
#     categories = os.path.listdir(submission_dir)
#     for category in categories:
#         subset_names = os.path.listdir(os.path.join(submission_dir, category))
#         for subset_name in subset_names:
#             os.path.listdir(
#                 os.path.listdir(os.path.join(submission_dir, category, subset_names))
#             )


def get_annotations_folder(phase_codename: str):
    assert phase_codename in {"dev", "test"}
    return os.path.join("annotations", phase_codename)
