import os
import json

from typing import Optional

from .data_types import CO3DSequenceSet, CO3DTask


def get_category_to_subset_name_list(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
):
    # make this file!
    json_file = os.path.join(dataset_root, "category_to_subset_name_list.json")
    with open(json_file, "r") as f:
        category_to_subset_name_list = json.load(json_file)
    
    # filter per-category subset lists by the selected task
    if task is not None:
        category_to_subset_name_list = {
            category: [
                subset_name for subset_name in subset_name_list
                if subset_name.startswith(task.value)
            ] for category, subset_name_list in category_to_subset_name_list.items()
        }

    # filter per-category subset lists by the selected sequence set
    if sequence_set is not None:
        category_to_subset_name_list = {
            category: [
                subset_name for subset_name in subset_name_list
                if f"_{sequence_set.value}" in subset_name
            ] for category, subset_name_list in category_to_subset_name_list.items()
        }

    # remove the categories with completely empty subset_name_lists
    category_to_subset_name_list = {
        c: l for c, l in category_to_subset_name_list.items() if len(l) > 0
    }

    return category_to_subset_name_list


def load_all_eval_batches(
    dataset_root: str,
    task: Optional[CO3DTask]=None,
    sequence_set: Optional[CO3DSequenceSet]=None,
    remove_frame_paths: bool = False,
):
    
    category_to_subset_name_list = get_category_to_subset_name_list(
        dataset_root,
        task=task,
        sequence_set=sequence_set,
    )
    
    eval_batches = {}
    for category, subset_name_list in category_to_subset_name_list:
        for subset_name in subset_name_list:
            # load the subset eval batches
            eval_batches[(category, subset_name)] = _load_eval_batches_file(
                dataset_root, category, subset_name, remove_frame_paths=remove_frame_paths
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
    eval_batches = [b[0] for b in eval_batches]  # take only the first (target evaluation) frame
    if remove_frame_paths:
        eval_batches = [b[:2] for b in eval_batches]
    return eval_batches


