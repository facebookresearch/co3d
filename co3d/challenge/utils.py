# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import zipfile
import glob
import logging
import multiprocessing
import numpy as np
import time

from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from .data_types import CO3DSequenceSet, CO3DTask, RGBDAFrame
from .metric_utils import eval_one, EVAL_METRIC_NAMES, Timer
from .io import load_rgbda_frame


logger = logging.getLogger(__file__)


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
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def check_user_submission_file_paths(
    ground_truth_files: Dict[str, str],
    user_submission_files: Dict[str, str],
):
    missing_gt_examples = [
        gt_example_name
        for gt_example_name in ground_truth_files
        if gt_example_name not in user_submission_files
    ]
    if len(missing_gt_examples) > 0:
        raise ValueError(
            f"There are missing evaluation examples: {str(missing_gt_examples)}"
        )

    additional_user_examples = [
        user_example
        for user_example in user_submission_files
        if user_example not in ground_truth_files
    ]
    if len(additional_user_examples) > 0:
        raise ValueError(
            f"Unexpected submitted evaluation examples {str(additional_user_examples)}"
        )


def get_data_type_postfix(data_type: str):
    assert data_type in ["image", "mask", "depth", "depth_mask"]
    return f"_{data_type}.png"


def get_result_directory_file_names(
    result_dir: str, has_depth_masks: bool = False,
) -> Dict[str, str]:
    """
    Result directory structure:
        <test_example_name>-image.png
        <test_example_name>-mask.png
        <test_example_name>-depth.png
        ...

    Returns:
        result_files: dict {test_example_name_i: root_path_i}
    """

    result_type_files = {}
    for result_type in ("image", "mask", "depth"):
        postfix = get_data_type_postfix(result_type)
        matching_files = sorted(glob.glob(os.path.join(result_dir, f"*{postfix}")))
        if has_depth_masks and result_type=="mask":
            matching_files = [
                f for f in matching_files 
                if not f.endswith(get_data_type_postfix("depth_mask"))
            ]
        result_type_files[result_type] = {
            os.path.split(f)[-1][: -len(postfix)]: f for f in matching_files
        }

    example_names = sorted(
        list(
            set(
                [
                    n
                    for t in ("image", "mask", "depth")
                    for n in result_type_files[t].keys()
                ]
            )
        )
    )

    missing_examples = defaultdict(list)
    for example_name in example_names:
        for result_type in ("image", "mask", "depth"):
            if example_name not in result_type_files[result_type]:
                missing_examples[example_name].append(result_type)

    if len(missing_examples) > 0:
        msg = "\n".join(
            [f"   {k} missing {str(v)}" for k, v in missing_examples.items()]
        )
        raise ValueError(
            f"Some evaluation examples in {result_dir} are incomplete:\n"
            + msg
        )

    result_files = {
        example_name: result_type_files["image"][example_name][: -len("_image.png")]
        for example_name in example_names
    }

    return result_files

def _evaluate_pred_gt_pair(args: Tuple[str, str, str, float, bool]):
    gt_example, gt_file, pred_file, max_time, print_status = args
    cur_time = time.time()
    if cur_time > max_time:
        raise ValueError(
            "    @@@@@@@@@@@@@@@@@@@@@\n"
            "    Evaluation timed out!\n"
            "    @@@@@@@@@@@@@@@@@@@@@"
        )
    # with Timer("io"):
    gt_rgbda = load_rgbda_frame(gt_file, check_for_depth_mask=True)
    pred_rgbda = load_rgbda_frame(pred_file)
    # with Timer("check"):
    check_same_rgbda_sizes(gt_rgbda, pred_rgbda, gt_example)
    # with Timer("eval"):
    eval_result_one = eval_one(pred_rgbda, gt_rgbda)
    for k, v in eval_result_one.items():
        if not np.isfinite(v):
            raise ValueError(f"{gt_example} - {k} is does not have a finite value.")
    if print_status:
        msg = "; ".join([f"{k}={v:.3f}" for k, v in eval_result_one.items()])
        sz = str(list(gt_rgbda.image.shape[-2:])).replace(" ", "")
        logger.info(
            f"eval_one({gt_example}-[{sz}]): {msg}; {max_time-cur_time:.1f} sec left"
        )
    return eval_result_one
    

def evaluate_file_folders(
    pred_folder: str,
    gt_folder: str,
    num_workers: int = 0,
    remaining_time: float = float("Inf"),
    print_per_example_results: bool = True,
):
    # determine how much time do we have for the evaluation
    max_time = time.time() + remaining_time 

    user_submission_files = get_result_directory_file_names(pred_folder)
    ground_truth_files = get_result_directory_file_names(gt_folder, has_depth_masks=True)

    logger.info(f"Evaluating folders: prediction={pred_folder}; gt={gt_folder}")
    check_user_submission_file_paths(
        ground_truth_files,
        user_submission_files,
    )

    # At this point we are sure that ground_truth_files contain the same
    # examples as user_submission_files.

    if num_workers <= 0:
        # Iterate over the gt examples:
        per_example_results = [
            _evaluate_pred_gt_pair(
                (
                    gt_example,
                    ground_truth_files[gt_example],
                    user_submission_files[gt_example],
                    max_time,
                    print_per_example_results,
                )
            ) for gt_example in tqdm(list(ground_truth_files))
        ]    
        # gt_rgbda = load_rgbda_frame(ground_truth_files[gt_example], check_for_depth_mask=True)
        # pred_rgbda = load_rgbda_frame(user_submission_files[gt_example])
        # check_same_rgbda_sizes(gt_rgbda, pred_rgbda, gt_example)
        # per_example_results.append(eval_one(pred_rgbda, gt_rgbda))
    else:
        # parallel processing
        arg_list = [
            (
                gt_example,
                ground_truth_files[gt_example],
                user_submission_files[gt_example],
                max_time,
                print_per_example_results,
            ) for gt_example in list(ground_truth_files)
        ]
        pool = multiprocessing.Pool(num_workers)
        per_example_results = [
            result for result in tqdm(
                pool.imap(_evaluate_pred_gt_pair, arg_list),
                total=len(arg_list),
            )
        ]
        pool.terminate()
        
    result = {
        metric: (sum(r[metric] for r in per_example_results) / len(per_example_results))
        for metric in EVAL_METRIC_NAMES
    }

    return result, per_example_results


def check_same_rgbda_sizes(gt: RGBDAFrame, pred: RGBDAFrame, example_name: str):
    for data_type in ("image", "mask", "depth"):
        gt_size, pred_size = [getattr(x, data_type).shape for x in [gt, pred]]
        if gt_size != pred_size:
            raise ValueError(
                f"{example_name}'s size does not match the ground truth."
                f"{data_type} size: {str(gt_size)} != {str(pred_size)}"
                " (ground-truth vs. prediction)."
            )
    return True


def get_annotations_folder(phase_codename: str):
    assert phase_codename in {"dev", "test"}
    return os.path.join("annotations", phase_codename)