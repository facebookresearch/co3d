import random
import tempfile

from .utils import (
    get_result_directory_file_names,
    get_annotations_folder,
    check_user_submission_file_paths,
    unzip,
    evaluate_file_folders,
)

from .metric_utils import eval_one, EVAL_METRIC_NAMES
from .io import load_rgbda_frame



def evaluate_file_folders(pred_folder: str, gt_folder: str):
    user_submission_files = get_result_directory_file_names(pred_folder)
    ground_truth_files = get_result_directory_file_names(gt_folder)

    check_user_submission_file_paths(
        ground_truth_files,
        user_submission_files,
    )

    # At this point we are sure that ground_truth_files contain the same
    # examples as user_submission_files.
    
    # Iterate over the gt examples:
    per_example_results = [
        eval_one(
            load_rgbda_frame(ground_truth_files[gt_example]),
            load_rgbda_frame(user_submission_files[gt_example]),
        ) for gt_example in ground_truth_files
    ]

    result = {
        metric: (
            sum(r[metric] for r in per_example_results)
            / len(per_example_results)
        ) for metric in EVAL_METRIC_NAMES
    }

    return result, per_example_results