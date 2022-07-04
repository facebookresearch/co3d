import os
import shutil
import tempfile

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate

from co3d.challenge.metric_utils import EVAL_METRIC_NAMES

from .utils import evaluate_file_folders
from .data_types import RGBDAFrame, CO3DTask, CO3DSequenceSet
from .io import (
    load_all_eval_batches,
    store_rgbda_frame,
)


class CO3DSubmission:
    def __init__(
        self,
        task: CO3DTask,
        sequence_set: CO3DSequenceSet,
        output_folder: str,
        dataset_root: Optional[str] = None,
        server_data_folder: Optional[str] = None,
        on_server: bool = False,
    ):
        self.task = task
        self.sequence_set = sequence_set
        self.output_folder = output_folder
        self.dataset_root = dataset_root
        self.server_data_folder = server_data_folder
        self.on_server = on_server
        self.submission_archive = os.path.join(
            output_folder,
            f"submission_{task.value}_{sequence_set.value}.zip"
        )
        self.submission_cache = os.path.join(output_folder, "submission_cache")
        os.makedirs(self.submission_cache, exist_ok=True)
        self._result_list = []

    @staticmethod
    def get_submission_cache_image_dir(
        root_submission_dir: str,
        category: str,
        subset_name: str,
    ):
        return os.path.join(root_submission_dir, category, subset_name)

    def add_result(
        self,
        category: str,
        subset_name: str,
        sequence_name: str,
        frame_number: int,
        image: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray
    ):
        res = CO3DSubmissionRender(
            category=category,
            subset_name=subset_name,
            sequence_name=sequence_name,
            frame_number=frame_number,
            rgbda_frame=None,
        )
        res_file = res.get_image_path(self.submission_cache)
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        store_rgbda_frame(
            RGBDAFrame(image=image, mask=mask, depth=depth),
            res_file,
        )
        self._result_list.append(res)

    def _get_result_frame_index(self):
        return {
            (res.sequence_name, res.frame_number): res
            for res in self._result_list
        }

    def _get_eval_batches_map(self):
        if self._eval_batches_map is None:
            self._eval_batches_map = load_all_eval_batches(
                self.dataset_root, self.task, self.sequence_set, remove_frame_paths=False
            )
        else:
            return self._eval_batches_map

    def _validate_results(self):
        raise NotImplementedError("")
        result_frame_index = self._get_result_frame_index()
        eval_batches_map = self._get_eval_batches_map()
        all_eval_batches = [
            b for eval_batches in eval_batches_map.items() for b in eval_batches
        ]
        for (category, subset_list_name), eval_batches in eval_batches_map.items():
            for b in eval_batches:
                assert b in result_frame_index
            for fi in result_frame_index:
                assert fi in all_eval_batches

    def vallidate_results(self):
        self._validate_results()

    def export_results(self, validate_results: bool = True):
        if validate_results:
            self.validate_results()

        # zip the directory
        shutil.make_archive(self.output_folder, self.submission_archive)

        print(
            f"Exported the result file {self.submission_archive}."
            "Please submit the file to the EvalAI server."
        )

    def evaluate(self):
        if self.on_server:
            if not os.path.isdir(self.server_data_folder):
                raise ValueError(
                    "For evaluation on the server server_data_folder has to be specified."
                )
        else:
            if not os.path.isdir(self.dataset_root):
                raise ValueError("For evaluation dataset_root has to be specified.")
            if self.sequence_set==CO3DSequenceSet.TEST:
                raise ValueError("Cannot evaluate on the hidden test set!")
            
        eval_batches_map = self._get_eval_batches_map()

        eval_exceptions = {}
        eval_results = {}

        for (category, subset_name), eval_batches in eval_batches_map.items():

            pred_category_subset_dir = CO3DSubmission.get_submission_cache_image_dir(
                self.submission_cache,
                category,
                subset_name,
            )

            # Make a temporary GT folder with symlinks to GT data based on eval batches
            with tempfile.TemporaryDirectory() as gt_category_subset_dir:
                for b in eval_batches:
                    if self.on_server:
                        _link_eval_batch_data_from_server_db_to_gt_tempdir(
                            self.server_data_folder,
                            gt_category_subset_dir,
                            category,
                            b[0],
                        )
                    else:
                        _link_eval_batch_data_from_dataset_root_to_gt_tempdir(
                            self.dataset_root,
                            gt_category_subset_dir,
                            category,
                            b[0],
                        )
                
                # Evaluate and catch any exceptions.
                try:
                    eval_results[(category, subset_name)] = evaluate_file_folders(
                        pred_category_subset_dir,
                        gt_category_subset_dir,
                    )

                except Exception as exc:
                    eval_results[(category, subset_name)] = None, None
                    eval_exceptions[(category, subset_name)] = exc


        # Automatically generates NaNs if some results are missing.
        average_results = {}
        for m in EVAL_METRIC_NAMES:
            average_results[m] = sum(
                eval_result[m] if eval_result is not None else float("NaN")
                for (eval_result, _) in eval_results.values()
            ) / len(eval_results)
        eval_results[("MEAN", "-")] = average_results

        # Generate a nice table and print.
        for ((category, subset_name), (eval_result, _)) in eval_results.items():
            tab_row = [category, subset_name]
            if eval_result[0] is None:
                tab_row.extend([float("NaN")] * len(EVAL_METRIC_NAMES)])
            else:
                tab_row.extend(list(eval_result.values()))

        print(
            tabulate(tab_row, headers=["Category", "Subset name", *EVAL_METRIC_NAMES])
        )

        return eval_results


@dataclass
class CO3DSubmissionRender:
    category: str
    subset_name: str
    sequence_name: str
    frame_number: int
    rgbda_frame: Optional[RGBDAFrame] = None
        
    def get_image_path(self, root_dir: str):
        return os.path.join(
            CO3DSubmission.get_submission_cache_image_dir(
                root_dir,
                self.category,
                self.subset_name,
            ),
            self.get_image_name(),
        )

    def get_image_name(self):
        return f"{self.category}_{self.sequence_name}_{self.frame_number}"


def get_submission_image_name(category: str, sequence_name: str, frame_number: str):
    return f"{category}_{sequence_name}_{frame_number}"


def _link_eval_batch_data_from_dataset_root_to_gt_tempdir(
    dataset_root: str,
    temp_dir: str,
    category: str,
    frame_index: Tuple[str, int, str],
):
    sequence_name, frame_number, gt_image_path = frame_index
    image_name = get_submission_image_name(category, sequence_name, frame_number)
    for data_type in ["image", "depth", "mask", "depth_mask"]:
        gt_data_path = gt_image_path.replace("/images/", f"/{data_type}s/")
        if data_type != "image":
            gt_data_path = gt_data_path.replace(".jpg", ".png")
        tgt_image_name = f"{image_name}_{data_type}.png"
        src = os.path.join(dataset_root, gt_image_path)
        dst = os.path.join(temp_dir, tgt_image_name)
        print(f"{src}\n<---\n{dst}")
        # os.symlink(src, dst)


def _link_eval_batch_data_from_server_db_to_gt_tempdir(
    server_folder: str,
    temp_dir: str,
    category: str,
    frame_index: Tuple[str, int, str],
):
    sequence_name, frame_number, _ = frame_index
    image_name = get_submission_image_name(category, sequence_name, frame_number)
    for data_type in ["image", "depth", "mask", "depth_mask"]:
        image_name_postfixed = image_name + f"_{data_type}.png"
        src = os.path.join(server_folder, image_name_postfixed)
        dst = os.path.join(temp_dir, image_name_postfixed)
        print(f"{src}\n<---\n{dst}")
        # os.symlink(src, dst)


