# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import torch
import warnings
from tqdm import tqdm
from omegaconf import DictConfig


from pytorch3d.implicitron.models.generic_model import ImplicitronRender
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2
)
from pytorch3d.implicitron.tools.config import expand_args_fields

from co3d.utils import dbir_utils
from co3d.challenge.co3d_submission import CO3DSubmission
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet
from co3d.dataset.utils import redact_eval_frame_data, _check_valid_eval_frame_data


DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")
DATASET_ROOT_HIDDEN = os.getenv("CO3DV2_HIDDEN_DATASET_ROOT")


logger = logging.getLogger(__name__)


def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
) -> DatasetMap:
    """
    Obtain the dataset map that contains the train/val/test dataset objects.
    """
    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map_provider = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig({"remove_empty_masks": False}),
    )
    return dataset_map_provider.get_dataset_map()


@torch.no_grad()
def update_dbir_submission_with_category_and_subset_predictions(
    submission: CO3DSubmission,
    dataset_root: str,
    category: str,
    subset_name: str,
    num_workers: int = 12,
    cheat_with_gt_data: bool = True,
    load_dataset_pointcloud: bool = False,
    point_radius: float = 0.01,
):
    """
    Updates the CO3DSubmission object `submission` with predictions of a DBIR
    model extracted for a given category, and a dataset subset.

    Args:
        submission: CO3DSubmission object.
        dataset_root: Path to the root dataset folder containing CO3Dv2.
        category: A CO3Dv2 category to evaluate.
        subset_name: The name of the evaluation subset of the category.
        num_workers: Number of processes to use for evaluation.
        cheat_with_gt_data: If `True`, bypasses the DBIR stage and only simply
            uses ground truth test data. This, of course, only works for the
            development set which is not redacted.
        load_dataset_pointcloud: If `True`, uses the ground truth dataset
            pointclouds instead of unprojecting known views.
        point_radius: The radius of the rendered points.
    """

    logger.info(
        "Runing depth-based image rendering (DBIR) new view synthesis "
        f"on category '{category}' subset '{subset_name}'"
    )

    # Get the evaluation device.
    device = torch.device("cuda") if torch.cuda.is_available() else device("cpu")

    # Determine the sequence set and the task we are solving
    sequence_set = submission.sequence_set
    task = submission.task

    # Obtain the CO3Dv2 dataset map
    dataset_map = get_dataset_map(dataset_root, category, subset_name)

    if task==CO3DTask.MANY_VIEW and not cheat_with_gt_data:
        # Obtain the point cloud of the corresponding evaluation sequence
        # by unprojecting depth maps of the known training views in the sequence:
        train_dataset = dataset_map["train"]
        sequence_name = train_dataset[0].sequence_name
        sequence_pointcloud = dbir_utils.get_sequence_pointcloud(
            train_dataset,
            sequence_name,
            load_dataset_pointcloud=load_dataset_pointcloud,
        )
        # Move the pointcloud to the right device
        sequence_pointcloud = sequence_pointcloud.to(device)

    # The test dataloader simply iterates over test_dataset.eval_batches
    # this is done by setting test_dataset.eval_batches as the batch sampler
    test_dataset = dataset_map["test"]
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_dataset.eval_batches,
        num_workers=num_workers,
        collate_fn=FrameData.collate,
    )

    # loop over eval examples
    logger.info(
        f"Rendering {len(test_dataloader)} test views for {category}/{subset_name}"
    )

    if sequence_set==CO3DSequenceSet.TEST:
        # the test set contains images with redacted foreground masks which cause
        # the test dataloader to spam a warning message,
        # we suppress this warning with the following line
        warnings.filterwarnings("ignore", message="Empty masks_for_bbox.*")
    
    for eval_index, eval_frame_data in enumerate(tqdm(test_dataloader)):
        # the first element of eval_frame_data is the actual evaluation image,
        # the 2nd-to-last elements are the knwon source images used for building 
        # the reconstruction (source images are present only for the few-view task)

        # move the eval data to the requested device
        eval_frame_data = eval_frame_data.to(device)

        # sanity check that the eval frame data has correctly redacted entries
        _check_valid_eval_frame_data(eval_frame_data, task, sequence_set)

        if cheat_with_gt_data:
            # Cheat by taking the ground truth data. This should give in perfect metrics.
            mask_render = (eval_frame_data.fg_probability[:1] > 0.5).float()
            render_crop = ImplicitronRender(
                depth_render = eval_frame_data.depth_map[:1],
                image_render = eval_frame_data.image_rgb[:1] * mask_render,
                mask_render = mask_render,
            )

        else:
            if task==CO3DTask.MANY_VIEW:
                # we use the sequence pointcloud extracted above
                scene_pointcloud = sequence_pointcloud
            elif task==CO3DTask.FEW_VIEW:
                # we build the pointcloud by unprojecting the depth maps of the known views
                # which are elements (1:end) of the eval batch
                scene_pointcloud = dbir_utils.get_eval_frame_data_pointcloud(
                    eval_frame_data,
                )
            else:
                raise ValueError(task)
            # Redact the frame data so we are sure we cannot use the data
            # from the actual unobserved evaluation sample
            eval_frame_data = redact_eval_frame_data(eval_frame_data)
            # Obtain the image render. In case dataset_test.box_crop==True,
            # we need to paste the render back to the original image bounds.
            render_crop = dbir_utils.render_point_cloud(
                eval_frame_data.camera[[0]],
                eval_frame_data.image_rgb.shape[-2:],
                scene_pointcloud,
                point_radius=point_radius,
            )

        # cut the valid part of the render and paste into the original image canvas
        render_full_image = dbir_utils.paste_render_to_original_image(
            eval_frame_data, render_crop
        )

        # get the image, mask, depth as numpy arrays for the challenge submission
        image, mask, depth = [
            getattr(render_full_image, f"{data_type}_render").cpu().numpy()[0]
            for data_type in ["image", "mask", "depth"]
        ]

        # add the results to the submission object
        submission.add_result(
            category=category,
            subset_name=subset_name,
            sequence_name=eval_frame_data.sequence_name[0],
            frame_number=int(eval_frame_data.frame_number[0]),
            image=image,
            mask=mask,
            depth=depth,
        )

    # reset all warnings
    warnings.simplefilter("always")


def make_dbir_submission(
    dataset_root = DATASET_ROOT,
    task = CO3DTask.MANY_VIEW,
    sequence_set = CO3DSequenceSet.DEV,
    clear_submission_files: bool = False,
    num_eval_workers: int = 4,
    cheat_with_gt_data: bool = False,
    fill_results_from_cache: bool = False,
    skip_evaluation: bool = False,
    submit_to_eval_ai: bool = False,
):
    """
    Make a Depth-based-image-rendering (DBIR) submission for the CO3DChallenge.

    Args:
        dataset_root: Path to the root dataset folder.
        task: The co3d task - either CO3DTask.MANY_VIEW or CO3DTask.FEW_VIEW.
        sequence_set: The sequence set to evaluate on:
            CO3DSequenceSet.DEV for for the development set
            CO3DSequenceSet.TEST for for the test set
        clear_submission_files: Delete all previous intermediate submission files before
            commencing the current submission run.
        num_eval_workers: Number of processes that conduct evaluation.
        cheat_with_gt_data: If `True`, bypasses the DBIR stage and only simply
            uses ground truth test data. This, of course, only works for the
            development set which is not redacted.
        fill_results_from_cache: If `True`, skips running the DBIR model and rather 
            loads the results exported from a previous run.
        skip_evaluation: If `True`, will not locally evaluate the predictions.
        submit_to_eval_ai: If `True`, will automatically submit the exported result
            archive to EvalAI using the CLI interface (needs to be installed with 
            `pip install evalai`). This requires setting the EVAL_AI_PERSONAL_TOKEN 
            environment variable to your personal EVAL_AI token.
    """
    # the folder storing all predictions and results of the submission
    submission_output_folder = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        f"dbir_submission_output_{task.value}_{sequence_set.value}",
    )

    if cheat_with_gt_data:
        # make sure that the cheated results have a cheater stamp in their name
        submission_output_folder += "_cheating"

    # create the submission object
    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=submission_output_folder,
        dataset_root=DATASET_ROOT,
    )
    
    if task==CO3DTask.FEW_VIEW and submission.has_only_single_sequence_subset():
        # if only a single-sequence dataset is downloaded, only the many-view task
        # is available
        logger.warning(
            f"Cannot evaluate the few-view task in {sequence_set.value} when only the"
            " singlesequence subset of CO3D is present."
        )
        return

    if fill_results_from_cache:
        # only take existing results
        submission.fill_results_from_cache()
    
    else:
        # Clear all files generated by potential previous submissions.
        # Hint: disable this in case you want to resume an evaluation.
        if clear_submission_files:
            submission.clear_files()

        # Get all category names and subset names for the selected task/sequence_set
        eval_batches_map = submission.get_eval_batches_map()

        # Iterate over the categories and the corresponding subset lists.
        for eval_i, (category, subset_name) in enumerate(eval_batches_map.keys()):
            logger.info(
                f"Evaluating category {category}; subset {subset_name}"
                + f" ({eval_i+1} / {len(eval_batches_map)})"
            )

            # Generate new views for all evaluation examples in category/subset_name.
            update_dbir_submission_with_category_and_subset_predictions(
                submission=submission,
                dataset_root=dataset_root,
                category=category,
                subset_name=subset_name,
                cheat_with_gt_data=cheat_with_gt_data,
            )

    # Locally evaluate the submission in case we dont evaluate on the hidden test set.
    if (not skip_evaluation and sequence_set != CO3DSequenceSet.TEST):
        submission.evaluate(num_workers=num_eval_workers)

    # Export the submission predictions for submition to the evaluation server.
    # This also validates completeness of the produced predictions.
    submission.export_results(validate_results=True)

    if submit_to_eval_ai:
        # submit the results to the EvalAI server.
        submission.submit_to_eval_ai()

    # sanity check - reevaluate the archive file and copare results
    # submission_reeval = CO3DSubmission(
    #     task=task,
    #     sequence_set=sequence_set,
    #     output_folder=os.path.join(submission_output_folder, "_reeval"),
    #     dataset_root=DATASET_ROOT,
    #     on_server=True,
    #     server_data_folder=DATASET_ROOT_HIDDEN,
    # )
    # submission_reeval.evaluate_archive_file(
    #     submission.submission_archive, num_workers=num_eval_workers
    # )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # iterate over all tasks and sequence sets
    for sequence_set in [CO3DSequenceSet.DEV, CO3DSequenceSet.TEST]:
        for task in [CO3DTask.MANY_VIEW, CO3DTask.FEW_VIEW]:
            make_dbir_submission(
                task=task,
                sequence_set=sequence_set,
                cheat_with_gt_data=False,
                fill_results_from_cache=False,
                skip_evaluation=False,
                submit_to_eval_ai=True,
            )