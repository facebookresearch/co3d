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


from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2
)
from pytorch3d.implicitron.tools.config import expand_args_fields

from co3d.utils import dbir_utils
from co3d.challenge.io import get_category_to_subset_name_list
from co3d.challenge.co3d_submission import CO3DSubmission
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet
from co3d.dataset.utils import redact_eval_frame_data, _check_valid_eval_frame_data

DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")
DATASET_ROOT_HIDDEN = os.path.join(DATASET_ROOT, "_hidden", "hidden")
DATASET_ROOT_HIDDEN_KNOWN = os.path.join(DATASET_ROOT, "_hidden", "known")
ON_SERVER = False


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
    dataset_map = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig({"remove_empty_masks": False}),
    )
    return dataset_map.get_dataset_map()


@torch.no_grad()
def update_dbir_submission_with_category_and_subset_predictions(
    submission: CO3DSubmission,
    dataset_root: str,
    category: str,
    subset_name: str,
    num_workers: int = 12,
):
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

    # Take the training dataset for building the rendered models.
    if task==CO3DTask.MANY_VIEW:
        # Obtain the point cloud of the corresponding evaluation sequence
        # by unprojecting depth maps of the known training views in the sequence:
        train_dataset = dataset_map["train"]
        sequence_name = train_dataset[0].sequence_name
        sequence_pointcloud = dbir_utils.get_sequence_pointcloud(
            train_dataset,
            sequence_name,
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

        # redact the frame data so we are sure we cannot use the data
        # from the actual unobserved evaluation sample
        eval_frame_data = redact_eval_frame_data(eval_frame_data)

        # obtain the image render in the image coords as output by the test dataloader
        render_crop = dbir_utils.render_point_cloud(
            eval_frame_data.camera[[0]],
            eval_frame_data.image_rgb.shape[-2:],
            scene_pointcloud,
            point_radius=0.01,
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
    clear_submission_files: bool = True,
):
    # the folder storing all predictions and results of the submission
    submission_output_folder = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        f"dbir_submission_files_{task.value}_{sequence_set.value}",
    )

    # create the submission object
    if not ON_SERVER:
        # local evaluation
        submission = CO3DSubmission(
            task=task,
            sequence_set=sequence_set,
            output_folder=submission_output_folder,
            dataset_root=DATASET_ROOT,
        )
    else:
        # evaluation on server (only for internal use)
        submission = CO3DSubmission(
            task=task,
            sequence_set=sequence_set,
            output_folder=submission_output_folder,
            dataset_root=DATASET_ROOT,
            on_server=True,
            server_data_folder=DATASET_ROOT_HIDDEN,
        )

    if task==CO3DTask.FEW_VIEW and submission.has_only_single_sequence_subset():
        # if only a single-sequence dataset is downloaded, only the many-view task
        # is available
        logger.warning(
            f"Cannot evaluate the few-view task in {sequence_set.value} when only the"
            " singlesequence subset of CO3D is present."
        )
        return

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
        )

    # Locally evaluate the submission in case we dont evaluate on the hidden test set.
    if not(sequence_set == CO3DSequenceSet.TEST and not ON_SERVER):
        submission.evaluate()

    # Export the submission predictions for submition to the evaluation server.
    # This also validates completeness of the produced predictions.
    submission.export_results(validate_results=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # iterate over all tasks and sequence sets
    for sequence_set in [CO3DSequenceSet.DEV, CO3DSequenceSet.TEST]:
        for task in [CO3DTask.FEW_VIEW, CO3DTask.MANY_VIEW]:
            make_dbir_submission(task=task, sequence_set=sequence_set)


# manyview_dev DBIR results
# remote      manyview_dev_1  16.1192    14.5797        0.913678  0.676331                                                                                                                [918/19810]
# plant       manyview_dev_0  17.8395    15.0772        0.993802  0.61332                                                                                                                            plant       manyview_dev_1  18.4156    15.0422        1.0066    0.557596
# orange      manyview_dev_0  23.8881    17.8702        0.578254  0.864773
# orange      manyview_dev_1  22.6604    19.0574        0.566899  0.822498
# mouse       manyview_dev_0  21.7745    14.4547        0.860928  0.698297
# mouse       manyview_dev_1  19.9409    15.0666        0.57766   0.681226
# hydrant     manyview_dev_0  18.5751    15.7301        0.95568   0.712309
# hydrant     manyview_dev_1  19.4675    15.4154        0.912708  0.724743
# donut       manyview_dev_0  19.414     11.6615        0.545807  0.756596
# donut       manyview_dev_1  30.0177    20.3176        0.460116  0.867101
# cake        manyview_dev_0  22.466     16.8395        0.379309  0.87776
# cake        manyview_dev_1  22.019     20.2707        1.55181   0.639633                                                                                                                           broccoli    manyview_dev_0  24.4093    17.8134        0.683778  0.733446
# broccoli    manyview_dev_1  25.0271    15.6186        0.469152  0.885385
# bowl        manyview_dev_0  22.9925    17.2806        0.414035  0.794273
# bowl        manyview_dev_1  15.708     14.603         0.789081  0.684373
# book        manyview_dev_0  22.4066    18.2975        0.705954  0.820631
# book        manyview_dev_1  17.5983    12.8939        0.978277  0.785022                                                                                                                           bench       manyview_dev_0  15.0113    16.0221        0.916731  0.565511
# bench       manyview_dev_1  15.3114    15.7725        0.747333  0.693373                                                                                                                           ball        manyview_dev_0  21.1948    16.8647        0.48141   0.820352
# ball        manyview_dev_1  15.4156    15.2333        0.370925  0.737077
# apple       manyview_dev_0  26.6793    18.8923        0.453349  0.900099
# apple       manyview_dev_1  19.9979    17.0401        0.57107   0.806537
# MEAN        -               20.202     15.9815        0.823287  0.735673