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
    clear_submission_files: bool = False,
    num_eval_workers: int = 4,
):
    # the folder storing all predictions and results of the submission
    submission_output_folder = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        f"dbir_submission_output_{task.value}_{sequence_set.value}",
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
        submission.evaluate(num_workers=num_eval_workers)

    # Export the submission predictions for submition to the evaluation server.
    # This also validates completeness of the produced predictions.
    submission.export_results(validate_results=True)

    # sanity check - reevaluate the zip file and copare results
    submission_reeval = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=os.path.join(submission_output_folder, "_reeval"),
        dataset_root=DATASET_ROOT,
        on_server=True,
        server_data_folder=DATASET_ROOT_HIDDEN,
    )
    submission_reeval.evaluate_zip_file(submission.submission_archive)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # iterate over all tasks and sequence sets
    for sequence_set in [CO3DSequenceSet.DEV, CO3DSequenceSet.TEST]:
        for task in [CO3DTask.MANY_VIEW, CO3DTask.FEW_VIEW]:
            make_dbir_submission(task=task, sequence_set=sequence_set)


# CO3D challenge results
# Category    Subset name        psnr    psnr_fg    depth_abs_fg       iou
# ----------  --------------  -------  ---------  --------------  --------
# vase        manyview_dev_0  24.7188    17.0456        0.750426  0.797131
# vase        manyview_dev_1  25.0695    18.4118        0.633954  0.800592
# toytruck    manyview_dev_0  22.5549    15.0314        0.909745  0.708762
# toytruck    manyview_dev_1  17.51      12.0589        0.928104  0.688089
# toytrain    manyview_dev_0  20.5487    15.2528        0.777094  0.731648
# toytrain    manyview_dev_1  25.2285    17.6799        0.95936   0.643873
# toaster     manyview_dev_0  21.578     13.7402        0.665852  0.696084
# toaster     manyview_dev_1  18.3078    12.1653        0.871723  0.693956
# teddybear   manyview_dev_0  23.7967    16.5354        0.460334  0.838103
# teddybear   manyview_dev_1  23.7718    18.0498        0.973132  0.726019
# suitcase    manyview_dev_0  23.0181    17.0669        1.39468   0.787117
# suitcase    manyview_dev_1  27.9742    18.7447        0.83167   0.759975
# skateboard  manyview_dev_0  17.0379    12.0626        0.88807   0.699728
# skateboard  manyview_dev_1  18.0029    14.911         1.30745   0.647891
# remote      manyview_dev_0  19.0747    10.8591        0.54138   0.780551
# remote      manyview_dev_1  18.707     12.7299        0.789577  0.708073
# plant       manyview_dev_0  20.3458    15.7513        0.786387  0.665394
# plant       manyview_dev_1  21.2544    15.9596        1.15135   0.616215
# orange      manyview_dev_0  22.1965    13.901         1.42191   0.767037
# orange      manyview_dev_1  21.1569    11.9308        1.09544   0.673262
# mouse       manyview_dev_0  23.5896    14.9893        0.834395  0.729335
# mouse       manyview_dev_1  22.2867    15.1618        0.666281  0.678549
# hydrant     manyview_dev_0  22.4799    16.9655        0.545597  0.836734
# hydrant     manyview_dev_1  22.9273    16.0069        0.558497  0.825344
# donut       manyview_dev_0  23.4415    14.8969        0.797778  0.72526
# donut       manyview_dev_1  29.782     18.2328        0.624935  0.795212
# cake        manyview_dev_0  22.7506    14.522         0.661419  0.783267
# cake        manyview_dev_1  25.3902    20.5686        1.07715   0.763787
# broccoli    manyview_dev_0  25.3132    16.0099        0.947191  0.692499
# broccoli    manyview_dev_1  26.19      16.1295        0.76257   0.777639
# bowl        manyview_dev_0  25.7727    15.2469        0.452283  0.810967
# bowl        manyview_dev_1  17.9696    13.8592        0.979186  0.723606
# book        manyview_dev_0  22.3838    14.7637        0.545505  0.806584
# book        manyview_dev_1  19.5617    12.1536        0.684921  0.833463
# bench       manyview_dev_0  19.2668    18.3066        0.504531  0.723418
# bench       manyview_dev_1  17.4578    13.0467        0.702295  0.716847
# ball        manyview_dev_0  20.1323    12.7333        0.904389  0.66537
# ball        manyview_dev_1  15.7553    11.0936        1.27253   0.61227
# apple       manyview_dev_0  25.3796    15.8355        0.8537    0.745961
# apple       manyview_dev_1  21.1       14.771         0.992911  0.738451
# MEAN        -               22.0196    15.1295        0.837643  0.735352


# Category    Subset name        psnr    psnr_fg    depth_abs_fg       iou    psnr_full_image
# ----------  --------------  -------  ---------  --------------  --------  -----------------
# vase        manyview_dev_0  24.7188    17.0456        0.750426  0.797131            3.50081
# vase        manyview_dev_1  25.0695    18.4118        0.633954  0.800592            6.41699
# toytruck    manyview_dev_0  22.5549    15.0314        0.909745  0.708762            4.99781
# toytruck    manyview_dev_1  17.51      12.0589        0.928104  0.688089            7.80806
# toytrain    manyview_dev_0  20.5487    15.2528        0.777094  0.731648            8.00421
# toytrain    manyview_dev_1  25.2285    17.6799        0.95936   0.643873            9.19275
# toaster     manyview_dev_0  21.578     13.7402        0.665852  0.696084            6.15986
# toaster     manyview_dev_1  18.3078    12.1653        0.871723  0.693956            6.80434
# teddybear   manyview_dev_0  23.7967    16.5354        0.460334  0.838103            5.63069
# teddybear   manyview_dev_1  23.7718    18.0498        0.973132  0.726019            5.58213
# suitcase    manyview_dev_0  23.0181    17.0669        1.39468   0.787117            6.67624
# suitcase    manyview_dev_1  27.9742    18.7447        0.83167   0.759975            5.53953
# skateboard  manyview_dev_0  17.0379    12.0626        0.88807   0.699728            8.6203
# skateboard  manyview_dev_1  18.0029    14.911         1.30745   0.647891            7.31757
# remote      manyview_dev_0  19.0747    10.8591        0.54138   0.780551            6.89078
# remote      manyview_dev_1  18.707     12.7299        0.789577  0.708073            4.34828
# plant       manyview_dev_0  20.3458    15.7513        0.786387  0.665394            7.40235
# plant       manyview_dev_1  21.2544    15.9596        1.15135   0.616215            9.42299
# orange      manyview_dev_0  22.1965    13.901         1.42191   0.767037            6.29999
# orange      manyview_dev_1  21.1569    11.9308        1.09544   0.673262            6.25906
# mouse       manyview_dev_0  23.5896    14.9893        0.834395  0.729335            6.09629
# mouse       manyview_dev_1  22.2867    15.1618        0.666281  0.678549            7.38962
# hydrant     manyview_dev_0  22.4799    16.9655        0.545597  0.836734            5.85919
# hydrant     manyview_dev_1  22.9273    16.0069        0.558497  0.825344            5.04459
# donut       manyview_dev_0  23.4415    14.8969        0.797778  0.72526             4.06814
# donut       manyview_dev_1  29.782     18.2328        0.624935  0.795212            4.27905
# cake        manyview_dev_0  22.7506    14.522         0.661419  0.783267            6.25355
# cake        manyview_dev_1  25.3902    20.5686        1.07715   0.763787            3.99719
# broccoli    manyview_dev_0  25.3132    16.0099        0.947191  0.692499            6.78466
# broccoli    manyview_dev_1  26.19      16.1295        0.76257   0.777639            4.81145
# bowl        manyview_dev_0  25.7727    15.2469        0.452283  0.810967            5.87919
# bowl        manyview_dev_1  17.9696    13.8592        0.979186  0.723606            7.25215
# book        manyview_dev_0  22.3838    14.7637        0.545505  0.806584            6.65789
# book        manyview_dev_1  19.5617    12.1536        0.684921  0.833463            7.18136
# bench       manyview_dev_0  19.2668    18.3066        0.504531  0.723418            4.45801
# bench       manyview_dev_1  17.4578    13.0467        0.702295  0.716847            5.13987
# ball        manyview_dev_0  20.1323    12.7333        0.904389  0.66537             5.96311
# ball        manyview_dev_1  15.7553    11.0936        1.27253   0.61227             3.14961
# apple       manyview_dev_0  25.3796    15.8355        0.8537    0.745961            4.46496
# apple       manyview_dev_1  21.1       14.771         0.992911  0.738451            6.90396
# MEAN        -               22.0196    15.1295        0.837643  0.735352            6.11271