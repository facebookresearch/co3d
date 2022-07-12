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
# DATASET_ROOT_HIDDEN = os.path.join(DATASET_ROOT, "_hidden", "hidden")
# ON_SERVER = False

DATASET_ROOT_HIDDEN = os.path.join("/large_experiments/p3/replay/datasets/co3d/co3d45k_220512/export_v20", "_hidden.hdf5")
ON_SERVER = True


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
    # submission_reeval = CO3DSubmission(
    #     task=task,
    #     sequence_set=sequence_set,
    #     output_folder=os.path.join(submission_output_folder, "_reeval"),
    #     dataset_root=DATASET_ROOT,
    #     on_server=True,
    #     server_data_folder=DATASET_ROOT_HIDDEN,
    # )
    # submission_reeval.evaluate_zip_file(submission.submission_archive, num_workers=num_eval_workers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # iterate over all tasks and sequence sets
    for sequence_set in [CO3DSequenceSet.DEV, CO3DSequenceSet.TEST]:
        for task in [CO3DTask.MANY_VIEW, CO3DTask.FEW_VIEW]:
            make_dbir_submission(task=task, sequence_set=sequence_set)


# CO3D challenge results for DBIR
# Category    Subset name        psnr    psnr_fg    depth_abs_fg       iou    psnr_full_image
# ----------  --------------  -------  ---------  --------------  --------  -----------------
# vase        manyview_dev_0  24.9622    17.2103        0.737651  0.802907            3.49721
# vase        manyview_dev_1  25.3102    18.44          0.59692   0.809792            6.40918
# toytruck    manyview_dev_0  23.7732    15.5102        0.861064  0.741494            4.99226
# toytruck    manyview_dev_1  20.2278    13.6948        0.690828  0.754487            7.87978
# toytrain    manyview_dev_0  21.8014    15.9132        0.660723  0.755107            7.98991
# toytrain    manyview_dev_1  25.8508    18.4285        0.951778  0.646153            9.20359
# toaster     manyview_dev_0  21.7936    13.891         0.648372  0.694767            6.15618
# toaster     manyview_dev_1  18.477     12.1587        0.825844  0.703912            6.79456
# teddybear   manyview_dev_0  23.7997    16.5381        0.46086   0.837897            5.63064
# teddybear   manyview_dev_1  26.1378    19.8502        0.882306  0.756235            5.56647                                                                                                        
# suitcase    manyview_dev_0  24.2254    17.5158        1.13533   0.809919            6.65941
# suitcase    manyview_dev_1  27.9944    18.7613        0.83145   0.759948            5.5394
# skateboard  manyview_dev_0  21.6529    15.3298        0.499439  0.814291            8.77353
# skateboard  manyview_dev_1  25.9824    19.8798        0.624538  0.844691            7.25049
# remote      manyview_dev_0  19.1542    10.9486        0.526491  0.781712            6.8934                                                                                                         
# remote      manyview_dev_1  20.2204    13.3           0.583544  0.763237            4.28249
# plant       manyview_dev_0  21.5417    16.5396        0.684964  0.711299            7.32539
# plant       manyview_dev_1  23.2408    17.1425        0.970093  0.708852            9.37529
# orange      manyview_dev_0  22.3462    13.9155        1.38631   0.778507            6.29187
# orange      manyview_dev_1  21.1582    11.9294        1.08968   0.673347            6.25896                                                                                                        
# mouse       manyview_dev_0  23.7059    15.0934        0.786687  0.745071            6.09387
# mouse       manyview_dev_1  23.4661    15.864         0.611     0.701353            7.37148
# hydrant     manyview_dev_0  23.3411    17.3916        0.516439  0.849561            5.859
# hydrant     manyview_dev_1  23.2579    16.2956        0.526429  0.836342            5.04959
# donut       manyview_dev_0  27.1901    17.9711        0.793923  0.775221            4.05381                                                                                                        
# donut       manyview_dev_1  29.7891    18.2375        0.624767  0.795361            4.27906
# cake        manyview_dev_0  23.1432    14.8656        0.649352  0.789436            6.26039
# cake        manyview_dev_1  26.9312    20.6573        0.719103  0.865847            3.9428
# broccoli    manyview_dev_0  25.5412    16.1942        0.933724  0.698763            6.78183
# broccoli    manyview_dev_1  26.2196    16.1596        0.762279  0.77777             4.81169                                                                                                        
# bowl        manyview_dev_0  25.7917    15.2551        0.452199  0.810974            5.87917
# bowl        manyview_dev_1  19.1636    14.4302        0.872776  0.776362            7.19918
# book        manyview_dev_0  22.3768    14.7519        0.54043   0.807283            6.6576
# book        manyview_dev_1  19.5715    12.1604        0.684169  0.833521            7.1818
# bench       manyview_dev_0  24.2136    19.4035        0.376075  0.831789            4.35312                                                                                                        
# bench       manyview_dev_1  19.781     13.959         0.607774  0.773409            5.0657
# ball        manyview_dev_0  20.7418    12.8557        0.876389  0.678307            5.96079
# ball        manyview_dev_1  18.5385    12.2345        1.15568   0.716587            3.00818
# apple       manyview_dev_0  25.3867    15.8412        0.852304  0.746271            4.46503
# apple       manyview_dev_1  21.9403    15.1499        0.952164  0.756788            6.89821                                                                                                        
# MEAN        -               23.2435    15.7917        0.748546  0.767864            6.09856