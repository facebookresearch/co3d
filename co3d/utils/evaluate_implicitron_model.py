# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation of Implicitron models on CO3Dv2 challenge.
"""


import logging
import os
import torch
import json
import warnings
from typing import Optional, Union, Dict, Tuple
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numpy as np

import pytorch3d
from pytorch3d.implicitron.models.generic_model import ImplicitronRender, GenericModel
from pytorch3d.implicitron.tools.config import get_default_args
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.tools.model_io import (
    parse_epoch_from_model_path,
    find_last_checkpoint,
)
from pytorch3d.implicitron.models.renderer.base import (
    # BaseRenderer,
    EvaluationMode,
    # ImplicitFunctionWrapper,
    # RendererOutput,
    # RenderSamplingMode,
)


from co3d.utils import dbir_utils
from co3d.challenge.co3d_submission import CO3DSubmission
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet
from co3d.challenge.utils import (
    get_co3d_task_from_subset_name,
    get_co3d_sequence_set_from_subset_name,
)
from co3d.dataset.utils import redact_eval_frame_data, _check_valid_eval_frame_data
from co3d.challenge.metric_utils import EVAL_METRIC_NAMES


DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")
DATASET_ROOT_HIDDEN = os.getenv("CO3DV2_HIDDEN_DATASET_ROOT")


# HACK: implicitron_trainer is not part of a package; forcing it in the path
_pytorch3d_root = os.path.dirname(os.path.dirname(pytorch3d.__file__))
implicitron_trainer_dir = os.path.join(_pytorch3d_root, "projects", "implicitron_trainer")
# sys.path.insert(0, implicitron_trainer_dir)
from projects.implicitron_trainer.experiment import Experiment


logger = logging.getLogger(__name__)


def evaluate_implicitron_exp_dir_map(
    category_subset_implicitron_exp_dirs: Union[Dict[Tuple[str, str], str], str],
    task: CO3DTask,
    sequence_set: CO3DSequenceSet,
    submission_output_folder: str,
    num_eval_workers: int = 4,
    submit_to_eval_ai: bool = False,
    skip_evaluation: bool = False,
    fill_results_from_cache: bool = False,
    implicitron_exp_dir_submission_output_subfolder: Optional[str] = None,
):
    """
    Evalulates and submits to EvalAI either:
        1) all Implicitron class-specific models, or
        2) a single model trained for all categories.

    Args:
        category_subset_implicitron_exp_dirs: Two options:
            1) a dict {(category_name, subset_name): implicitron_exp_dir_path} containing
                a mapping from each CO3Dv2 category and subset to the path of the
                corresponding implicitron model exp dir.
            2) a string containing the path to a single model used for reconstructing
                all categories.
        task: The co3d task - either CO3DTask.MANY_VIEW or CO3DTask.FEW_VIEW.
        sequence_set: The sequence set to evaluate on:
            CO3DSequenceSet.DEV for for the development set
            CO3DSequenceSet.TEST for for the test set
        submission_output_folder: Directory containing the submission output files.
        num_eval_workers: Number of processes that conduct evaluation.
        submit_to_eval_ai: If `True`, will automatically submit the exported result
            archive to EvalAI using the CLI interface (needs to be installed with 
            `pip install evalai`). This requires setting the EVAL_AI_PERSONAL_TOKEN 
            environment variable to your personal EVAL_AI token.
        skip_evaluation: Skip the local evaluation.
        implicitron_exp_dir_submission_output_subfolder:
            If set to a string, loads precomputed results from
                ```
                category_subset_implicitron_exp_dirs[(category, subset)]
                /implicitron_exp_dir_submission_output_subfolder
                ```
            for each (category, subset).
            Such precomputed results are typically output by:
                ```
                evaluate_implicitron_exp_dir(
                    category_subset_implicitron_exp_dirs[(category, subset)],
                    ...
                )
    """

    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=submission_output_folder,
        dataset_root=DATASET_ROOT,
    )

    if fill_results_from_cache:
        submission.fill_results_from_cache()

    else:

        if not isinstance(category_subset_implicitron_exp_dirs, str):
            # check that we have all models in case the we were given one model per 
            # category/subset_name
            for category, subset_name in submission.get_eval_batches_map():
                if (category, subset_name) not in category_subset_implicitron_exp_dirs:
                    raise ValueError(
                        f"Missing implicitron exp dir for {category}/{subset_name}."
                    )

        for category, subset_name in submission.get_eval_batches_map():
            if isinstance(category_subset_implicitron_exp_dirs, str):
                # a single model that does it all
                current_implicitron_exp_dir = category_subset_implicitron_exp_dirs
            else:
                # subset-specific models
                current_implicitron_exp_dir = category_subset_implicitron_exp_dirs[
                    (category, subset_name)
                ]

            if implicitron_exp_dir_submission_output_subfolder is not None:
                submission.link_results_from_existing_output_folder(
                    os.path.join(
                        current_implicitron_exp_dir,
                        implicitron_exp_dir_submission_output_subfolder,
                    )
                )

            else:
                update_implicitron_submission_with_category_and_subset_predictions(
                    submission=submission,
                    implicitron_exp_dir=current_implicitron_exp_dir,
                    dataset_root=DATASET_ROOT,
                    category=category,
                    subset_name=subset_name,
                    n_known_frames_for_test=9 if task==CO3DTask.MANY_VIEW else 0,
                )
        
    # Locally evaluate the submission in case we dont evaluate on the hidden test set.
    if sequence_set != CO3DSequenceSet.TEST and not skip_evaluation:
        submission.evaluate(num_workers=num_eval_workers)
    
    if submit_to_eval_ai:
        # Export the submission predictions for submition to the evaluation server.
        # This also validates completeness of the produced predictions.    
        submission.export_results(validate_results=True)
        # submit the results to the EvalAI server.
        submission.submit_to_eval_ai()


def evaluate_implicitron_exp_dir(
    implicitron_exp_dir: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
    subset_name: Optional[str] = None,
    category: Optional[str] = None,
    result_dump_file: Optional[str] = None,
    clear_submission_cache_before_evaluation: bool = False,
    clear_submission_cache_after_evaluation: bool = False,
    submission_output_folder: Optional[str] = None,
    num_eval_workers: int = 4,
):
    """
    Run evaluation for an experiment directory of Implicitron.
    Unless overriden by the user, this function automatically parses the
    category / subset / task / sequence_set / dataset_root
    from the implicitron experiment config stored in implicitron_exp_dir.

    Args:
        implicitron_exp_dir: The directory of an Implicitron experiment.
        task: The co3d task - either CO3DTask.MANY_VIEW or CO3DTask.FEW_VIEW.
        sequence_set: The sequence set to evaluate on:
            CO3DSequenceSet.DEV for for the development set
            CO3DSequenceSet.TEST for for the test set
        subset_name: The name of the CO3Dv2 subset.
            E.g. "manyview_dev_0", "fewview_dev", ...
        category: The name of the CO3Dv2 category to evaluate.
        result_dump_file: Path to the json file with evaluation results.
        clear_submission_cache_before_evaluation: Delete all previous intermediate
            submission files before commencing the current evaluation run.
        clear_submission_cache_after_evaluation: Delete all intermediate
            submission files after the evaluation run.
        submission_output_folder: The path to the folder with intermediate
            submission files.
        num_eval_workers: Number of processes that conduct evaluation.
    """
    
    if result_dump_file is None:
        result_dump_file = os.path.join(
            implicitron_exp_dir, "results_challenge_eval.json"
        )

    cfg = load_implicitron_config_from_exp_dir(implicitron_exp_dir)  

    # assert few config settings    
    assert (
        cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_class_type
        =="JsonIndexDatasetMapProviderV2"
    )

    # read the category / subset / task / sequence_set / dataset_root from
    # the implicitron config
    dataset_provider_args = (
        cfg
        .data_source_ImplicitronDataSource_args
        .dataset_map_provider_JsonIndexDatasetMapProviderV2_args
    )
    if subset_name is None:
        subset_name = dataset_provider_args.subset_name
    if category is None:
        category = dataset_provider_args.category
    if task is None:
        task = get_co3d_task_from_subset_name(subset_name)
    if sequence_set is None:
        sequence_set = get_co3d_sequence_set_from_subset_name(subset_name)
    
    dataset_root = (
        DATASET_ROOT
        if DATASET_ROOT is not None
        else dataset_provider_args.dataset_root
    )

    logger.info(
        f"Evaluating Implicitron model on category {category}; subset {subset_name}"
    )

    # the folder storing all predictions and results of the submission
    if submission_output_folder is None:
        submission_output_folder = get_default_implicitron_exp_dir_submission_output_folder(
            implicitron_exp_dir,
            task,
            sequence_set,
        )
        
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
        raise ValueError(
            f"Cannot evaluate the few-view task in {sequence_set.value} when only the"
            " singlesequence subset of CO3D is present."
        )
        
    if clear_submission_cache_before_evaluation:
        submission.clear_files()    

    # Generate new views for all evaluation examples in category/subset_name.
    update_implicitron_submission_with_category_and_subset_predictions(
        submission=submission,
        implicitron_exp_dir=implicitron_exp_dir,
        dataset_root=dataset_root,
        category=category,
        subset_name=subset_name,
        n_known_frames_for_test=9 if task==CO3DTask.MANY_VIEW else 0,
    )

    # Locally evaluate the submission in case we dont evaluate on the hidden test set.
    if sequence_set == CO3DSequenceSet.TEST:
        logger.warning("Cannot evaluate on the hidden test set. Skipping evaluation.")
        category_subset_results = {m: 0.0 for m in EVAL_METRIC_NAMES}
    else:
        results = submission.evaluate(num_workers=num_eval_workers)
        category_subset_results = results[(category, subset_name)][0]

    # add the eval epoch as well
    category_subset_results["eval_epoch"] = parse_epoch_from_model_path(
        find_last_checkpoint(implicitron_exp_dir)
    )

    logger.info("Implicitron model results:")
    logger.info(f"category={category} / subset_name={subset_name}")
    print_category_subset_results(category_subset_results)

    if clear_submission_cache_after_evaluation:
        submission.clear_files()
        
    logger.info(f"Dumping challenge eval results to {result_dump_file}.")
    with open(result_dump_file, "w") as f:
        json.dump(category_subset_results, f)

    return category_subset_results


@torch.no_grad()
def update_implicitron_submission_with_category_and_subset_predictions(
    submission: CO3DSubmission,
    implicitron_exp_dir: str,
    dataset_root: str,
    category: str,
    subset_name: str,
    num_workers: int = 12,
    n_known_frames_for_test: int = 0,
):
    """
    Updates the CO3DSubmission object `submission` with predictions of a DBIR
    model extracted for a given category, and a dataset subset.

    Args:
        submission: CO3DSubmission object.
        implicitron_exp_dir: Implicitron experiment directory to load the model from.
        dataset_root: Path to the root dataset folder containing CO3Dv2.
        category: A CO3Dv2 category to evaluate.
        subset_name: The name of the evaluation subset of the category.
        num_workers: Number of processes to use for evaluation.
        n_known_frames_for_test: The number of known frames to append to the test batches.
    """

    logger.info(
        "Runing depth-based image rendering (DBIR) new view synthesis "
        f"on category '{category}' subset '{subset_name}'"
    )

    # Get the evaluation device.
    device = torch.device("cuda") if torch.cuda.is_available() else device("cpu")

    # load the implicitron model
    model = load_model_from_implicitron_exp_dir(implicitron_exp_dir)

    # Determine the sequence set and the task we are solving
    sequence_set = submission.sequence_set
    task = submission.task

    # Obtain the CO3Dv2 dataset map
    dataset_map = get_dataset_map(
        dataset_root,
        category,
        subset_name,
        n_known_frames_for_test=n_known_frames_for_test,
    )

    # The test dataloader simply iterates over test_dataset.eval_batches
    # this is done by setting test_dataset.eval_batches as the batch sampler
    test_dataset = dataset_map["test"]
    eval_batches = test_dataset.get_eval_batches()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=eval_batches,
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

        # Redact the frame data so we are sure we cannot use the data
        # from the actual unobserved evaluation sample
        eval_frame_data = redact_eval_frame_data(eval_frame_data)

        # Obtain the image render. In case dataset_test.box_crop==True,
        # we need to paste the render back to the original image bounds.
        model_preds = model(
            **eval_frame_data,
            eval_mode=EvaluationMode.EVALUATION,
        )
        render_crop = model_preds["implicitron_render"]

        # cut the valid part of the render and paste into the original image canvas
        render_full_image = dbir_utils.paste_render_to_original_image(
            eval_frame_data, render_crop
        )

        # get the image, mask, depth as numpy arrays for the challenge submission
        image, mask, depth = [
            getattr(render_full_image, f"{data_type}_render").cpu().numpy()[0]
            for data_type in ["image", "mask", "depth"]
        ]

        # clip the rendered image to [0, 1] range
        image = image.clip(0.0, 1.0)

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


def get_default_implicitron_exp_dir_submission_output_folder(
    implicitron_exp_dir: str,
    task: CO3DTask,
    sequence_set: CO3DSequenceSet,
):
    return os.path.join(
        implicitron_exp_dir,
        f"implicitron_submission_output_{task.value}_{sequence_set.value}",
    )


def parse_co3d_challenge_settings_from_implicitron_exp_dir(
    implicitron_exp_dir: str
) -> Tuple[CO3DSequenceSet, CO3DTask, str, str]:
    """
    Reads the config of an implicitron experiment stored in `implicitron_exp_dir` and
    returns the configuration of the corresponding challenge entry.

    Args:
        implicitron_exp_dir: The directory of an Implicitron experiment.
    Returns:
        sequence_set: CO3D sequence set of the experiment.
        task: The CO3D task of the experiment.
        category: The category of the experiment.
        subset_name: The name of the CO3D subset.
    """

    cfg = load_implicitron_config_from_exp_dir(implicitron_exp_dir)
    dataset_provider_args = (
        cfg
        .data_source_ImplicitronDataSource_args
        .dataset_map_provider_JsonIndexDatasetMapProviderV2_args
    )
    subset_name = dataset_provider_args.subset_name
    category = dataset_provider_args.category
    task = get_co3d_task_from_subset_name(subset_name)
    sequence_set = get_co3d_sequence_set_from_subset_name(subset_name)
    return sequence_set, task, category, subset_name


def load_implicitron_config_from_exp_dir(implicitron_exp_dir: str):
    cfg_filename = os.path.join(implicitron_exp_dir, "expconfig.yaml")
    cfg_load = OmegaConf.load(cfg_filename)
    cfg_default = get_default_args(Experiment)
    cfg = OmegaConf.merge(cfg_default, cfg_load)
    cfg.exp_dir = implicitron_exp_dir
    return cfg


def load_model_from_implicitron_exp_dir(exp_dir: str) -> GenericModel:
    cfg = load_implicitron_config_from_exp_dir(exp_dir)
    experiment = Experiment(**cfg)
    experiment.model_factory.force_resume = True
    model = experiment.model_factory(accelerator=None, exp_dir=exp_dir)
    model.cuda()
    model.eval()
    return model


def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
    n_known_frames_for_test: int = 0,
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
        n_known_frames_for_test=n_known_frames_for_test,
    )
    return dataset_map_provider.get_dataset_map()


def print_category_subset_results(category_subset_results: Dict[str, float]):
    for k, v in category_subset_results.items():
        print(f"{k:20s}: {v:1.3f}")
