import os
import torch
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.models.base_model import ImplicitronRender
from pytorch3d.implicitron.dataset.visualize import get_implicitron_sequence_pointcloud
from pytorch3d.implicitron.utils.point_cloud_utils import render_point_cloud_pytorch3d
from co3d.challenge.co3d_submission import CO3DSubmission
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet
from co3d.challenge.utils import (
    get_co3d_sequence_set_from_subset_name,
    get_co3d_task_from_subset_name,
)
from co3d.dataset.co3d_dataset_v2 import (
    CO3DV2DatasetMapProvider,
    get_available_categories_and_subset_names,
)


def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
):
    task = get_co3d_task_from_subset_name(subset_name)
    dataset_map = CO3DV2DatasetMapProvider(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        task_str=co3d_task_to_task_str(task),
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
    )    
    return dataset_map


def co3d_task_to_task_str(co3d_task: CO3DTask) -> str:
    return {
        CO3DTask.MANY_VIEW: "singlesequence",
        CO3DTask.FEW_VIEW: "multisequence"
    }[co3d_task]  # this is the old co3d naming of the task


def update_dbir_submission_with_category_and_subset(
    submission: CO3DSubmission,
    dataset_root: str,
    category: str,
    subset_name: str,
    num_workers: int = 12,
):
    # Determine the sequence set and the task we are solving
    sequence_set = submission.sequence_set
    task = submission.task

    # Obtain the CO3Dv2 dataset map
    dataset_map = get_dataset_map(dataset_root, category, subset_name)

    # Many-view
    train_dataset = dataset_map["train"]
    # Obtain the colored sequence pointcloud using the depth maps and images
    # in the training set.
    sequence_pointcloud, _ = get_implicitron_sequence_pointcloud(
        train_dataset,
        train_dataset[0].sequence_name,
        mask_points=True,
        max_frames=50,
        num_workers=num_workers,
        load_dataset_point_cloud=True,
    )

    # The test dataloader simply iterates over test_dataset.eval_batches
    # this is done by setting test_dataset.eval_batches as the batch sampler
    test_dataset = dataset_map["test"]
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=1,
        batch_sampler=test_dataset.eval_batches,
        num_workers=0, 
        collate_fn=test_dataset.frame_annotations_type.collate,
    )
    
    # loop over eval examples
    for eval_frame_data in test_dataloader:

        # render the sequence point cloud to each evaluation view
        data_rendered, render_mask, depth_rendered = render_point_cloud_pytorch3d(
            eval_frame_data.camera[[0]],
            sequence_pointcloud,
            render_size=eval_frame_data.image_rgb.shape[-2:],
            point_radius=0.03,
            topk=10,
            eps=1e-2,
            bin_size=0,
        )
        
        render = ImplicitronRender(
            depth_render=depth_rendered,
            image_render=data_rendered,
            mask_render=render_mask,
        )

        # cut the valid part of the render and resize to orig image size
        render_full_image = paste_render_to_original_image(
            eval_frame_data.image_rgb[:1],
            render,
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


def make_dbir_submission():

    task = CO3DTask.MANY_VIEW
    sequence_set = CO3DSequenceSet.DEV
    output_folder = os.path.join(os.path.split(__file__)[0], "dbir_submission_files")
    
    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=output_folder,
        dataset_root=DATASET_ROOT,
    )

    # category_to_subset_names = get_available_categories_and_subset_names(DATASET_ROOT)
    # CO3D_CHALLENGE_CATEGORY_SUBSETS_MANYVIEW_DEV = {cat: subset_list}
    # CO3D_CHALLENGE_CATEGORY_SUBSETS_MANYVIEW_TEST = {cat: subset_list}
    # CO3D_CHALLENGE_CATEGORY_SUBSETS_FEWVIEW_DEV = {cat: subset_list}
    # CO3D_CHALLENGE_CATEGORY_SUBSETS_FEWVIEW_TEST = {cat: subset_list}

    category_to_subset_name_list = get_category_to_subset_name_list(task, sequence_set)

    for category, category_subset_name_list in category_to_subset_name_list.items():
        for subset_name in category_subset_name_list:
            update_dbir_submission_with_category_and_subset(
                submission=submission,
                dataset_root=DATASET_ROOT,
                category=category,
                subset_name=category_subset_name_list,
            )

    submission.evaluate_locally()

    submission.export_results(validate_results=True)


def paste_render_to_original_image(
    frame_data: FrameData,
    render: ImplicitronRender,
) -> ImplicitronRender:
    raise NotImplementedError()