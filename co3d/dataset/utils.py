# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import copy
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet


def redact_eval_frame_data(fd: FrameData) -> FrameData:
    """
    Redact all information about the test element (1st image)
    of the evaluation frame data `fd`.

    This is done by zeroing all elements of the relevant tensors in `fd`
    followed by removing the sequence_point_cloud field.
    """
    fd_redacted = copy.deepcopy(fd)
    for redact_field_name in [
        "fg_probability",
        "image_rgb",
        "depth_map",
        "mask_crop",
    ]:
        # zero-out all elements in the redacted tensor
        field_val = getattr(fd, redact_field_name)
        field_val[:1] *= 0
    # also remove the point cloud info
    fd_redacted.sequence_point_cloud_idx = None
    fd_redacted.sequence_point_cloud = None
    return fd_redacted


def _check_valid_eval_frame_data(
    fd: FrameData,
    task: CO3DTask,
    sequence_set: CO3DSequenceSet,
):
    """
    Check that the evaluation batch `fd` is redacted correctly.
    """
    is_redacted = torch.stack(
        [
            getattr(fd, k).abs().sum((1,2,3)) <= 0
            for k in ["image_rgb", "depth_map", "fg_probability"]
        ]
    )
    if sequence_set==CO3DSequenceSet.TEST:
        # first image has to be redacted
        assert is_redacted[:, 0].all()
        # all depth maps have to be redacted
        assert is_redacted[1, :].all()
        # no known views should be redacted
        assert not is_redacted[:, 1:].all(dim=0).any()
    elif sequence_set==CO3DSequenceSet.DEV:
        # nothing should be redacted
        assert not is_redacted.all(dim=0).any()
    else:
        raise ValueError(sequence_set)