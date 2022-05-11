# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass, field, fields
from distutils.version import StrictVersion
import functools
import gzip
import hashlib
from itertools import islice
import os
import random
from typing import List, Optional, Union, Tuple
import warnings
import json

import numpy as np
from PIL import Image
from plyfile import PlyData  # TODO: get rid of it after pt3d supports loading normals
import torch

import pytorch3d
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures.pointclouds import Pointclouds
from tools.camera_utils import assert_pytorch3d_has_new_ndc_convention

from . import types


@dataclass
class FrameData:
    """
    A type of the elements returned by indexing the Co3dDataset object.
    It can represent both individual frames and batches of thereof;
    in this documentation, the sizes of tensors refer to single frames;
    add the first batch dimension for the collation result.

    Args:
        frame_number: The number of the frame within its sequence.
            0-based continuous integers.
        frame_timestamp: The time elapsed since the start of a sequence in sec.
        sequence_name: The unique name of the frame's sequence.
        sequence_category: The object category of the sequence.
        image_size_hw: The size of the image in pixels; (height, width) tuple.
        image_path: The qualified path to the loaded image (with dataset_root).
        image_rgb: A Tensor of shape `(3, H, W)` holding the RGB image
            of the frame; elements are floats in [0, 1].
        mask_crop: A binary mask of shape `(1, H, W)` denoting the valid image
            regions. Regions can be invalid (mask_crop[i,j]=0) in case they
            are a result of zero-padding of the image after cropping around
            the object bounding box; elements are floats in {0.0, 1.0}.
        depth_path: The qualified path to the frame's depth map.
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        depth_mask: A binary mask of shape `(1, H, W)` denoting pixels of the
            depth map that are valid for evaluation, they have been checked for
            consistency across views; elements are floats in {0.0, 1.0}.
        mask_path: A qualified path to the foreground probability mask.
        fg_probability: A Tensor of `(1, H, W)` denoting the probability of the
            pixels belonging to the captured object; elements are floats
            in [0, 1].
        bbox_xywh: The bounding box capturing the object in the
            format (x0, y0, width, height).
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
        camera_quality_score: The score proportional to the confidence of the
            frame's camera estimation (the higher the more accurate).
        point_cloud_quality_score: The score proportional to the accuracy of the
            frame's sequence point cloud (the higher the more accurate).
        sequence_point_cloud_path: The path to the sequence's point cloud.
        sequence_point_cloud: A PyTorch3D Pointclouds object holding the
            point cloud corresponding to the frame's sequence. When the object
            represents a batch of frames, point clouds may be deduplicated;
            see `sequence_point_cloud_idx`.
        sequence_point_cloud_idx: Integer indices mapping frame indices to the
            corresponding point clouds in `sequence_point_cloud`; to get the
            corresponding point cloud to `image_rgb[i]`, use
            `sequence_point_cloud[sequence_point_cloud_idx[i]]`.
        frame_type: The type of the loaded frame specified in
            `subset_lists_file`, if provided.
        meta: A dict for storing additional frame information.
    """

    frame_number: int
    frame_timestamp: float
    sequence_name: Union[str, List[str]]
    sequence_category: Union[str, List[str]]
    image_size_hw: torch.Tensor
    image_path: Union[str, List[str], None] = None
    image_rgb: Optional[torch.Tensor] = None
    # masks out padding added due to cropping the square bit
    mask_crop: Optional[torch.Tensor] = None
    depth_path: Union[str, List[str], None] = None
    depth_map: Optional[torch.Tensor] = None
    depth_mask: Optional[torch.Tensor] = None
    mask_path: Union[str, List[str], None] = None
    fg_probability: Optional[torch.Tensor] = None
    bbox_xywh: Optional[torch.Tensor] = None
    camera: Optional[PerspectiveCameras] = None
    camera_quality_score: Optional[torch.Tensor] = None
    point_cloud_quality_score: Optional[torch.Tensor] = None
    sequence_point_cloud_path: Union[str, List[str], None] = None
    sequence_point_cloud: Optional[Pointclouds] = None
    sequence_point_cloud_idx: Optional[torch.Tensor] = None
    frame_type: Union[str, List[str], None] = None  # seen | unseen
    meta: dict = field(default_factory=lambda: {})

    def to(self, *args, **kwargs):
        new_params = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (torch.Tensor, Pointclouds, CamerasBase)):
                new_params[f.name] = value.to(*args, **kwargs)
            else:
                new_params[f.name] = value
        return type(self)(**new_params)

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    # the following functions make sure **frame_data can be passed to functions
    def keys(self):
        for f in fields(self):
            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            pointcloud_ids = [id(el.sequence_point_cloud) for el in batch]
            id_to_idx = defaultdict(list)
            for i, pc_id in enumerate(pointcloud_ids):
                id_to_idx[pc_id].append(i)

            sequence_point_cloud = []
            sequence_point_cloud_idx = -np.ones((len(batch),))
            for i, ind in enumerate(id_to_idx.values()):
                sequence_point_cloud_idx[ind] = i
                sequence_point_cloud.append(batch[ind[0]].sequence_point_cloud)
            assert (sequence_point_cloud_idx >= 0).all()

            override_fields = {
                "sequence_point_cloud": sequence_point_cloud,
                "sequence_point_cloud_idx": sequence_point_cloud_idx.tolist(),
            }
            # note that the pre-collate value of sequence_point_cloud_idx is unused

            collated = {}
            for f in fields(elem):
                list_values = override_fields.get(
                    f.name, [getattr(d, f.name) for d in batch]
                )
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(l is not None for l in list_values)
                    else None
                )
            return cls(**collated)
        elif isinstance(elem, Pointclouds):
            # TODO: use concatenation
            pointclouds = type(elem)(
                points=[p.points_padded()[0] for p in batch],
                normals=[p.normals_padded()[0] for p in batch],
                features=[p.features_padded()[0] for p in batch],
            )
            return pointclouds
        elif isinstance(elem, CamerasBase):
            # TODO: make a function for it
            # TODO: don't store K; enforce working in NDC space
            return type(elem)(
                R=torch.cat([c.R for c in batch], dim=0),
                T=torch.cat([c.T for c in batch], dim=0),
                K=torch.cat([c.K for c in batch], dim=0)
                if elem.K is not None
                else None,
                focal_length=torch.cat([c.focal_length for c in batch], dim=0),
                principal_point=torch.cat([c.principal_point for c in batch], dim=0),
            )
        else:
            return torch.utils.data._utils.collate.default_collate(batch)


@dataclass(eq=False)
class Co3dDataset(torch.utils.data.Dataset):
    """
    A class for the Common Objects in 3D (CO3D) dataset.

    Args:
        frame_annotations_file: A zipped json file containing metadata of the
            frames in the dataset, serialised List[types.FrameAnnotation].
        sequence_annotations_file: A zipped json file containing metadata of the
            sequences in the dataset, serialised List[types.SequenceAnnotation].
        subset_lists_file: A json file containing the lists of frames corresponding
            corresponding to different subsets (e.g. train/val/test) of the dataset;
            format: {subset: (sequence_name, frame_id, file_path)}.
        subsets: Restrict frames/sequences only to the given list of subsets
            as defined in subset_lists_file (see above).
        limit_to: Limit the dataset to the first #limit_to frames (after other
            filters have been applied).
        limit_sequences_to: Limit the dataset to the first
            #limit_sequences_to sequences (after other sequence filters have been
            applied but before frame-based filters).
        pick_sequence: A list of sequence names to restrict the dataset to.
        exclude_sequence: A list of the names of the sequences to exclude.
        limit_category_to: Restrict the dataset to the given list of categories.
        dataset_root: The root folder of the dataset; all the paths in jsons are
            specified relative to this root (but not json paths themselves).
        load_images: Enable loading the frame RGB data.
        load_depths: Enable loading the frame depth maps.
        load_depth_masks: Enable loading the frame depth map masks denoting the
            depth values used for evaluation (the points consistent across views).
        load_masks: Enable loading frame foreground masks.
        load_point_clouds: Enable loading sequence-level point clouds.
        max_points: Cap on the number of loaded points in the point cloud;
            if reached, they are randomly sampled without replacement.
        mask_images: Whether to mask the images with the loaded foreground masks;
            0 value is used for background.
        mask_depths: Whether to mask the depth maps with the loaded foreground
            masks; 0 value is used for background.
        image_height: The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        image_width: The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        box_crop: Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected.
        box_crop_mask_thr: The threshold used to separate pixels into foreground
            and background based on the foreground_probability mask; if no value
            is greater than this threshold, the loader lowers it and repeats.
        box_crop_context: The amount of additional padding added to each
            dimention of the cropping bounding box, relative to vox size.
        remove_empty_masks: Removes the frames with no active foreground pixels
            in the segmentation mask after thresholding (see box_crop_mask_thr).
        n_frames_per_sequence: If > 0, randomly samples #n_frames_per_sequence
            frames in each sequences uniformly without replacement if it has
            more frames than that; applied before other frame-level filters.
        seed: The seed of the random generator sampling #n_frames_per_sequence
            random frames per sequence.
        eval_batches: A list of batches that form the evaluation set;
            list of batch-sized lists of indices corresponding to __getitem__
            of this class, thus it can be used directly as a batch sampler.
    """

    frame_annotations_file: str = ""
    sequence_annotations_file: str = ""
    subset_lists_file: str = ""
    subsets: Optional[List[str]] = None
    limit_to: int = 0
    limit_sequences_to: int = 0
    pick_sequence: List[str] = field(default_factory=lambda: [])
    exclude_sequence: List[str] = field(default_factory=lambda: [])
    limit_category_to: List[int] = field(default_factory=lambda: [])
    dataset_root: str = ""
    load_images: bool = True
    load_depths: bool = True
    load_depth_masks: bool = True
    load_masks: bool = True
    load_point_clouds: bool = False
    max_points: int = 0
    mask_images: bool = False
    mask_depths: bool = False
    image_height: int = 256
    image_width: int = 256
    box_crop: bool = False
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 1.0
    remove_empty_masks: bool = False
    n_frames_per_sequence: int = -1
    seed: int = 0
    eval_batches: Optional[List[List[int]]] = None

    def __post_init__(self):
        assert_pytorch3d_has_new_ndc_convention()
        self.subset_to_image_path = None
        self._load_frames()
        self._load_sequences()
        self._load_subset_lists()
        self._filter_db()  # also computes sequence indices
        print(str(self))

    def seq_frame_index_to_dataset_index(
        self,
        seq_frame_index: Union[
            List[List[Union[Tuple[str, int, str], Tuple[str, int]]]],
        ],
    ):
        """
        Obtain indices into the dataset object given a list of frames specified as
        `seq_frame_index = List[List[Tuple[sequence_name:str, frame_number:int]]]`.
        """
        _dataset_seq_frame_n_index = {
            seq: {
                self.frame_annots[idx]["frame_annotation"].frame_number: idx
                for idx in seq_idx
            }
            for seq, seq_idx in self.seq_to_idx.items()
        }

        def _get_batch_idx(seq_name, frame_no, path=None):
            idx = _dataset_seq_frame_n_index[seq_name][frame_no]
            if path is not None:
                # Check that the loaded frame path is consistent
                # with the one stored in self.frame_annots.
                assert os.path.normpath(
                    self.frame_annots[idx]["frame_annotation"].image.path
                ) == os.path.normpath(
                    path
                ), f"Inconsistent batch {seq_name, frame_no, path}."
            return idx

        batches_idx = [[_get_batch_idx(*b) for b in batch] for batch in seq_frame_index]
        return batches_idx

    def __str__(self):
        return f"CO3D Dataset #frames={len(self.frame_annots)}"

    def __len__(self):
        return len(self.frame_annots)

    def _get_frame_type(self, entry):
        return entry["subset"]

    def __getitem__(self, index):
        assert index < len(
            self.frame_annots
        ), f"index {index} out of range {len(self.frame_annots)}"

        entry = self.frame_annots[index]["frame_annotation"]
        point_cloud = self.seq_annots[entry.sequence_name].point_cloud
        frame_data = FrameData(
            frame_number=_safe_as_tensor(entry.frame_number, torch.long),
            frame_timestamp=_safe_as_tensor(entry.frame_timestamp, torch.float),
            sequence_name=entry.sequence_name,
            sequence_category=self.seq_annots[entry.sequence_name].category,
            # original image size
            image_size_hw=_safe_as_tensor(entry.image.size, torch.long),
            camera_quality_score=_safe_as_tensor(
                self.seq_annots[entry.sequence_name].viewpoint_quality_score,
                torch.float,
            ),
            point_cloud_quality_score=_safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
        )

        # The rest of the fields are optional
        frame_data.frame_type = self._get_frame_type(self.frame_annots[index])

        (
            frame_data.fg_probability,
            frame_data.mask_path,
            frame_data.bbox_xywh,
            clamp_bbox_xyxy,
        ) = self._load_crop_fg_probability(entry)

        scale = 1.0
        if self.load_images:
            (
                frame_data.image_rgb,
                frame_data.image_path,
                frame_data.mask_crop,
                scale,
            ) = self._load_crop_images(
                entry, frame_data.fg_probability, clamp_bbox_xyxy
            )

        if self.load_depths and entry.depth is not None:
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(entry, clamp_bbox_xyxy, frame_data.fg_probability)

        frame_data.camera = self._get_pytorch3d_camera(
            entry,
            scale,
            clamp_bbox_xyxy,
        )

        if point_cloud is not None:
            frame_data.sequence_point_cloud_path = os.path.join(
                self.dataset_root, point_cloud.path
            )
            frame_data.sequence_point_cloud = (
                _load_pointcloud(
                    frame_data.sequence_point_cloud_path,
                    max_points=self.max_points,
                )
                if os.path.isfile(frame_data.sequence_point_cloud_path)
                else None
            )

        return frame_data

    def _load_crop_fg_probability(self, entry):
        fg_probability = None
        full_path = None
        bbox_xywh = None
        clamp_bbox_xyxy = None
        if (self.load_masks or self.box_crop) and entry.mask is not None:
            full_path = os.path.join(self.dataset_root, entry.mask.path)
            mask = _load_mask(full_path)

            if mask.shape[-2:] != entry.image.size:
                raise ValueError(
                    f"bad mask size: {mask.shape[-2:]} vs {entry.image.size}!"
                )

            bbox_xywh = torch.tensor(_get_bbox_from_mask(mask, self.box_crop_mask_thr))

            if self.box_crop:
                clamp_bbox_xyxy = _get_clamp_bbox(bbox_xywh, self.box_crop_context)
                mask = _crop_around_box(mask, clamp_bbox_xyxy, full_path)

            fg_probability, _, _ = self._resize_image(mask, mode="nearest")
        return fg_probability, full_path, bbox_xywh, clamp_bbox_xyxy

    def _load_crop_images(self, entry, fg_probability, clamp_bbox_xyxy):
        assert self.dataset_root is not None and entry.image is not None
        path = os.path.join(self.dataset_root, entry.image.path)
        image_rgb = _load_image(path)

        if image_rgb.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!"
            )

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            image_rgb = _crop_around_box(image_rgb, clamp_bbox_xyxy, path)

        image_rgb, scale, mask_crop = self._resize_image(image_rgb)

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb, path, mask_crop, scale

    def _load_mask_depth(self, entry, clamp_bbox_xyxy, fg_probability):
        path = os.path.join(self.dataset_root, entry.depth.path)
        depth_map = _load_depth(path, entry.depth.scale_adjustment)

        if self.box_crop:
            depth_bbox_xyxy = _rescale_bbox(
                clamp_bbox_xyxy, entry.image.size, depth_map.shape[-2:]
            )
            depth_map = _crop_around_box(depth_map, depth_bbox_xyxy, path)

        depth_map, _, _ = self._resize_image(depth_map, mode="nearest")

        if self.mask_depths:
            assert fg_probability is not None
            depth_map *= fg_probability

        if self.load_depth_masks:
            assert entry.depth.mask_path is not None
            mask_path = os.path.join(self.dataset_root, entry.depth.mask_path)
            depth_mask = _load_depth_mask(mask_path)

            if self.box_crop:
                depth_mask_bbox_xyxy = _rescale_bbox(
                    clamp_bbox_xyxy, entry.image.size, depth_mask.shape[-2:]
                )
                depth_mask = _crop_around_box(
                    depth_mask, depth_mask_bbox_xyxy, mask_path
                )

            depth_mask, _, _ = self._resize_image(depth_mask, mode="nearest")
        else:
            depth_mask = torch.ones_like(depth_map)

        return depth_map, path, depth_mask

    def _get_pytorch3d_camera(self, entry, scale, clamp_bbox_xyxy):
        # principal point and focal length
        principal_point = torch.tensor(
            entry.viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(
            entry.viewpoint.focal_length, dtype=torch.float
        )

        # first, we convert from the legacy Pytorch3D NDC convention
        # (currently used in CO3D for storing intrinsics) to pixels
        half_image_size_wh_orig = (
            torch.tensor(list(reversed(entry.image.size)), dtype=torch.float)
            / 2.0
        )

        # principal point and focal length in pixels
        principal_point_px = (
            -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
        )
        focal_length_px = focal_length * half_image_size_wh_orig
        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            principal_point_px -= clamp_bbox_xyxy[:2]

        # now, convert from pixels to Pytorch3D v0.5+ NDC convention
        if self.image_height is None or self.image_width is None:
            out_size = list(reversed(entry.image.size))
        else:
            out_size = [self.image_width, self.image_height]

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry.viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry.viewpoint.T, dtype=torch.float)[None],
        )


    def _load_frames(self):
        print(f"Loading Co3D frames from {self.frame_annotations_file}.")
        with gzip.open(self.frame_annotations_file, "rt", encoding="utf8") as zipfile:
            frame_annots_list = types.load_dataclass(
                zipfile, List[types.FrameAnnotation]
            )
        if not frame_annots_list:
            raise ValueError("Empty dataset!")
        self.frame_annots = [
            {"frame_annotation": a, "subset": None} for a in frame_annots_list
        ]

    def _load_sequences(self):
        print(f"Loading Co3D sequences from {self.sequence_annotations_file}.")
        with gzip.open(
            self.sequence_annotations_file, "rt", encoding="utf8"
        ) as zipfile:
            seq_annots = types.load_dataclass(zipfile, List[types.SequenceAnnotation])
        if not seq_annots:
            raise ValueError("Empty sequences file!")
        self.seq_annots = {entry.sequence_name: entry for entry in seq_annots}

    def _load_subset_lists(self):
        print(f"Loading Co3D subset lists from {self.subset_lists_file}.")
        if not self.subset_lists_file:
            return

        with open(self.subset_lists_file, "r") as f:
            subset_to_seq_frame = json.load(f)

        frame_path_to_subset = {
            path: subset
            for subset, frames in subset_to_seq_frame.items()
            for _, _, path in frames
        }

        for frame in self.frame_annots:
            frame["subset"] = frame_path_to_subset.get(
                frame["frame_annotation"].image.path, None
            )
            if frame["subset"] is None:
                warnings.warn(
                    "Subset lists are given but don't include "
                    + frame["frame_annotation"].image.path
                )

    def _filter_db(self):
        if self.remove_empty_masks:
            print("Removing images with empty masks.")
            old_len = len(self.frame_annots)
            self.frame_annots = [
                frame
                for frame in self.frame_annots
                if frame["frame_annotation"].mask is not None
                and frame["frame_annotation"].mask.mass > 1
            ]
            print("... filtered %d -> %d" % (old_len, len(self.frame_annots)))

        # this has to be called after joining with categories!!
        if self.subsets:
            if not self.subset_lists_file:
                raise ValueError(
                    "Subset filter is on but subset_lists_file was not given"
                )

            print(f"Limitting Co3D dataset to the '{self.subsets}' subsets.")
            # truncate the list of subsets to the valid one
            self.frame_annots = [
                entry for entry in self.frame_annots if entry["subset"] in self.subsets
            ]
            if len(self.frame_annots) == 0:
                raise ValueError(
                    f"There are no frames in the '{self.subsets}' subsets!"
                )

            self._invalidate_indexes(filter_seq_annots=True)

        if len(self.limit_category_to) > 0:
            print(f"Limitting dataset to categories: {self.limit_category_to}")
            self.seq_annots = {
                name: entry
                for name, entry in self.seq_annots.values()
                if entry.category in self.limit_category_to
            }

        # sequence filters
        for prefix in ("pick", "exclude"):
            orig_len = len(self.seq_annots)
            attr = f"{prefix}_sequence"
            arr = getattr(self, attr)
            if len(arr) > 0:
                print(f"{attr}: {str(arr)}")
                cond = lambda name, exclude=False: (name in arr) != exclude
                self.seq_annots = {
                    name: entry
                    for name, entry in self.seq_annots.items()
                    if cond(name, exclude=prefix == "exclude")
                }
                print("... filtered %d -> %d" % (orig_len, len(self.seq_annots)))

        if self.limit_sequences_to > 0:
            self.seq_annots = dict(
                islice(self.seq_annots.items(), self.limit_sequences_to)
            )

        # retain only frames from retained sequences
        self.frame_annots = [
            f
            for f in self.frame_annots
            if f["frame_annotation"].sequence_name in self.seq_annots
        ]

        self._invalidate_indexes()

        if self.n_frames_per_sequence > 0:
            print(f"Taking max {self.n_frames_per_sequence} per sequence.")
            keep_idx = []
            for seq, seq_indices in self.seq_to_idx.items():
                # infer the seed from the sequence name, this is reproducible
                # and makes the selection differ for different sequences
                seed = _seq_name_to_seed(seq) + self.seed
                seq_idx_shuffled = random.Random(seed).sample(
                    sorted(seq_indices), len(seq_indices)
                )
                keep_idx.extend(seq_idx_shuffled[: self.n_frames_per_sequence])

            print("... filtered %d -> %d" % (len(self.frame_annots), len(keep_idx)))
            self.frame_annots = [self.frame_annots[i] for i in keep_idx]
            self._invalidate_indexes(filter_seq_annots=False)
            # sequences are not decimated, so self.seq_annots is valid

        if self.limit_to > 0 and self.limit_to < len(self.frame_annots):
            print(
                "limit_to: filtered %d -> %d" % (len(self.frame_annots), self.limit_to)
            )
            self.frame_annots = self.frame_annots[: self.limit_to]
            self._invalidate_indexes(filter_seq_annots=True)

    def _invalidate_indexes(self, filter_seq_annots=False):
        # update seq_to_idx and filter seq_meta according to frame_annots change
        # if filter_seq_annots, also uldates seq_annots based on the changed seq_to_idx
        self._invalidate_seq_to_idx()

        if filter_seq_annots:
            self.seq_annots = {
                k: v for k, v in self.seq_annots.items() if k in self.seq_to_idx
            }

    def _invalidate_seq_to_idx(self):
        seq_to_idx = defaultdict(list)
        for idx, entry in enumerate(self.frame_annots):
            seq_to_idx[entry["frame_annotation"].sequence_name].append(idx)
        self.seq_to_idx = seq_to_idx

    def _resize_image(self, image, mode="bilinear"):
        if self.image_height is None and self.image_width is None:
            # skip the resizing
            imre_ = torch.from_numpy(image)
            return imre_, 1.0, torch.ones_like(imre_[:1])
        # takes numpy array, returns pytorch tensor
        minscale = min(
            self.image_height / image.shape[-2],
            self.image_width / image.shape[-1],
        )
        imre = torch.nn.functional.interpolate(
            torch.from_numpy(image)[None],
            scale_factor=minscale,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
            recompute_scale_factor=True,
        )[0]
        imre_ = torch.zeros(image.shape[0], self.image_height, self.image_width)
        imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
        mask = torch.zeros(1, self.image_height, self.image_width)
        mask[:, 0 : imre.shape[1] - 1, 0 : imre.shape[2] - 1] = 1.0
        return imre_, minscale, mask


def _seq_name_to_seed(seq_name):
    return int(hashlib.sha1(seq_name.encode("utf-8")).hexdigest(), 16)


def _load_image(path):
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im


def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def _load_1bit_png_mask(file: str):
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def _load_depth_mask(path):
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = _load_1bit_png_mask(path)
    return m[None]  # fake feature channel


def _load_depth(path, scale_adjustment):
    if not path.lower().endswith(".jpg.geometric.png"):
        raise ValueError('unsupported depth file name "%s"' % path)

    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


def _load_mask(path):
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)
    mask = mask.astype(np.float32) / 255.0
    return mask[None]  # fake feature channel


def _get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]


def _get_bbox_from_mask(mask, thr, decrease_quant=0.05):
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(f"Empty masks_for_bbox (thr={thr}) => using full image.")

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def _get_clamp_bbox(bbox, box_crop_context=0.0, impath=""):
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        warnings.warn(f"squashed image {impath}!!")
        return None

    bbox[2:] = torch.clamp(bbox[2:], 2)
    bbox[2:] += bbox[0:2] + 1  # convert to [xmin, ymin, xmax, ymax]
    # +1 because upper bound is not inclusive

    return bbox


def _crop_around_box(tensor, bbox, impath=""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox[[0, 2]] = torch.clamp(bbox[[0, 2]], 0.0, tensor.shape[-1])
    bbox[[1, 3]] = torch.clamp(bbox[[1, 3]], 0.0, tensor.shape[-2])
    bbox = bbox.round().long()
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"

    return tensor


def _rescale_bbox(bbox, orig_res, new_res):
    assert bbox is not None
    assert np.prod(orig_res) > 1e-8
    # average ratio of dimensions
    rel_size = (new_res[0] / orig_res[0] + new_res[1] / orig_res[1]) / 2.0
    return bbox * rel_size


def _safe_as_tensor(data, dtype):
    if data is None:
        return None
    return torch.tensor(data, dtype=dtype)


# NOTE this cache is per-worker; they are implemented as processes.
# each batch is loaded and collated by a single worker;
# since sequences tend to co-occur within bathes, this is useful.
@functools.lru_cache(maxsize=256)
def _load_pointcloud(pcl_path, max_points=0):
    with open(pcl_path, "rb") as f:
        plydata = PlyData.read(f)

    pcl_data = torch.stack(
        [
            torch.FloatTensor(np.array(plydata["vertex"][c]).astype(float))
            for c in ("x", "y", "z", "red", "green", "blue", "nx", "ny", "nz")
        ],
        dim=1,
    )

    if pcl_data.shape[0] > max_points > 0:
        prm = torch.randperm(pcl_data.shape[0])[:max_points]
        pcl_data = pcl_data[prm]

    points, points_rgb, normals = pcl_data.split([3, 3, 3], dim=1)

    return Pointclouds(points[None], normals[None], features=points_rgb[None] / 255.0)
