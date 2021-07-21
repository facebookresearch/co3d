# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from dataclasses import dataclass
import json
import typing
import gzip
from typing import Optional, Tuple, IO, Any, Union

TF3 = Tuple[float, float, float]


@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]  # TODO: rename size_hw?


@dataclass
class DepthAnnotation:
    # path to png file, relative w.r.t. dataset_root, storing `depth / scale_adjustment`
    path: str
    # a factor to convert png values to actual depth: `depth = png * scale_adjustment`
    scale_adjustment: float
    # path to png file, relative w.r.t. dataset_root, storing binary `depth` mask
    mask_path: Optional[str]


@dataclass
class MaskAnnotation:
    # path to png file storing (Prob(fg | pixel) * 255)
    path: str
    # (soft) number of pixels in the mask; sum(Prob(fg | pixel))
    mass: Optional[float] = None


@dataclass
class ViewpointAnnotation:
    # In right-multiply (Pytorch3D) format. X_cam = X_world @ R + T
    R: Tuple[TF3, TF3, TF3]
    T: TF3

    # TODO: not sure if those should be optional. If we have been able to
    # estimate (R, t), we must have idea of the intrinsic as well.
    # Also probaby safe to assume perspective camera
    # in pixels (i.e. not NDC)
    focal_length: Tuple[float, float]
    # in pixels (i.e. not NDC)
    principal_point: Tuple[float, float]


@dataclass
class FrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, contunuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    depth: Optional[DepthAnnotation] = None
    mask: Optional[MaskAnnotation] = None
    viewpoint: Optional[ViewpointAnnotation] = None


@dataclass
class PointCloudAnnotation:
    # path to ply file with points only, relative w.r.t. dataset_root
    path: str
    # the bigger the better
    quality_score: float
    n_points: Optional[int]


@dataclass
class VideoAnnotation:
    # path to the original video file, relative w.r.t. dataset_root
    path: str
    # length of the video in seconds
    length: float


@dataclass
class SequenceAnnotation:
    sequence_name: str
    category: str
    video: Optional[VideoAnnotation] = None
    point_cloud: Optional[PointCloudAnnotation] = None
    # the bigger the better
    viewpoint_quality_score: Optional[float] = None


def dump_dataclass(obj: Any, f: Union[IO, str], binary: bool = False):
    """
    Args:
        f: Either a path to a file, or a file opened for writing.
        obj: A @dataclass or collection hiererchy including dataclasses.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        f.write(json.dumps(_asdict_rec(obj)).encode("utf8"))
    else:
        json.dump(_asdict_rec(obj), f)


def load_dataclass(f: Union[IO, str], cls: Any, binary: bool = False):
    """
    Loads to a @dataclass or collection hiererchy including dataclasses
    from a json recursively.
    Call it like load_dataclass(f, typing.List[FrameAnnotationAnnotation]).
    raises KeyError if json has keys not mapping to the dataclass fields.

    Args:
        f: Either a path to a file, or a file opened for writing.
        cls: The class of the loaded dataclass.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        asdict = json.loads(f.read().decode("utf8"))
    else:
        asdict = json.load(f)
    return _dataclass_from_dict(asdict, cls)


def _asdict_rec(obj):
    return dataclasses._asdict_inner(obj, dict)


# same as typing.get_origin(cls) added in Python 3.8
def _get_origin(cls):
    return getattr(cls, "__origin__", None)


def _dataclass_from_dict(d, typeannot):
    cls = _get_origin(typeannot) or typeannot
    if d is None:
        return d
    elif issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        types = cls._field_types.values()
        return cls(*[_dataclass_from_dict(v, tp) for v, tp in zip(d, types)])
    elif issubclass(cls, (list, tuple)):
        types = typing.get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(d)
        return cls(_dataclass_from_dict(v, tp) for v, tp in zip(d, types))
    elif issubclass(cls, dict):
        key_t, val_t = typing.get_args(typeannot)
        return cls(
            (_dataclass_from_dict(k, key_t), _dataclass_from_dict(v, val_t))
            for k, v in d.items()
        )
    elif not dataclasses.is_dataclass(typeannot):
        return d

    assert dataclasses.is_dataclass(cls)
    fieldtypes = {f.name: _unwrap_type(f.type) for f in dataclasses.fields(typeannot)}
    return cls(**{k: _dataclass_from_dict(v, fieldtypes[k]) for k, v in d.items()})


def _unwrap_type(tp):
    # strips Optional wrapper, if any
    if _get_origin(tp) is typing.Union:
        args = typing.get_args(tp)
        if len(args) == 2 and any(a is type(None) for a in args):
            # this is typing.Optional
            return args[0] if args[1] is type(None) else args[1]
    return tp


def dump_dataclass_jgzip(outfile: str, obj: Any):
    """
    Dumps obj to a gzipped json outfile.

    Args:
        obj: A @dataclass or collection hiererchy including dataclasses.
        outfile: The path to the output file.
    """
    with gzip.GzipFile(outfile, "wb") as f:
        dump_dataclass(f, obj, binary=True)


def load_dataclass_jgzip(outfile, cls):
    """
    Loads a dataclass from a gzipped json outfile.

    Args:
        outfile: The path to the loaded file.
        cls: The type annotation of the loaded dataclass.

    Returns:
        loaded_dataclass: The loaded dataclass.
    """
    with gzip.GzipFile(outfile, "rb") as f:
        return load_dataclass(f, cls, binary=True)
