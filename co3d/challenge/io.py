# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import logging
import numpy as np
import dbm
import functools
import h5py

from io import BytesIO
from PIL import Image
from typing import Optional, Callable, Dict, Union
from tqdm import tqdm
from .data_types import CO3DSequenceSet, CO3DTask, RGBDAFrame


logger = logging.getLogger(__file__)


def store_rgbda_frame(rgbda_frame: RGBDAFrame, fl: str):
    assert np.isfinite(rgbda_frame.depth).all()
    store_mask(rgbda_frame.mask[0], fl + "_mask.png")
    store_depth(rgbda_frame.depth[0], fl + "_depth.png")
    store_image(rgbda_frame.image, fl + "_image.png")
    if rgbda_frame.depth_mask is not None:
        store_1bit_png_mask(rgbda_frame.depth_mask[0], fl + "depth_mask.png")


def link_rgbda_frame_files(fl_existing: str, fl_src_link: str):
    for pfix in ["_mask.png", "_depth.png", "_image.png", "_depth_mask.png"]:
        link_tgt = fl_existing+pfix
        link_src = fl_src_link+pfix
        if os.path.islink(link_src):
            os.remove(link_src)
        elif os.path.isfile(link_src):
            raise ValueError(f"Link source {link_src} is an actual file (not a link).")
        if not os.path.isfile(link_tgt):
            if pfix=="_depth_mask.png":
                pass
            else:
                raise ValueError(f"Target file {link_tgt} does not exist!")
        else:
            if os.path.islink(link_src):
                os.remove(link_src)
            os.symlink(link_tgt, link_src)
        

def load_rgbda_frame(fl: str, check_for_depth_mask: bool = False) -> RGBDAFrame:
    f = RGBDAFrame(
        mask=load_mask(fl + "_mask.png")[None],
        depth=load_depth(fl + "_depth.png")[None],
        image=load_image(fl + "_image.png"),
    )
    if not np.isfinite(f.depth).all():
        f.depth[~np.isfinite(f.depth)] = 0.0  # chuck the infs in depth
    if check_for_depth_mask:
        depth_mask_path = fl + "_depth_mask.png"
        if os.path.isfile(depth_mask_path):
            f.depth_mask = load_1bit_png_mask(depth_mask_path)[None]
    return f


def store_1bit_png_mask(mask: np.ndarray, fl: str):
    """
    mask: HxW
    """
    Image.fromarray((mask*255).astype('u1'), mode='L').convert('1').save(fl, "PNG")


def load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(_handle_db_file(file)) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def load_mask(fl: str):
    return np.array(Image.open(_handle_db_file(fl))).astype(np.float32) / 255.0


def store_mask(mask: np.ndarray, fl: str, mode: str = "L"):
    """
    mask: HxW
    """
    assert mask.ndim == 2
    if mode == "L":
        mpil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").convert("L")
    elif mode == "I;16":
        mpil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="I;16").convert(
            "I;16"
        )
    else:
        raise ValueError(mode)
    mpil.save(fl, "PNG")


def load_depth(fl: str):
    depth_pil = Image.open(_handle_db_file(fl))
    depth = (
        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((depth_pil.size[1], depth_pil.size[0]))
    )
    assert depth.ndim == 2
    return depth


def store_depth(depth: np.ndarray, fl: str):
    assert depth.ndim == 2
    depth_uint16 = np.frombuffer(depth.astype(np.float16), dtype=np.uint16).reshape(
        depth.shape
    )
    Image.fromarray(depth_uint16).save(fl)


def load_image(fl: str):
    return np.array(Image.open(_handle_db_file(fl))).astype(np.float32).transpose(2, 0, 1) / 255.0


def store_image(image: np.ndarray, fl: str):
    assert image.ndim == 3
    Image.fromarray((image.transpose(1, 2, 0) * 255.0).astype(np.uint8)).save(fl)


def _handle_db_file(fl_or_db_link: str):
    """
    In case `fl_or_db_link` is a symlink pointing at an .hdf5 or .dbm database file,
    this function returns a BytesIO object yielding the underlying file's binary data.

    Otherwise, the function simply returns `fl_or_db_link`.
    """

    fl_or_bytes_io = fl_or_db_link
    for db_format, data_load_fun in (
        (".hdf5", _get_image_data_from_h5),
        (".dbm", _get_image_data_from_dbm),
    ):
        fl_or_bytes_io = _maybe_get_db_image_data_bytes_io_from_file(
            fl_or_db_link,
            db_format,
            data_load_fun,
        )
        if not isinstance(fl_or_bytes_io, str):
            # logger.info(f"{fl} is {db_format}!")
            break
    return fl_or_bytes_io


def _maybe_get_db_image_data_bytes_io_from_file(
    fl_or_db_link: str,
    db_format: str,
    data_load_fun: Callable,
) -> Union[str, BytesIO]:
    """
    In case `fl_or_db_link` is a symlink pointing at a database file `db_path` with
    of type `db_format`, this function calls `data_load_fun(fl_or_db_link, db_path)` 
    to retrieve a BytesIO object yielding the `fl`s binary data.

    Otherwise, the function simply returns `fl_or_db_link`.
    """
    if os.path.islink(fl_or_db_link):
        realpath = os.readlink(fl_or_db_link)
        if not realpath.endswith(db_format):
            return fl_or_db_link
        db_path = fl_or_db_link
    else:
        return fl_or_db_link
    return data_load_fun(realpath, db_path)


@functools.lru_cache(maxsize=1)
def _cached_dbm_open_for_read(dbmpath: str):
    db = dbm.open(dbmpath, "r")
    return db


def _get_image_data_from_dbm(dbmpath: str, fl: str):
    flname = os.path.split(fl)[-1]
    db = _cached_dbm_open_for_read(dbmpath)
    # with dbm.open(dbmpath, "r") as db:
    bin_data = db[flname]
    return BytesIO(bin_data)


def _get_image_data_from_h5(h5path: str, fl: str):
    with h5py.File(h5path, "r") as f:
        flname = os.path.split(fl)[-1]
        file_index = f["binary_data"].attrs
        if flname not in file_index:
            raise IndexError(f"{flname} not in {h5path}!")
        idx = file_index[flname]
        bin_data = f["binary_data"][idx]
    return BytesIO(bin_data)


def get_category_to_subset_name_list(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
):
    """
    Get the mapping from categories to existing subset names.

    Args:
        dataset_root: The dataset root folder.
        task: CO3D Challenge task.
        sequence_set: CO3D Challenge sequence_set.

    Returns:
        category_to_subset_name_list: A dict of the following form:
            {
                category: [subset_name_0, subset_name_1, ...],
                ...
            }
    """

    json_file = os.path.join(dataset_root, "category_to_subset_name_list.json")
    with open(json_file, "r") as f:
        category_to_subset_name_list = json.load(f)

    # filter per-category subset lists by the selected task
    if task is not None:
        category_to_subset_name_list = {
            category: [
                subset_name
                for subset_name in subset_name_list
                if subset_name.startswith(task.value)
            ]
            for category, subset_name_list in category_to_subset_name_list.items()
        }

    # filter per-category subset lists by the selected sequence set
    if sequence_set is not None:
        category_to_subset_name_list = {
            category: [
                subset_name
                for subset_name in subset_name_list
                if f"_{sequence_set.value}" in subset_name
            ]
            for category, subset_name_list in category_to_subset_name_list.items()
        }

    # remove the categories with completely empty subset_name_lists
    category_to_subset_name_list = {
        c: l for c, l in category_to_subset_name_list.items() if len(l) > 0
    }

    # sort by category
    category_to_subset_name_list = dict(sorted(category_to_subset_name_list.items()))

    return category_to_subset_name_list


def load_all_eval_batches(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
    remove_frame_paths: bool = False,
    only_target_frame: bool = True,
):
    """
    Load eval batches files stored in dataset_root into a dictionary:
    {
        (category, subset_name): eval_batches_index,
        ...
    }

    Args:
        dataset_root: The root of the CO3DV2 dataset.
        task: CO3D challenge task.
        sequence_set: CO3D challenge sequence set.
        remove_frame_paths: If `True`, removes the paths to frames from the loaded
            dataset index.
        only_target_frame: Loads only the first (evaluation) frame from each eval batch.

    Returns:
        eval_batches_dict: Output dictionary.
    """

    category_to_subset_name_list = get_category_to_subset_name_list(
        dataset_root,
        task=task,
        sequence_set=sequence_set,
    )

    eval_batches_dict = {}
    for category, subset_name_list in category_to_subset_name_list.items():
        for subset_name in subset_name_list:
            # load the subset eval batches
            eval_batches_dict[(category, subset_name)] = _load_eval_batches_file(
                dataset_root,
                category,
                subset_name,
                remove_frame_paths=remove_frame_paths,
                only_target_frame=only_target_frame,
            )
    return eval_batches_dict


def _load_eval_batches_file(
    dataset_root: str,
    category: str,
    subset_name: str,
    remove_frame_paths: bool = True,
    only_target_frame: bool = True,
):
    eval_batches_fl = os.path.join(
        dataset_root,
        category,
        "eval_batches",
        f"eval_batches_{subset_name}.json",
    )
    with open(eval_batches_fl, "r") as f:
        eval_batches = json.load(f)
    
    if only_target_frame:
        eval_batches = [
            b[0] for b in eval_batches
        ]  # take only the first (target evaluation) frame
    
    if remove_frame_paths:
        eval_batches = [b[:2] for b in eval_batches]
    return eval_batches


def export_result_file_dict_to_hdf5(h5path: str, filedict: Dict[str, str]):
    """
    Export the result files to an hdf5 file that will be sent to the EvalAI server:

    Args:
        h5path: Target hdf5 file path.
        filedict: Dict in form {relative_file_path: absolute_file_path}
    """
    logger.info(f"Exporting {len(filedict)} files to HDF5 file {h5path}.")
    if len(filedict)==0:
        raise ValueError("No data to export!")
    assert h5path.endswith(".hdf5")
    if os.path.isfile(h5path):
        os.remove(h5path)
    os.makedirs(os.path.dirname(h5path), exist_ok=True)
    with h5py.File(h5path, "w", libver='latest') as fh5:
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        max_path_len = max(len(p) for p in filedict.keys())        
        dset = fh5.create_dataset(
            'binary_data', (len(filedict), ), dtype=dt, compression="gzip"
        )
        filepath_dset = fh5.create_dataset(
            'filepaths',
            (len(filedict), ), 
            dtype=h5py.string_dtype('utf-8', max_path_len),
            # dtype=np.dtype(f'U{max_path_len}'), 
            compression="gzip"
        )
        index = {}
        for idx, (rel_path, store_file) in enumerate(tqdm(filedict.items(), total=len(filedict))):
            _store_binary_file_data_to_hd5_dataset(dset, store_file, idx)
            flname = os.path.split(rel_path)[-1]
            assert flname not in index, "Duplicate filenames!"
            index[flname] = idx
            filepath_dset[idx] = rel_path
        logger.info(f"Updating index of {h5path}.")
        dset.attrs.update(index)


def make_hdf5_file_links(h5path: str, root: str):
    """
    Link all files whose binary data are stored in an HDF5 file `h5path` to
    files under the root folder.

    Args:
        h5path: HDF5 file.
        root: The root folder for exporting symlinks.
    """
    logger.info(f"Making file links in {root} to DB data in {h5path}.")
    assert h5path.endswith(".hdf5")
    with h5py.File(h5path, "r") as fh5:
        filepaths = [f.decode("UTF-8") for f in np.array(fh5["filepaths"])]
        file_name_to_tgt_file = {
            os.path.split(p)[-1]: os.path.join(root, p) for p in filepaths
        }
        dset = fh5["binary_data"]
        index = dset.attrs
        all_dirs = set(os.path.dirname(p) for p in file_name_to_tgt_file.values())
        for dir_ in all_dirs:
            os.makedirs(dir_, exist_ok=True)
        for flname, _ in tqdm(index.items(), total=len(index)):
            tgt_file = file_name_to_tgt_file[flname]
            link_file_to_db_file(h5path, tgt_file)


def link_file_to_db_file(db_file: str, file: str, overwrite: bool = True):
    """
    Make a symlink file->db_file
    """
    if db_file.endswith(".hdf5"):
        token = "__HDF5__:"
    elif db_file.endswith(".dbm"):
        token = "__DBM__:"
    else:
        raise ValueError(db_file)
    if overwrite and (os.path.isfile(file) or os.path.islink(file)):
        os.remove(file)
    os.symlink(db_file, file)
    
    # symlinks are cleaner ... do not use this anymore:
    # with open(file, "w") as f:
    #     f.write(token+os.path.normpath(os.path.abspath(db_file)))


def _store_binary_file_data_to_hd5_dataset(dset, fl: str, idx: int):
    with open(fl, "rb") as fin:
        binary_data = fin.read()
    dset[idx] = np.fromstring(binary_data, dtype='uint8')