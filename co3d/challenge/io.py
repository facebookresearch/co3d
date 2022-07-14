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
from io import BytesIO
from PIL import Image
from typing import Optional, Callable, Dict
from tqdm import tqdm
from .data_types import CO3DSequenceSet, CO3DTask, RGBDAFrame


logger = logging.getLogger(__file__)


try:
    import h5py
except ImportError:
    logger.debug("No h5py library - make sure not to evaluate on a server.")


def store_rgbda_frame(rgbda_frame: RGBDAFrame, fl: str):
    assert np.isfinite(rgbda_frame.depth).all()
    store_mask(rgbda_frame.mask[0], fl + "_mask.png")
    store_depth(rgbda_frame.depth[0], fl + "_depth.png")
    store_image(rgbda_frame.image, fl + "_image.png")
    if rgbda_frame.depth_mask is not None:
        store_1bit_png_mask(rgbda_frame.depth_mask[0], fl + "depth_mask.png")


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


def _handle_db_file(fl: str):
    for token, data_load_fun in (
        ("__DBM__:", _get_image_data_from_dbm),
        ("__HDF5__:", _get_image_data_from_h5),
    ):
        fl = _maybe_get_db_image_data_bytes_io_from_file(fl, token, data_load_fun)
        if not isinstance(fl, str):
            # logger.info(f"{fl} is {token}!")
            break
    return fl


def _maybe_get_db_image_data_bytes_io_from_file(
    fl: str,
    token: str,
    data_load_fun: Callable,
):
    """
    In case `fl` is a unicode text file starting with `token`, 
    the file `fl` contains a string with a path to a database file that holds the actual
    file binary data.
    
    This function makes sure that either the filepath is returned if `fl`
    does not contain database-file-path, otherwise returns the BytesIO object with
    the `fl`s binary data.
    """

    if os.path.islink(fl):
        realpath = os.readlink(fl)
        if (
            (token=="__HDF5__:" and realpath.endswith(".hdf5"))
            or (token=="__DBM__:" and realpath.endswith(".dbm"))
        ):
            db_path_clean = realpath
        else:
            return fl
    else:
        with open(fl, "rb") as f:
            first_bytes = f.read(len(token))
            try:
                first_bytes_decoded = first_bytes.decode()
            except UnicodeDecodeError:
                return fl
            if first_bytes_decoded != token:
                return fl
        with open(fl, "r") as f:
            db_path = f.readlines()[0]
        assert db_path.startswith(token)
        db_path_clean = db_path[len(token):]
    return data_load_fun(db_path_clean, fl)


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
        idx = f["binary_data"].attrs[flname]
        bin_data = f["binary_data"][idx]
    return BytesIO(bin_data)


def get_category_to_subset_name_list(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
):
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

    return category_to_subset_name_list


def load_all_eval_batches(
    dataset_root: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
    remove_frame_paths: bool = False,
):

    category_to_subset_name_list = get_category_to_subset_name_list(
        dataset_root,
        task=task,
        sequence_set=sequence_set,
    )

    eval_batches = {}
    for category, subset_name_list in category_to_subset_name_list.items():
        for subset_name in subset_name_list:
            # load the subset eval batches
            eval_batches[(category, subset_name)] = _load_eval_batches_file(
                dataset_root,
                category,
                subset_name,
                remove_frame_paths=remove_frame_paths,
            )
    return eval_batches


def _load_eval_batches_file(
    dataset_root: str,
    category: str,
    subset_name: str,
    remove_frame_paths: bool = True,
):
    eval_batches_fl = os.path.join(
        dataset_root,
        category,
        "eval_batches",
        f"eval_batches_{subset_name}.json",
    )
    with open(eval_batches_fl, "r") as f:
        eval_batches = json.load(f)
    eval_batches = [
        b[0] for b in eval_batches
    ]  # take only the first (target evaluation) frame
    if remove_frame_paths:
        eval_batches = [b[:2] for b in eval_batches]
    return eval_batches


def export_result_file_dict_to_hdf5(h5path: str, filedict: Dict[str, str]):
    logger.info(f"Exporting {len(filedict)} files to HDF5 file {h5path}.")
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
    logger.info(f"Making file links in {root} to DB data in {h5path}.")
    assert h5path.endswith(".hdf5")
    with h5py.File(h5path, "r", libver='latest') as fh5:
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
    if db_file.endswith(".hdf5"):
        token = "__HDF5__:"
    elif db_file.endswith(".dbm"):
        token = "__DBM__:"
    else:
        raise ValueError(db_file)
    if overwrite and (os.path.isfile(file) or os.path.islink(file)):
        os.remove(file)
    os.symlink(db_file, file)
    
    # symlinks are cleaner ... do not use this anymore
    # with open(file, "w") as f:
    #     f.write(token+os.path.normpath(os.path.abspath(db_file)))


def _store_binary_file_data_to_hd5_dataset(dset, fl: str, idx: int):
    with open(fl, "rb") as fin:
        binary_data = fin.read()
    dset[idx] = np.fromstring(binary_data, dtype='uint8')