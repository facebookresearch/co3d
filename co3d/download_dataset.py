# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import requests
import argparse
import functools
import json
import warnings
from typing import List, Optional

from check_checksum import check_co3d_sha256, DEFAULT_SHA256S_FILE
from multiprocessing import Pool
from tqdm import tqdm


DEFAULT_LINK_LIST_FILE = os.path.join(os.path.dirname(__file__), "links.json")


def main(
    link_list_file: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    download_categories: Optional[List[str]] = None,
    checksum_check: bool = False,
    single_sequence_subset: bool = False,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
    sha256s_file: str = DEFAULT_SHA256S_FILE,
):
    """
    Downloads and unpacks the CO3D dataset.

    Note: The script will make a folder `<download_folder>/_in_progress`, which
        stores files whose download is in progress. The folder can be safely deleted
        the download is finished.

    Args:
        link_list_file: A text file with the list of CO3D file download links.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        download_categories: A list of categories to download.
            If `None`, downloads all.
        checksum_check: Enable validation of the downloaded file's checksum before
            extraction.
        single_sequence_subset: Whether the downloaded dataset is the single-sequence
            subset of the full dataset.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
        skip_downloaded_archives: Skip re-downloading already downloaded archives.
    """

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a file"
            " with CO3D download links."
            " The file is stored in the co3d github:"
            " https://github.com/facebookresearch/co3d/blob/main/co3d/links.json"
        )

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the CO3D dataset."
            + f" {download_folder} does not exist."
        )

    # read the link file
    with open(link_list_file, "r") as f:
        links = json.load(f)

    # get the full dataset links or the single-sequence subset links
    links = links["singlesequence"] if single_sequence_subset else links["full"]

    # split to data links and the links containing json metadata
    metadata_links = []
    data_links = []
    for category_name, urls in links.items():
        for url in urls:
            link_name = os.path.split(url)[-1]
            if single_sequence_subset:
                link_name = link_name.replace("_singlesequence", "")
            if category_name=="METADATA":
                metadata_links.append((link_name, url))
            else:
                data_links.append((category_name, link_name, url))
        
    if download_categories is not None:
        co3d_categories = set(l[0] for l in data_links)
        not_in_co3d = [c for c in download_categories if c not in co3d_categories]
        if len(not_in_co3d) > 0:
            raise ValueError(
                f"download_categories {str(not_in_co3d)} are not valid"
                + "CO3D categories."
            )
        data_links = [(c, ln, l) for c, ln, l in data_links if c in download_categories]

    with Pool(processes=n_download_workers) as download_pool:
        print(f"Downloading {len(metadata_links)} CO3D metadata files ...")
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_metadata_file, download_folder),
                metadata_links,
            ),
            total=len(metadata_links),
        ):
            pass

        print(f"Downloading {len(data_links)} CO3D dataset files ...")
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_category_file,
                    download_folder,
                    checksum_check,
                    single_sequence_subset,
                    sha256s_file,
                    skip_downloaded_archives,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

        if not all(download_ok.values()):
            not_ok_links = [n for n, ok in download_ok.items() if not ok]
            not_ok_links_str = "\n".join(not_ok_links)
            raise AssertionError(
                "The SHA256 checksums did not match for some of the downloaded files:\n"
                + not_ok_links_str + "\n"
                + "This is most likely due to a network failure."
                + " Please restart the download script."
            )

    print(f"Extracting {len(data_links)} CO3D dataset files ...")
    with Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_category_file,
                    download_folder,
                    clear_archives_after_unpacking,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print("Done")


def _unpack_category_file(
    download_folder: str,
    clear_archive: bool,
    link: str,
):
    category, link_name, url = link
    local_fl = os.path.join(download_folder, link_name)
    print(f"Unpacking CO3D dataset file {local_fl} ({link_name}) to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)
    if clear_archive:
        os.remove(local_fl)


def _download_category_file(
    download_folder: str,
    checksum_check: bool,
    single_sequence_subset: bool,
    sha256s_file: str,
    skip_downloaded_files: bool,
    link: str,
):
    category, link_name, url = link
    
    local_fl_final = os.path.join(download_folder, link_name)

    if skip_downloaded_files and os.path.isfile(local_fl_final):
        print(f"Skippping {local_fl_final}, already downloaded!")
        return link_name, True

    in_progress_folder = os.path.join(download_folder, "_in_progress")
    os.makedirs(in_progress_folder, exist_ok=True)
    local_fl = os.path.join(in_progress_folder, link_name)

    print(f"Downloading CO3D dataset file {link_name} ({url}) to {local_fl}.")
    _download_with_progress_bar(url, local_fl, link_name)
    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        try:
            check_co3d_sha256(
                local_fl,
                single_sequence_subset=single_sequence_subset,
                sha256s_file=sha256s_file,
            )
        except AssertionError:
            warnings.warn(
                f"Checksums for {local_fl} did not match!"
                + " This is likely due to a network failure,"
                + " please restart the download script." 
            )
            return link_name, False
        
    os.rename(local_fl, local_fl_final)
    return link_name, True


def _download_metadata_file(download_folder: str, link: str):
    local_fl = os.path.join(download_folder, link[0])
    # remove the singlesequence postfix in case we are downloading the s.s. subset
    local_fl = local_fl.replace("_singlesequence", "")
    print(f"Downloading CO3D metadata file {link[1]} ({link[0]}) to {local_fl}.")
    _download_with_progress_bar(link[1], local_fl, link[0])


def _download_with_progress_bar(url: str, fname: str, filename: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    print(url)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
            size = file.write(data)
            bar.update(size)
            if datai % max((max(total // 1024, 1) // 20), 1) == 0:
                print(f"{filename}: Downloaded {100.0*(float(bar.n)/max(total, 1)):3.1f}%.")
                print(bar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the CO3D dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--download_categories",
        type=lambda x: [x_.strip() for x_ in x.split(",")],
        default=None,
        help="A comma-separated list of CO3D categories to download."
        + " Example: 'orange,car' will download only oranges and cars",
    )
    parser.add_argument(
        "--link_list_file",
        type=str,
        default=DEFAULT_LINK_LIST_FILE,
        help=(
            "The file with html links to the CO3D dataset files."
            + " In most cases the default local file `links.json` should be used."
        ),
    )
    parser.add_argument(
        "--sha256_file",
        type=str,
        default=DEFAULT_SHA256S_FILE,
        help=(
            "The file with SHA256 hashes of CO3D dataset files."
            + " In most cases the default local file `co3d_sha256.json` should be used."
        ),
    )
    parser.add_argument(
        "--checksum_check",
        action="store_true",
        default=False,
        help="Check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.add_argument(
        "--single_sequence_subset",
        action="store_true",
        default=False,
        help="Download the single-sequence subset of the dataset.",
    )
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    parser.add_argument(
        "--redownload_existing_archives",
        action="store_true",
        default=False,
        help="Redownload the already-downloaded archives.",
    )
    args = parser.parse_args()
    main(
        str(args.link_list_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_categories=args.download_categories,
        checksum_check=bool(args.checksum_check),
        single_sequence_subset=bool(args.single_sequence_subset),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        sha256s_file=str(args.sha256_file),
        skip_downloaded_archives=not bool(args.redownload_existing_archives),
    )
