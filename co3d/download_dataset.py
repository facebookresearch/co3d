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
from typing import List, Optional

from check_checksum import check_co3d_sha256
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
):
    """
    Downloads and unpacks the CO3D dataset.

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
    """

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a file"
            " with CO3D download links."
            " The file is stored in the co3d github:"
            " https://https://github.com/facebookresearch/co3d/blob/main/co3d/links.txt"
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

    if len(links) != 52:
        raise ValueError(
            f"Unexpected number of links in the `link_list_file` (should be {52})."
        )

    # convert to a list of tuples [(link_name, link)]
    links = [(os.path.splitext(lname)[0], l) for lname, l in links.items()]

    # split to data links and the links containing json metadata
    json_links = []
    data_links = []
    for link_name, url in links:
        if url.endswith(".zip"):
            data_links.append((link_name, url))
        elif url.endswith(".json"):
            json_links.append((link_name, url))
        else:
            raise ValueError(f"Unexpected link name {link_name}.")

    if download_categories is not None:
        co3d_categories = [l[0] for l in data_links]
        not_in_co3d = [c for c in download_categories if c not in co3d_categories]
        if len(not_in_co3d) > 0:
            raise ValueError(
                f"download_categories {str(not_in_co3d)} are not valid"
                + "CO3D categories."
            )
        data_links = [(c, l) for c, l in data_links if c in download_categories]

    with Pool(processes=n_download_workers) as download_pool:
        print(f"Downloading {len(json_links)} CO3D metadata files ...")
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_metadata_file, download_folder),
                json_links,
            ),
            total=len(json_links),
        ):
            pass

        print(f"Downloading {len(data_links)} CO3D dataset files ...")
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_category_file, download_folder),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print(f"Extracting {len(data_links)} CO3D dataset files ...")
    with Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_category_file,
                    download_folder,
                    checksum_check,
                    single_sequence_subset,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print("Done")


def _unpack_category_file(
    download_folder: str,
    checksum_check: bool,
    single_sequence_subset: bool,
    link: str,
):
    local_fl = os.path.join(download_folder, link[0] + ".zip")
    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        check_co3d_sha256(local_fl, single_sequence_subset=single_sequence_subset)
    print(f"Unpacking CO3D dataset file {local_fl} ({link[1]}) to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)


def _download_category_file(download_folder: str, link: str):
    local_fl = os.path.join(download_folder, link[0] + ".zip")
    print(f"Downloading CO3D dataset file {link[1]} ({link[0]}) to {local_fl}.")
    _download_with_progress_bar(link[1], local_fl, link[0])


def _download_metadata_file(download_folder: str, link: str):
    local_fl = os.path.join(download_folder, link[0] + ".json")
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
            if datai % max(((total // 1024) // 20), 1) == 0:
                print(f"{filename}: Downloaded {100.0*(float(bar.n)/total):3.1f}%.")
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
            + " In most cases the default local file `co3d_links.txt` should be used."
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
    args = parser.parse_args()
    main(
        str(args.link_list_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_categories=args.download_categories,
        checksum_check=bool(args.checksum_check),
        single_sequence_subset=bool(args.single_sequence_subset),
    )
