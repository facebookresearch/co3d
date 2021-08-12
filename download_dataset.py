# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import requests
import argparse
import functools
from typing import List, Optional

from dataset.dataset_zoo import CO3D_CATEGORIES
from multiprocessing import Pool
from tqdm import tqdm


def main(
    link_list_file: str,
    download_folder: str,
    n_download_workers: int=4, 
    n_extract_workers: int=4,
    download_categories: Optional[List[str]] = None,
):
    """
    Downloads the CO3D dataset.
    
    Args:
        link_list_file: A text file with the list of CO3D file download links.
            Please visit https://ai.facebook.com/datasets/co3d-downloads/ 
            in order to obtain the file.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers 
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers 
            for extracting the dataset files.
        download_categories: A list of categories to download. 
            If `None`, downloads all.
    """

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a file"
            " with CO3D download links."
            + " The file with links can be downloaded at"
            + " https://ai.facebook.com/datasets/co3d-downloads/ ."
        )

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder" 
            + " for downloading the CO3D dataset."
        )


    with open(link_list_file, 'r') as f:
        links = f.readlines()

    if (len(links) != 51) or any(not l.startswith('https://') for l in links):
        raise ValueError(
            "Unexpected format of `link_list_file`."
            " The file has to contain 51 lines, each line must be a hyperlink."    
        )

    if download_categories is not None:    
        not_in_co3d = [c for c in download_categories if c not in CO3D_CATEGORIES]
        if len(not_in_co3d) > 0:
            raise ValueError(
                f"download_categories {str(not_in_co3d)} are not valid"
                + "CO3D categories."
            )
        links = [
            l for li, l in enumerate(links)
            if CO3D_CATEGORIES[li] in download_categories
        ]

    print(f"Downloading {len(links)} CO3D dataset files ...")
    with Pool(processes=n_download_workers) as download_pool: 
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_category_file, download_folder),
                links,
            ),
            total=len(links),
        ):
            pass

    print(f"Extracting {len(links)} CO3D dataset files ...")
    with Pool(processes=n_extract_workers) as extract_pool: 
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(_unpack_category_file, download_folder),
                links
            ),
            total=len(links),
        ):
            pass
        
    print("Done")


def _get_local_fl_from_link(download_folder: str, link: str):
    file_name = os.path.split(link)[-1]
    local_fl = os.path.join(download_folder, file_name)
    return local_fl

def _unpack_category_file(download_folder: str, link: str):
    local_fl = _get_local_fl_from_link(download_folder, link)
    print(f"Unpacking CO3D dataset file {local_fl} to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)
    
def _download_category_file(download_folder: str, link: str):
    local_fl = _get_local_fl_from_link(download_folder, link)
    print(f"Downloading CO3D dataset file {link} to {local_fl}.")
    r = requests.get(link)
    with open(local_fl, "wb") as f:
        f.write(r.content)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Download the CO3D dataset.')
    parser.add_argument(
        '--link_list_file', 
        type=str,
        help=(
            "The file with html links to the CO3D dataset files."
            + "The file can be obtained at https://ai.facebook.com/datasets/co3d-downloads/ ."
        )
    )
    parser.add_argument(
        '--download_folder', 
        type=str,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        '--n_download_workers', 
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        '--n_extract_workers', 
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        '--download_categories', 
        type=lambda x: [x_.strip() for x_ in x.split(',')],
        default=None,
        help="A comma-separated list of CO3D categories to download."
        + " Example: 'orange,car' will download only oranges and cars",
    )
    args = parser.parse_args()
    main(
        str(args.link_list_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_categories=args.download_categories,
    )