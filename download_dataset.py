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
    Downloads and unpacks the CO3D dataset.
    
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
            + f" {download_folder} does not exist."
        )

    # read the link file
    with open(link_list_file, 'r') as f:
        links = [l.strip() for l in f.readlines()[1:]]

    if (len(links) != 51):
        raise ValueError(
            "Unexpected format of `link_list_file`."
            " The file has to contain 51 lines, each line must be a hyperlink."    
        )

    # convert to a list of tuples [(category, link)]
    links = {
        (l_[0].replace('CO3D_', '').replace('.zip', ''), l_[1])
        for l_ in [l.split('\t') for l in links]
    }

    if download_categories is not None:    
        co3d_categories = [l[0] for l in links]
        not_in_co3d = [c for c in download_categories if c not in co3d_categories]
        if len(not_in_co3d) > 0:
            raise ValueError(
                f"download_categories {str(not_in_co3d)} are not valid"
                + "CO3D categories."
            )
        links = [(c, l) for c, l in links if c in download_categories]

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
    local_fl = os.path.join(download_folder, link[0] + '.zip')
    print(f"Unpacking CO3D dataset file {local_fl} ({link[1]}) to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)
    
def _download_category_file(download_folder: str, link: str):
    local_fl = os.path.join(download_folder, link[0] + '.zip')
    print(f"Downloading CO3D dataset file {link[1]} ({link[0]}) to {local_fl}.")
    _download_with_progress_bar(link[1], local_fl, link[0])

def _download_with_progress_bar(url: str, fname: str, category: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
            size = file.write(data)
            bar.update(size)
            if datai % max(((total//1024)//20),1) == 0:
                print(f"{category}: Downloaded {100.0*(float(bar.n)/total):3.1f}%.")
                print(bar)
                

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


