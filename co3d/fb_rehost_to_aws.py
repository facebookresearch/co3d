# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import argparse
import warnings
import functools
import tempfile
import shutil
import json

from multiprocessing import Pool
from tqdm import tqdm

from check_checksum import check_co3d_sha256, get_expected_sha256s, sha256_file


LICENSE_FILE = os.path.join(os.path.dirname(__file__), "LICENSE")
AWS_REMOTE_FOLDER = "s3://dl.fbaipublicfiles.com/co3d/"
AWS_REMOTE_FOLDER_HTTP = "https://dl.fbaipublicfiles.com/co3d/"
LICENSED_SHA256S_FILE = os.path.join(os.path.dirname(__file__), "co3d_sha256.json")
LINKS_FILE = os.path.join(os.path.dirname(__file__), "co3d_links.txt")


def _dump_links_file():
    with open(LICENSED_SHA256S_FILE, "r") as f:
        shas = json.load(f)
    zipnames = list(shas.keys())    
    with open(LINKS_FILE, "w") as f:
        f.write(f"file_name\tlink\n")
        for zipname in zipnames:
            link = _get_aws_public_link(zipname)
            f.write(f"{zipname}\t{link}\n")

# file_name	cdn_link
# CO3D_apple.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An_QSC6hT-8cb3Gd3PJ9U2VYscdbDWFj_Ny11wi4ptmIspsE70S_BTc8R6OkSBdIZzWNbqbOu6LEyWovGIk.zip?ccb=10-5&oh=00_AT-YNXAhaIEeqDCfT1ai5X-d8wClOEy1wjPzRmEaXO6bVg&oe=6204E3F6&_nc_sid=857daf
# CO3D_backpack.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9tyyq7fEyndpQjdl4d2UbqMuyGGEYRt32qZRLnxTLLpCQ2PC1QzurnpThigtMS9iS8ggbGL67p7t4aKqg.zip?ccb=10-5&oh=00_AT-QfEE3EQx2HDkAF7v2qNZ5l5d0A1yu_rZo-wPQ29Riqg&oe=6206C040&_nc_sid=857daf
# CO3D_ball.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An-jG2tlOIue1umcM2xNXEiGJ89IUpNYdfoa5RTWpBUzaDyS_ZyyUCwEznmuQ6K_cN6THR-IpOCSTXzN6QQ.zip?ccb=10-5&oh=00_AT9gogywZNOwxxUVg6tvtLVUURJXLVKI7rocrfldiXR2Hg&oe=6205EEC6&_nc_sid=857daf
# CO3D_banana.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9eGc31j9mtSwJJynbI8cCiLecCcwGQ2Q7V5Q9aTcWxHQQqNFbR9LeIL2WmC12AVvzJGPDIHSeRwJVCC1U.zip?ccb=10-5&oh=00_AT-hJveUXdTV3ONrv_6w-j1yr5Fi8ZUESw6Ma5TCFmlp5g&oe=6205705D&_nc_sid=857daf
# CO3D_baseballbat.zip	https://scontent.xx.fbcdn.net/m1/v/t6/An9Hwjr4vTebobXoMz51rKFcl2ucITXLOkYW3_Bj0T5eyiH00u80KE0U2e5WpMFAz49CWFl6qFQKhJ0pt2k.zip?ccb=10-5&oh=00_AT_44dSuX-__bQNr8Ya7EFRoNZXrQxzailUhCxn0BYt6Hg&oe=6205E024&_nc_sid=857daf


def _get_aws_public_link(zipname: str):
    return os.path.join(AWS_REMOTE_FOLDER_HTTP, zipname)


def _copy_to_aws(download_folder: str, zipname: str):
    local_zipfile = os.path.join(download_folder, zipname)
    # first check the file
    print(f"Checking sha256 of {local_zipfile}.")
    # check_co3d_sha256(local_zipfile)
    licensed_zip = _add_license(
        local_zipfile, 
        outdir=os.path.normpath(download_folder) + "_with_license"
    )
    # fs3cmd put {local_file} s3://fairusersglobal/users/maj/h2/checkpoint/maj/some/results.csv
    digest_ = sha256_file(licensed_zip)
    cmd = f"aws s3 --profile saml cp {licensed_zip} {os.path.join(AWS_REMOTE_FOLDER,zipname)}"
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise ValueError(f"Copying {zipname} failed!")
    return digest_


def _add_license(
    zipfile: str, 
    outdir: str, 
    temproot: str = "/checkpoint/dnovotny/tmp/",
):
    os.makedirs(temproot, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    print(f"Adding license file to {zipfile}.")
    with tempfile.TemporaryDirectory(prefix=temproot) as tmpdir:
        shutil.unpack_archive(zipfile, tmpdir)
        cat_dir = os.listdir(tmpdir)[0]
        archive_file = os.path.join(outdir, cat_dir)
        shutil.copyfile(LICENSE_FILE, os.path.join(tmpdir, cat_dir, "LICENSE"))
        shutil.make_archive(
            base_name=archive_file, 
            root_dir=tmpdir,
            format="zip", 
            base_dir=cat_dir,
        )
    return archive_file + ".zip"


def _load_aws_module():
    os.system("module load saml2aws")


def _saml_aws_login():
    os.system("saml2aws -s --disable-keychain login")


def main(download_folder: str, num_workers: int):
    # zipfiles = sorted(glob.glob(os.path.join(download_folder, "*.zip")))    
    # setup the environment
    _load_aws_module()
    # login to aws
    _saml_aws_login()   
    expected = get_expected_sha256s()

    if False:
        zipnames = list(expected.keys())
        # zipnames = [z for z in zipnames if "tv.zip" in z]

        digests_list = []
        with Pool(processes=num_workers) as download_pool: 
            for d in tqdm(
                download_pool.imap(
                    functools.partial(
                        _copy_to_aws, 
                        download_folder,
                    ),
                    zipnames
                ),
                total=len(zipnames),
            ):
                digests_list.append(d)
        # for zipname in zipnames: 
        #     digests_list.append(_copy_to_aws(download_folder, zipname))

        digests = dict(
            zip([os.path.split(z)[-1] for z in zipnames], 
            digests_list),
        )

        with open(LICENSED_SHA256S_FILE, "w") as f:
            json.dump(digests, f, indent=2)

    _dump_links_file()

    import pdb; pdb.set_trace()

    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Check SHA256 hashes of the CO3D dataset.'
    )
    parser.add_argument(
        '--download_folder', 
        type=str,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        '--num_workers', 
        type=int,
        default=51,
        help="The number of workers.",
    )
    args = parser.parse_args()
    main(
        str(args.download_folder), 
        num_workers=int(args.num_workers),
    )
