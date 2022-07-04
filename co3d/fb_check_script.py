import os
import tempfile
from download_dataset import main as download_main
from check_checksum import main as md5_main
from fb_get_co3d_urls import dump_default_co3d_urls


N_DOWNLOAD_WORKERS = 20
N_EXTRACT_WORKERS = 20


def main(work_dir: str):
    link_list_file = os.path.join(work_dir, "links.txt")
    print(f"Downloading links to {link_list_file}.")
    dump_default_co3d_urls(link_list_file)
    print("Downloading dataset.")
    download_main(
        link_list_file,
        work_dir,
        n_download_workers=N_DOWNLOAD_WORKERS,
        n_extract_workers=N_EXTRACT_WORKERS,
    )
    print("Checking md5s.")
    md5_main(work_dir, False)

    
if __name__=="__main__":
    root_folder = "/checkpoint/dnovotny/co3d_download_check/"
    os.makedirs(root_folder, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=root_folder) as temporary_dir:
        main(temporary_dir)