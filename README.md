<center>
<img src="./co3d_logo.png" width="400" />
</center>

<br>

CO3D: Common Objects In 3D
==========================

This repository contains a set of tools for working with the Common Objects in 3D <i>(CO3D)</i> dataset. 
The dataset has been introduced in our ICCV'21 paper: [Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction](https://arxiv.org/abs/2109.00512)

<center>
<img src="./grid.gif" width="600" />
</center>


## [Download the dataset](https://ai.facebook.com/datasets/CO3D-dataset)
The dataset can be downloaded from the following Facebook AI Research web page:
[download link](https://ai.facebook.com/datasets/co3d-downloads/)


### Automatic batch-download
We also provide a python script that allows downloading all dataset files at once:
1) Open [CO3D downloads page](https://ai.facebook.com/datasets/co3d-downloads/) in your browser.
2) Download the file with CO3D file links at the bottom of the page.
3) Execute the download script:
    ```
    python ./download_dataset.py --link_list_file LINK_LIST_FILE --download_folder DOWNLOAD_FOLDER
    ```
where `LINK_LIST_FILE` is the file downloaded at step 2) above, and `DOWNLOAD_FOLDER` is is a local target folder for downloading the dataset files.


## Installation
This is a `Python 3` / `PyTorch` codebase.
1) [Install `PyTorch`.](https://pytorch.org/)
2) [Install `PyTorch3D`.](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
3) Install the remaining dependencies in `requirements.txt`:
```
pip install lpips visdom tqdm requests
```
Note that the core data model in `dataset/types.py` is independent of `PyTorch` and can be imported and used with other machine-learning frameworks.


##  Dependencies
- [`PyTorch`](https://pytorch.org/)
- [`PyTorch3D`](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`visdom`](https://github.com/facebookresearch/visdom)
- [`lpips`](https://github.com/richzhang/PerceptualSimilarity)
- [`requests`](https://docs.python-requests.org/en/master/)


## Getting started
1. Install dependencies - See [Installation](#installation) above.
2. Download the dataset [here](https://ai.facebook.com/datasets/co3d-downloads/) to a given root folder `DATASET_ROOT_FOLDER`.
3. In `dataset/dataset_zoo.py` set the `DATASET_ROOT` variable to your DATASET_ROOT_FOLDER`:
    ```
    dataset_zoo.py:25: DATASET_ROOT = DATASET_ROOT_FOLDER
    ```
4. Run `eval_demo.py`:
    ```
    python eval_demo.py
    ```
    Note that `eval_demo.py` runs an evaluation of a simple depth-based image rendering (DBIR) model on the same data as in the paper. Hence, the results are directly comparable to the numbers reported in the paper.


## Running tests
Unit tests can be executed with:
```
python -m unittest
```


## Reference
If you use our dataset, please use the following citation:
```
@inproceedings{reizenstein21co3d,
	Author = {Reizenstein, Jeremy and Shapovalov, Roman and Henzler, Philipp and Sbordone, Luca and Labatut, Patrick and Novotny, David},
	Booktitle = {International Conference on Computer Vision},
	Title = {Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction},
	Year = {2021},
}
```


## License
The CO3D codebase is released under the [BSD License](LICENSE).


## Overview video
The following presentation of the dataset was delivered at the Extreme Vision Workshop at CVPR 2021:
[![Overview](https://img.youtube.com/vi/hMx9nzG50xQ/0.jpg)](https://www.youtube.com/watch?v=hMx9nzG50xQ)
