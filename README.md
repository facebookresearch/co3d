<center>
<img src="./co3d_logo.png" width="400" />
</center>

<br>

CO3Dv2: Common Objects In 3D (version 2) 
========================================

This repository contains a set of tools for working with the <b>2nd version</b> of the Common Objects in 3D <i>(CO3Dv2)</i> dataset.

The original dataset has been introduced in our ICCV'21 paper: [Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction](https://arxiv.org/abs/2109.00512). For accessing the original data, please switch to the `v1` branch of this repository.

<center>
<img src="./grid.gif" width="800" />
</center>


## New features in CO3Dv2
- <b>[Common Objects in 3D Challenge](https://eval.ai/web/challenges/challenge-page/1819/overview) which allows transparent evaluation on a hidden test server - more details in the [challenge README](./co3d/challenge/README.md) </b>
- 2x larger number of sequences, and 4x larger number of frames
- Improved image quality - less blocky artifacts due to better video decoding
- Improved segmentation masks - stable tracking of the main foreground object without jumping to background objects
- Enabled downloading of a smaller single-sequence subset of ~100 sequences consisting only of the sequences used to evalute the many-view single-sequence task
- Dataset files are hosted in 20 GB chunks facilitating more stable downloads
- A novel, more user-friendly, dataset format
- All images within a sequence are cropped to the same height x width


## Download the dataset
The links to all dataset files are present in this repository in `dataset/links.json`.


### Automatic batch-download
We also provide a python script that allows downloading all dataset files at once.
In order to do so, execute the download script:

```
python ./co3d/download_dataset.py --download_folder DOWNLOAD_FOLDER
```

where `DOWNLOAD_FOLDER` is a local target folder for downloading the dataset files.
Make sure to create this folder before commencing the download.

<b>Size:</b> All zip files of the dataset occupy <b>5.5 TB of disk-space</b>.


### Single-sequence dataset subset
We also provide a subset of the dataset consisting only of the sequences selected for the many-view single-sequence task where both training and evaluation are commonly conducted on a single image sequence. In order to download this subset add the
`--single_sequence_subset` option to `download_dataset.py`:
    
```
python ./co3d/download_dataset.py --download_folder DOWNLOAD_FOLDER --single_sequence_subset
```

<b>Size:</b> The single-sequence subset is much smaller than the full dataset and takes <b>8.9 GB of disk-space</b>.


# Common Objects in 3D Challenge
<center>
<img src="./co3d/challenge/co3d_challenge_logo.png" width="400" />
</center>
Together with releasing v2 of the dataset, we also organize the Common Objects in 3D Challenge hosted on EvalAI.
Please visit the [challenge website](https://eval.ai/web/challenges/challenge-page/1819/overview) and [challenge README](./co3d/challenge/README.md) for the more information.


# Installation
This is a `Python 3` / `PyTorch` codebase.
1) [Install `PyTorch`.](https://pytorch.org/)
2) [Install `PyTorch3D`.](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone)
    - <b>Please note that Pytorch3D has to be built from source to enable the Implicitron module</b>
3) Install the remaining dependencies in `requirements.txt`:
```
pip install visdom tqdm requests h5py 
```
4) Install the CO3D package itself: `pip install -e .`


##  Dependencies
- [`PyTorch`](https://pytorch.org/)
- [`PyTorch3D`](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone) (built from source)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`visdom`](https://github.com/facebookresearch/visdom)
- [`requests`](https://docs.python-requests.org/en/master/)
- ['h5py'](http://www.h5py.org/)

Note that the core data model in `co3d/dataset/data_types.py` is independent of `PyTorch`/`PyTorch3D` and can be imported and used with other machine-learning frameworks.


# Getting started
1. Install dependencies - See [Installation](#installation) above.
2. Download the dataset [here] to a given root folder `CO3DV2_DATASET_ROOT`.
3. Set the environment variable `CO3DV2_DATASET_ROOT` to the dataset root:
    ```bash
    export CO3DV2_DATASET_ROOT="your_dataset_root_folder"
    ```
4. Run `example_co3d_challenge_submission.py`:
    ```
    cd examples
    python example_co3d_challenge_submission.py
    ```
    Note that `example_co3d_challenge_submission.py` runs an evaluation of a simple depth-based image rendering (DBIR) model on all challenges and sets of the CO3D Challenge. Feel free to extend the script in order to provide your own submission to the CO3D Challenge.


# Running tests
Unit tests can be executed with:
```
python -m unittest
```


# Reproducing results
[Implicitron](https://github.com/facebookresearch/pytorch3d/tree/main/projects/implicitron_trainer) is our open-source framework used to train all implicit shape learning methods from the CO3D paper.
Please visit the following link for more details:
https://github.com/facebookresearch/pytorch3d/tree/main/projects/implicitron_trainer


# Dataset format
The dataset is organized in the filesystem as follows:

```
CO3DV2_DATASET_ROOT
    ├── <category_0>
    │   ├── <sequence_name_0>
    │   │   ├── depth_masks
    │   │   ├── depths
    │   │   ├── images
    │   │   ├── masks
    │   │   └── pointcloud.ply
    │   ├── <sequence_name_1>
    │   │   ├── depth_masks
    │   │   ├── depths
    │   │   ├── images
    │   │   ├── masks
    │   │   └── pointcloud.ply
    │   ├── ...
    │   ├── <sequence_name_N>
    │   ├── set_lists
    │       ├── set_lists_<subset_name_0>.json
    │       ├── set_lists_<subset_name_1>.json
    │       ├── ...
    │       ├── set_lists_<subset_name_M>.json
    │   ├── eval_batches
    │   │   ├── eval_batches_<subset_name_0>.json
    │   │   ├── eval_batches_<subset_name_1>.json
    │   │   ├── ...
    │   │   ├── eval_batches_<subset_name_M>.json
    │   ├── frame_annotations.jgz
    │   ├── sequence_annotations.jgz
    ├── <category_1>
    ├── ...
    ├── <category_K>
```

The dataset contains sequences named `<sequence_name_i>` from `K` categories with
names `<category_j>`. Each category comprises sequence folders `<category_k>/<sequence_name_i>` containing the list of sequence images, depth maps, foreground masks, and valid-depth masks `images`, `depths`, `masks`, and `depth_masks` respectively. Furthermore, `<category_k>/<sequence_name_i>/set_lists/` stores `M` json files `set_lists_<subset_name_l>.json`, each describing a certain sequence subset.

Users specify the loaded dataset subset by setting `self.subset_name` to one of the
available subset names `<subset_name_l>`.

`frame_annotations.jgz` and `sequence_annotations.jgz` are gzipped json files containing the list of all frames and sequences of the given category stored as lists of `FrameAnnotation` and `SequenceAnnotation` objects respectivelly.


## Set lists

Each `set_lists_<subset_name_l>.json` file contains the following dictionary:   
```
{
    "train": [
        (sequence_name: str, frame_number: int, image_path: str),
        ...
    ],
    "val": [
        (sequence_name: str, frame_number: int, image_path: str),
        ...
    ],
    "test": [
        (sequence_name: str, frame_number: int, image_path: str),
        ...
    ],
}
```
defining the list of frames (identified with their `sequence_name` and `frame_number`) in the "train", "val", and "test" subsets of the dataset.

<i>Note that `frame_number` can be obtained only from `frame_annotations.jgz` and does not necesarrily correspond to the numeric suffix of the corresponding image file name (e.g. a file `<category_0>/<sequence_name_0>/images/frame00005.jpg` can have its frame number set to 20, not 5).</i>


### Available subset names in CO3Dv2

In CO3DV2, by default, each category contains a _subset_ of the following set lists:
```
"set_lists_fewview_test.json"  # Few-view task on the "test" sequence set.
"set_lists_fewview_dev.json"  # Few-view task on the "dev" sequence set.
"set_lists_manyview_test.json"  # Many-view task on the "test" sequence of a category.
"set_lists_manyview_dev_0.json"  # Many-view task on the 1st "dev" sequence of a category.
"set_lists_manyview_dev_1.json"  # Many-view task on the 2nd "dev" sequence of a category.
```

## Eval batches

Each `eval_batches_<subset_name_l>.json` file contains a list of evaluation examples in the following form:
```
[
    [  # batch 1
        (sequence_name: str, frame_number: int, image_path: str),
        ...
    ],
    [  # batch 1
        (sequence_name: str, frame_number: int, image_path: str),
        ...
    ],
]
```
Note that the evaluation examples always come from the `"test"` part of the corresponding set list `set_lists_<subset_name_l>.json`.

<b>The evaluation task</b> then consists of generating the first image in each batch given the knowledge of the other ones. Hence, the first image in each batch represents the (unseen) target frame, for which only the camera parameters are known, while the rest of the images in the batch are the known source frames whose cameras and colors are given.

Note that for the Many-view task, where a user is given many known views of a particular sequence and the goal is to generate held-out views from the same sequence, `eval_batches_manyview_<sequence_set>_<sequence_id>.json` contain a single (target) frame per evaluation batch. Users can obtain the known views from the corresponding `"train"` list of frames in the set list `set_lists_manyview_<sequence_set>_<sequence_id>.json`.


# PyTorch-independent usage
The core data model in `co3d/dataset/data_types.py` is independent of `PyTorch`/`PyTorch3D` and can be imported and used with other machine-learning frameworks.

For example, in order to load the per-category frame and sequence annotations users can execute the following code:
```python
from typing import List
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
category_frame_annotations = load_dataclass_jgzip(
    f"{CO3DV2_DATASET_ROOT}/{category_name}/frame_annotations.jgz", List[FrameAnnotation]
)
category_sequence_annotations = load_dataclass_jgzip(
    f"{CO3DV2_DATASET_ROOT}/{category_name}/sequence_annotations.jgz", List[SequenceAnnotation]
)
```

Furthermore, all challenge-related code under `co3d/challenge` also does not depend on `PyTorch`.


# Reference
If you use our dataset, please use the following citation:
```
@inproceedings{reizenstein21co3d,
	Author = {Reizenstein, Jeremy and Shapovalov, Roman and Henzler, Philipp and Sbordone, Luca and Labatut, Patrick and Novotny, David},
	Booktitle = {International Conference on Computer Vision},
	Title = {Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction},
	Year = {2021},
}
```


# License
The CO3D codebase is released under the [CC BY-NC 4.0](LICENSE).


# Overview video
The following presentation of the dataset was delivered at the Extreme Vision Workshop at CVPR 2021:
[![Overview](https://img.youtube.com/vi/hMx9nzG50xQ/0.jpg)](https://www.youtube.com/watch?v=hMx9nzG50xQ)
