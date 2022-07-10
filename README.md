<center>
<img src="./co3d_logo.png" width="400" />
</center>

<br>

CO3D: Common Objects In 3D 
==========================

This repository contains a set of tools for working with the <b>2nd version</b> of the Common Objects in 3D <i>(CO3D)</i> dataset.

The original dataset has been introduced in our ICCV'21 paper: [Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction](https://arxiv.org/abs/2109.00512). For accessing the original data, please switch to the `v1` branch of this repository.

<center>
<img src="./grid.gif" width="800" />
</center>


## Download the dataset
The links to all dataset files are present in this repository in `dataset/links.txt`.

### Automatic batch-download
We also provide a python script that allows downloading all dataset files at once.
In order to do so, execute the download script:
    ```
    python ./co3d/dataset/download_dataset.py --download_folder DOWNLOAD_FOLDER
    ```
where `DOWNLOAD_FOLDER` is a local target folder for downloading the dataset files.
Make sure to create this folder before commencing the download.


### Single-sequence dataset subset
We also provide a subset of the dataset consisting only of the sequences selected for the
many-view single-sequence task where both training and evaluation are commonly conducted
on the same single sequence. In order to download this subset add the
`--single_sequence_subset` option to `download_dataset.py`:
    ```
    python ./co3d/dataset/download_dataset.py --download_folder DOWNLOAD_FOLDER --single_sequence_subset
    ```


## Installation
This is a `Python 3` / `PyTorch` codebase.
1) [Install `PyTorch`.](https://pytorch.org/)
2) [Install `PyTorch3D`.](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
3) Install the remaining dependencies in `requirements.txt`:
```
pip install lpips visdom tqdm requests
```
4) Install the CO3D package itself: `pip install -e .`


##  Dependencies
- [`PyTorch`](https://pytorch.org/)
- [`PyTorch3D`](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`visdom`](https://github.com/facebookresearch/visdom)
- [`lpips`](https://github.com/richzhang/PerceptualSimilarity)
- [`requests`](https://docs.python-requests.org/en/master/)


## Getting started
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


## Running tests
Unit tests can be executed with:
```
python -m unittest
```

## Reproducing results
[Implicitron](https://github.com/facebookresearch/pytorch3d/tree/main/projects/implicitron_trainer) is our open-source framework used to train all implicit shape learning methods from the CO3D paper.
Please visit the following link for more details:
https://github.com/facebookresearch/pytorch3d/tree/main/projects/implicitron_trainer


## PyTorch-independent usage
Note that the core data model in `co3d/dataset/data_types.py` is independent of `PyTorch`/`PyTorch3D` and can be imported and used with other machine-learning frameworks.

For example, in order to load the per-category frame and sequence annotations users can execute the following code:
```python
from typing import List
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
category_frame_annotations = load_dataclass_jgzip(
    f"{DATASET_ROOT}/{category_name}/frame_annotations.jgz", List[FrameAnnotation]
)
category_sequence_annotations = load_dataclass_jgzip(
    f"{DATASET_ROOT}/{category_name}/sequence_annotations.jgz", List[SequenceAnnotation]
)
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
The CO3D codebase is released under the [CC BY 4.0](LICENSE).


## Overview video
The following presentation of the dataset was delivered at the Extreme Vision Workshop at CVPR 2021:
[![Overview](https://img.youtube.com/vi/hMx9nzG50xQ/0.jpg)](https://www.youtube.com/watch?v=hMx9nzG50xQ)
