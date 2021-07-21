<center>
<img src="./co3d_logo.png" width="400" />
</center>

<br>

CO3D: Common Objects In 3D
==========================

This repository contains a set of tools for working with the Common Objects in 3D <i>(CO3D)</i> dataset.

<center>
<img src="./grid.gif" width="600" />
</center>


## [Download the dataset](https://ai.facebook.com/datasets/CO3D-dataset)
The dataset can be downloaded from the following Facebook AI Research web page:
[download link](https://ai.facebook.com/datasets/co3d-downloads/)


## Installation
This is a `python3 / PyTorch` codebase.
1) [Install `PyTorch`.](https://pytorch.org/)
2) [Install `PyTorch3D`.](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
3) Install the remaining dependencies in `requirements.txt`:
```
pip install lpips visdom tqdm
```
Note that the core data model in `dataset/types.py` is independent of `PyTorch` and can be imported and used with other machine-learning frameworks.


##  Dependencies
`requirements.txt` lists the following dependencies:
- [`PyTorch`](https://pytorch.org/)
- [`PyTorch3D`](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`visdom`](https://github.com/facebookresearch/visdom)
- [`lpips`](https://github.com/richzhang/PerceptualSimilarity)


## Getting started
1. Install dependencies - See [Instalation](#installation) above.
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


## License
PyTorch3D is released under the [BSD License](LICENSE).


## Overview video
The following presentation of the dataset was delivered at the Extreme Vision Workshop at CVPR 2021:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/hMx9nzG50xQ/0.jpg)](https://www.youtube.com/watch?v=hMx9nzG50xQ)
