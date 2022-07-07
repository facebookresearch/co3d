<style>
r { color: Red }
o { color: Orange }
g { color: Green }
</style>

Common Objects in 3D Challenge
==============================

The following describes the Common Objects in 3D Challenge (CO3DC).

# Quick start
See example code for creating a CO3DC submission in:
```
<co3d_repository_root>/examples/example_co3d_challenge_submission.py
```
note that the codebase also requires the latest PyTorch3D installation.

After running the evaluation code, please send the produced .zip file to the
EvalAI evaluation server:
!!!! TODO !!!!


# CO3D challenge overview

CO3D challenge evaluates New-view Synthesis methods.

More specifically, <b>given a set of known "<o>source</o>" views of an object, the goal
is to generate new, previously unobserved, "<g>target</g>" views of the scene</b>.

The challenge has 2 tracks - _Many-view_, and _Few-view_.

### _Many-view_ task

This is the standard scenario popularized by e.g. NeRF. Given many
(~100) known source views of a scene, the goal is to generate target views
that are relative close to the source ones.

### _Few-view_ task

Here, the goal is the same as in Many-view, with the difference that only
a very small number of source views is known (2-10). Methods are likely to succced
only if they exploit category-centric geometry/appearance prior that can be learned
from the category-centric training data.

### CO3Dv2 Dataset

The CO3Dv2 dataset provides all training and testing data needed for a submission.

### Evaluation data

Each evaluation example contains several <o>source</o> views and a single
<g>target</g> view. For each <o>source view</o>, the corresponding color image,
foreground segmentation mask, and camera parameters are given.
Given this information, the goal is to generate the <g>target</g> view, for which only the
camera parameters are given.


# CO3D challenge software framework
The `co3d` repository contains tooling that allow a simple generation and submission
of challenge entries.

## Submission guide
1) Install the `co3d` package:
    ```
    git clone https://github.com/facebookresearch/co3d
    cd co3d
    pip install -e .
    ```

2) Start by importing the `CO3DSubmission` class and instantiate a submission run.
    For example, the following code:
    ```python
    from co3d.challenge.co3d_submission import CO3DSubmission
    output_folder = "./co3d_submission_files"
    task = CO3DTask.MANY_VIEW
    sequence_set = CO3DSequenceSet.TEST
    
    submission = CO3DSubmission(
        task=task
        sequence_set=sequence_set,
        output_folder=output_folder,
        dataset_root=dataset_root,
    )
    ```
    will instantiate a CO3D submission object `submission` that stores (and optionally
    evaluates) results of the `manyview` task on the `test` set. All results will be
    stored in the `output_folder`. Note that a user has to also specify the local root
    folder of the CO3D dataset in `dataset_root`.

3) Obtain the dictionary of evaluation examples `eval_batches_map` from `submission`.
    ```python
    eval_batches_map = submission.get_eval_batches_map()
    ```
    here, `eval_batches_map` is a dictionary of the following form:
    ```
    {(category: str, subset_name: str): eval_batches}  # eval_batches_map
    ```
    where `eval_batches` look as follows:
    ```python
    [
        [
            (sequence_name_0: str, frame_number_0: int),
            (sequence_name_0: str, frame_number_1: int),
            ...
            (sequence_name_0: str, frame_number_M_0: int),
        ],
        ...
        [
            (sequence_name_N: str, frame_number_0: int),
            (sequence_name_N: str, frame_number_1: int),
            ...
            (sequence_name_N: str, frame_number_M_N: int),
        ]
    ]  # eval_batches
    ```
    Containing a list of `N` evaluation examples, each consisting of a tuple of 
    `M_i` frames with numbers `frame_number_j` from a given sequence name `sequence_name_i`.
    Note that the mapping between `frame_number` and `sequence_name` to the CO3D data
    is stored in the respective `frame_annotations.jgz` and `sequence_annotation.jgz`
    files in `<dataset_root>/<sequence_category>`.

    For the <b>Many-view task</b> (`CO3DTask.MANYVIEW`), each evaluation batch has a single
    (`M_i=1`) frame, which is the target evaluation frame.

    For the <b>Few-view task</b> (`CO3DTask.FEWVIEW`), each batch has several frames (`M_i>1`),
    where the first frame is the target frame which should be predicted given the knowledge
    of the source frames that correspondond oto the 2nd-to-last elements of each batch.


4) Next we iterate over eval_batches, predict new views, and store our predictions
with the `submission` object.
    ```python
    # iterate over evaluation subsets and categories
    for (category, subset_name), eval_batches in eval_batches_map.items():
        
        # iterate over all evaluation examples of a given category and subset
        for eval_batch in eval_batches:
            # parse the evaluation sequence name and target frame number from eval_batch
            sequence_name, frame_number = eval_batch[0][:2]
            
            # `predict_new_view` is a user-defined function which generates
            # the test view (corresponding to the first element of the eval batch)
            image, depth, mask = predict_new_view(eval_batch, ...)  
            
            # add the render to the submission
            submission.add_result(
                category=category,
                subset_name=subset_name,
                sequence_name=sequence_name,
                frame_number=frame_number,
                image=image,
                mask=mask,
                depth=depth,
            )
    ```

5) Finally, we export the submission object to a zip file that can be uploaded to the
EvalAI server:
    ```
    submission.export_results()
    ```

6) To submit an official evaluation entry, submit the resulting zip file to the
EvalAI submission server:
    !!!!! TODO !!!!!
