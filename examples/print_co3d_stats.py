import os
import json
from joblib import Parallel, delayed
from collections import defaultdict
from tabulate import tabulate
from typing import List
from collections import Counter
from co3d.dataset.data_types import (
    load_dataclass_jgzip,
    FrameAnnotation,
    SequenceAnnotation,
)


DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")


def _count_category(category):
    fa_file = os.path.join(DATASET_ROOT, category, "frame_annotations.jgz")
    sa_file = os.path.join(DATASET_ROOT, category, "sequence_annotations.jgz")

    frame_annos = load_dataclass_jgzip(fa_file, List[FrameAnnotation])
    # sequence_annos = load_dataclass_jgzip(sa_file, List[SequenceAnnotation])
    
    seq_to_frame_annos = defaultdict(list)
    for fa in frame_annos:
        seq_to_frame_annos[fa.sequence_name].append(fa)
    seq_to_frame_annos = dict(seq_to_frame_annos)

    seq_set_cnt = Counter()
    for _, frame_anno_list in seq_to_frame_annos.items():
        seq_set, _ = frame_anno_list[0].meta["frame_type"].split("_")
        seq_set_cnt.update([seq_set])

    return dict(seq_set_cnt)


def main():
    # get the category list
    with open(os.path.join(DATASET_ROOT, "category_to_subset_name_list.json"), "r") as f:
        category_to_subset_name_list = json.load(f)

    categories = sorted(list(category_to_subset_name_list.keys()))
    cat_to_n_per_set = {}
    # for category in tqdm(categories):
    #     cat_to_n_per_set[category] = dict(seq_set_cnt)

    counts_per_category = Parallel(n_jobs=20)(
        delayed(_count_category)(c) for c in categories
    )

    cat_to_n_per_set = dict(zip(categories, counts_per_category))

    seq_sets_ = list(cat_to_n_per_set[categories[0]].keys())
    tab = [
        [category, *[cat_to_n_per_set[category].get(set_, 0) for set_ in seq_sets_]]
        for category in cat_to_n_per_set
    ]
    print(tabulate(tab, headers=["category", *seq_sets_]))


if __name__=="__main__":
    main()