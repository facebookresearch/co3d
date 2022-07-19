import os
import csv
from typing import List, Any

from ..data_types import CO3DTask, CO3DSequenceSet


BLANK_PREDICTION_RESULTS = {}


def _read_result_csv(fl: str):
    with open(fl, "r") as f:
        csvreader = csv.reader(f)
        rows = [row for row in csvreader]
    header = rows[0]
    data = rows[1:-1]
    def _getcol(col_name: str, row: List[Any]) -> Any:
        c = row[header.index(col_name)]
        try:
            return float(c)
        except:
            return c
    parsed = {
        (_getcol("Category", r), _getcol("Subset name", r)): {
            k: _getcol(k, r) for k in header
        } for r in data
    }
    return parsed


for task in [CO3DTask.FEW_VIEW, CO3DTask.MANY_VIEW]:
    for seq_set in [CO3DSequenceSet.DEV, CO3DSequenceSet.TEST]:
        result_file = os.path.join(
            os.path.dirname(__file__),
            f"{task.value}_{seq_set.value}.csv"
        )
        BLANK_PREDICTION_RESULTS[(task, seq_set)] = _read_result_csv(result_file)

