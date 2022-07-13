# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RGBDAFrame:
    image: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    depth_mask: Optional[np.ndarray] = None


class CO3DTask(Enum):
    MANY_VIEW = "manyview"
    FEW_VIEW = "fewview"


class CO3DSequenceSet(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"