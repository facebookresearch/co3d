# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from typing import Dict, List, NamedTuple, Tuple
import unittest

from co3d.dataset import data_types as types
from co3d.dataset.data_types import FrameAnnotation


class TestDatasetTypes(unittest.TestCase):
    def setUp(self):
        self.entry = FrameAnnotation(
            frame_number=23,
            sequence_name="1",
            frame_timestamp=1.2,
            image=types.ImageAnnotation(path="/tmp/1.jpg", size=(224, 224)),
            mask=types.MaskAnnotation(path="/tmp/1.png", mass=42.0),
            viewpoint=types.ViewpointAnnotation(
                R=(
                    (1, 0, 0),
                    (1, 0, 0),
                    (1, 0, 0),
                ),
                T=(0, 0, 0),
                principal_point=(100, 100),
                focal_length=(200, 200),
            ),
        )

    def test_asdict_rec(self):
        first = [dataclasses.asdict(self.entry)]
        second = types._asdict_rec([self.entry])
        self.assertEqual(first, second)

    def test_parsing(self):
        """Test that we handle collections enclosing dataclasses."""

        class NT(NamedTuple):
            annot: FrameAnnotation

        dct = dataclasses.asdict(self.entry)

        parsed = types._dataclass_from_dict(dct, FrameAnnotation)
        self.assertEqual(parsed, self.entry)

        # namedtuple
        parsed = types._dataclass_from_dict(NT(dct), NT)
        self.assertEqual(parsed.annot, self.entry)

        # tuple
        parsed = types._dataclass_from_dict((dct,), Tuple[FrameAnnotation])
        self.assertEqual(parsed, (self.entry,))

        # list
        parsed = types._dataclass_from_dict(
            [
                dct,
            ],
            List[FrameAnnotation],
        )
        self.assertEqual(
            parsed,
            [
                self.entry,
            ],
        )

        # dict
        parsed = types._dataclass_from_dict({"k": dct}, Dict[str, FrameAnnotation])
        self.assertEqual(parsed, {"k": self.entry})
