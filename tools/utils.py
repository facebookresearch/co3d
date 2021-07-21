# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import collections
from contextlib import contextmanager
import dataclasses
import torch
import time


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def try_to_cuda(t):
    try:
        t = t.cuda()
    except AttributeError:
        pass
    return t


def dict_to_cuda(batch):
    return {k: try_to_cuda(v) for k, v in batch.items()}


def dataclass_to_cuda_(obj):
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


# TODO: test it
def cat_dataclass(batch, tensor_collator):
    elem = batch[0]
    collated = {}

    for f in dataclasses.fields(elem):
        elem_f = getattr(elem, f.name)
        if elem_f is None:
            collated[f.name] = None
        elif torch.is_tensor(elem_f):
            collated[f.name] = tensor_collator([getattr(e, f.name) for e in batch])
        elif dataclasses.is_dataclass(elem_f):
            collated[f.name] = cat_dataclass(
                [getattr(e, f.name) for e in batch], tensor_collator
            )
        elif isinstance(elem_f, collections.abc.Mapping):
            collated[f.name] = {
                k: tensor_collator([getattr(e, f.name)[k] for e in batch])
                if elem_f[k] is not None
                else None
                for k in elem_f
            }
        else:
            raise ValueError("Unsupported field type for concatenation")

    return type(elem)(**collated)


class Timer:
    def __init__(self, name="timer", quiet=False):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if not self.quiet:
            print("%20s: %1.6f sec" % (self.name, self.interval))
