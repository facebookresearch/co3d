# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


def make_depth_image(
    depths, 
    masks, 
    max_quantile=0.98, 
    min_quantile=0.02,
    min_out_depth=0.1,
    max_out_depth=0.9,
):
    normfacs = []
    for d, m in zip(depths, masks):
        ok = (d.view(-1) > 1e-6) * (m.view(-1) > 0.5)
        if ok.sum() <= 1:
            print('empty depth!')
            normfacs.append(torch.zeros(2).type_as(depths)) 
            continue
        dok = d.view(-1)[ok].view(-1)
        _maxk = max(int(round((1-max_quantile) * (dok.numel()))),1)
        _mink = max(int(round(min_quantile * (dok.numel()))),1)
        normfac_max = dok.topk(k=_maxk, dim=-1).values[-1]
        normfac_min = dok.topk(k=_mink, dim=-1, largest=False).values[-1]        
        normfacs.append(torch.stack([normfac_min, normfac_max]))
    normfacs = torch.stack(normfacs)
    _min, _max = (
        normfacs[:, 0].view(-1, 1, 1, 1), normfacs[:, 1].view(-1, 1, 1, 1)
    )
    depths = (depths - _min) / (_max - _min).clamp(1e-4)
    depths = (
        (depths * (max_out_depth - min_out_depth) + min_out_depth) 
        * masks.float()
    ).clamp(0.0, 1.0)
    return depths
