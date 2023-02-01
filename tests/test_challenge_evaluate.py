# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest
import numpy as np
import tempfile
import torch

from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.implicitron.dataset.json_index_dataset import FrameData
from pytorch3d.implicitron.evaluation.evaluate_new_view_synthesis import eval_batch
from pytorch3d.implicitron.models.base_model import ImplicitronRender

from co3d.challenge.io import (
    load_mask,
    store_mask,
    load_depth,
    store_depth,
    load_image,
    store_image,
    load_1bit_png_mask,
    store_1bit_png_mask,
    store_rgbda_frame,
    load_rgbda_frame,
)
from co3d.challenge.utils import get_result_directory_file_names, evaluate_file_folders
from co3d.challenge.metric_utils import eval_one
from co3d.challenge.data_types import RGBDAFrame


class TestIO(unittest.TestCase):
    def test_save_load(self):
        H = 100
        W = 200
        with tempfile.TemporaryDirectory() as tmpd:
            for data_type in ["image", "mask", "depth", "depth_mask"]:
                with self.subTest(data_type):
                    for _ in range(10):
                        C = {"depth_mask": 1, "mask": 1, "depth": 1, "image": 3}[data_type]
                        data = np.random.uniform(size=(C, H, W))
                        if data_type in ("mask", "depth_mask"):
                            data = (data > 0.5).astype(np.float32)
                        if C == 1:
                            data = data[0]
                        load_fun, store_fun = {
                            "mask": (load_mask, store_mask),
                            "depth": (load_depth, store_depth),
                            "image": (load_image, store_image),
                            "depth_mask": (load_1bit_png_mask, store_1bit_png_mask),
                        }[data_type]
                        fl = os.path.join(tmpd, f"{data_type}.png")
                        store_fun(data, fl)
                        data_ = load_fun(fl)
                        self.assertTrue(np.allclose(data, data_, atol=1 / 255))


class TestMetricUtils(unittest.TestCase):
    def test_against_eval_batch(self):
        H = 100
        W = 200
        for _ in range(20):
            implicitron_render = _random_implicitron_render(2, H, W, "cpu")
            
            for has_depth_mask in [True, False]:
            
                frame_data = _random_frame_data(2, H, W, "cpu")
                if not has_depth_mask:
                    frame_data.depth_mask = None

                eval_batch_result = eval_batch(
                    frame_data,
                    implicitron_render,
                )

                pred_rgbda = RGBDAFrame(
                    image=implicitron_render.image_render[0].numpy(),
                    mask=implicitron_render.mask_render[0].numpy(),
                    depth=implicitron_render.depth_render[0].numpy(),
                )

                gt_rgbda = RGBDAFrame(
                    image=frame_data.image_rgb[0].numpy(),
                    mask=frame_data.fg_probability[0].numpy(),
                    depth=frame_data.depth_map[0].numpy(),
                    depth_mask=frame_data.depth_mask[0].numpy() if has_depth_mask else None,
                )

                eval_one_result = eval_one(
                    pred=pred_rgbda,
                    target=gt_rgbda,
                )

                # print("eval_batch; eval_one")
                for k in ["iou", "psnr_fg", "psnr", "depth_abs_fg"]:
                    self.assertTrue(
                        np.allclose(eval_batch_result[k], eval_one_result[k], atol=1e-5)
                    )
                    # print(f"{k:15s}: {eval_batch_result[k]:1.3e} - {eval_one_result[k]:1.3e}")


class TestEvalScript(unittest.TestCase):
    def test_fake_data(self):
        N = 30
        H = 120
        W = 200
        with tempfile.TemporaryDirectory() as tmp_pred, tempfile.TemporaryDirectory() as tmp_gt:
            _generate_random_submission_data(tmp_pred, N, H, W)
            _generate_random_submission_data(tmp_gt, N, H, W)
            avg_result, per_example_result = evaluate_file_folders(tmp_pred, tmp_gt)
            metrics = list(avg_result.keys())
            for m in metrics:
                self.assertTrue(
                    np.allclose(
                        np.mean([r[m] for r in per_example_result]),
                        avg_result[m],
                    )
                )
            self.assertTrue(len(per_example_result) == N)


    def test_wrong_fake_data(self):
        N = 30
        H = 120
        W = 200

        # different number of eval/test examples
        for N_pred in [N - 2, N + 2]:
            with tempfile.TemporaryDirectory() as tmp_pred, tempfile.TemporaryDirectory() as tmp_gt:
                _generate_random_submission_data(tmp_pred, N_pred, H, W)
                _generate_random_submission_data(tmp_gt, N, H, W)
                msg = (
                    "Unexpected submitted evaluation examples"
                    if N_pred > N
                    else "There are missing evaluation examples"
                )
                with self.assertRaisesRegex(ValueError, msg):
                    evaluate_file_folders(tmp_pred, tmp_gt)

        # some eval examples missing depth/image
        with tempfile.TemporaryDirectory() as tmp_pred, tempfile.TemporaryDirectory() as tmp_gt:
            _generate_random_submission_data(tmp_pred, N_pred, H, W)
            _generate_random_submission_data(tmp_gt, N, H, W)
            pred_file_names = get_result_directory_file_names(tmp_pred)
            first_ex = pred_file_names[list(pred_file_names.keys())[0]]
            for file_type in ["depth", "image"]:
                os.remove(first_ex + f"_{file_type}.png")
                with self.assertRaisesRegex(
                    ValueError,
                    "Some evaluation examples are incomplete",
                ):
                    evaluate_file_folders(tmp_pred, tmp_gt)


def _generate_random_submission_data(folder, N, H, W):
    for example_num in range(N):
        root_path = os.path.join(folder, f"example_{example_num}")
        store_rgbda_frame(_random_rgbda_frame(H, W), root_path)


def _random_implicitron_render(
    N: int,
    H: int,
    W: int,
    device: torch.device,
):
    mask = _random_input_tensor(N, 1, H, W, True, device)
    return ImplicitronRender(
        depth_render=_random_input_tensor(N, 1, H, W, False, device),
        image_render=_random_input_tensor(N, 3, H, W, False, device) * mask,
        mask_render=mask,
    )


def _random_rgbda_frame(H: int, W: int):
    return RGBDAFrame(
        image=np.random.uniform(size=(3, H, W)).astype(np.float32),
        mask=(np.random.uniform(size=(1, H, W)) > 0.5).astype(np.float32),
        depth=np.random.uniform(size=(1, H, W)).astype(np.float32) + 0.1,
    )


def _random_frame_data(
    N: int,
    H: int,
    W: int,
    device: torch.device,
):
    R, T = look_at_view_transform(azim=torch.rand(N) * 360)
    cameras = PerspectiveCameras(R=R, T=T, device=device)
    depth_map_common = (
        torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, H),
                torch.linspace(0.0, 1.0, W),
            )
        ).mean(dim=0)
        + 0.1
    )
    depth_map = _random_input_tensor(N, 1, H, W, False, device) + depth_map_common[None]
    random_args = {
        "frame_number": torch.arange(N),
        "frame_timestamp": torch.linspace(0.0, 1.0, N),
        "sequence_category": ["random"] * N,
        "camera": cameras,
        "fg_probability": _random_input_tensor(N, 1, H, W, True, device),
        "depth_map": depth_map,
        "mask_crop": torch.ones(N, 1, H, W, device=device),
        "depth_mask": _random_input_tensor(N, 1, H, W, True, device),
        "sequence_name": ["sequence"] * N,
        "image_rgb": _random_input_tensor(N, 3, H, W, False, device),
        "frame_type": ["test_unseen", *(["test_known"] * (N - 1))],
    }
    return FrameData(**random_args)


def _random_input_tensor(
    N: int,
    C: int,
    H: int,
    W: int,
    is_binary: bool,
    device: torch.device,
) -> torch.Tensor:
    T = torch.rand(N, C, H, W, device=device)
    if is_binary:
        T = (T > 0.5).float()
    return T


if __name__ == "__main__":
    unittest.main()
