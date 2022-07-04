from typing import Enum
from numpy import np
from dataclasses import dataclass
from PIL import Image


@dataclass
class RGBDAFrame:
    image: np.ndarray
    mask: np.ndarray
    depth: np.ndarray


class CO3DTask(Enum):
	MANY_VIEW="many_view"
	FEW_VIEW="few_view"


class CO3DSequenceSet(Enum):
    TRAIN="train"
    DEV="dev"
    TEST="test"


def store_rgbda_frame(rgbda_frame: RGBDAFrame, fl: str):
    store_mask(rgbda_frame.mask[0], fl + "_mask.png")
    store_depth(rgbda_frame.depth[0], fl + "_depth.png")
    store_image(rgbda_frame.image, fl + "_image.png")
    

def load_rgbda_frame(fl: str) -> RGBDAFrame:
    return RGBDAFrame(
        mask=load_mask(fl + "_mask.png")[None],
        depth=load_depth(fl + "_depth.png")[None],
        image=load_image(fl + "_image.png"),
    )
    

def load_mask(fl: str):
    return np.array(Image.open(fl)).astype(np.float32) / 255.0


def store_mask(mask: np.ndarray, fl: str, mode: str = "L"):
    """
    mask: HxW
    """
    assert mask.ndim==2
    if mode == "L":
        mpil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").convert("L")
    elif mode == "I;16":
        mpil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="I;16").convert(
            "I;16"
        )
    else:
        raise ValueError(mode)
    mpil.save(fl, "PNG")


def load_depth(fl: str):
    depth_pil = Image.open(fl)
    depth = (
        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((depth_pil.size[1], depth_pil.size[0]))
    )
    assert depth.ndim==2
    return depth


def store_depth(depth: np.ndarray, fl: str):
    assert depth.ndim==2
    depth_uint16 = np.frombuffer(depth.astype(np.float16), dtype=np.uint16).reshape(
        depth.shape
    )
    Image.fromarray(depth_uint16).save(fl)


def load_image(fl: str):
    return np.array(Image.open(fl)).astype(np.float32).transpose(2, 0, 1) / 255.0


def store_image(image: np.ndarray, fl: str):
    assert image.ndim==3
    Image.fromarray((image.transpose(1, 2, 0) * 255.0).astype(np.uint8)).save(fl)
    

