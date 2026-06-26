# coding=utf-8
"""Shared decoder utilities."""

# ===== helpers.py =====
from argparse import Namespace
import collections.abc
import importlib.metadata as importlib_metadata
import importlib.util
import json
import logging
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils.import_utils import compare_versions
from easydict import EasyDict
from loguru import logger
from packaging.version import parse
from PIL import Image
from torch.distributed import ProcessGroup
from torch.utils.data import default_collate
from torchvision import transforms


def default(value, default_val):
    return default_val if value is None else value


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def readable_time(src, key=None, left_steps=None):
    """ Convert time seconds to a readable format: DD Days, HH Hours, MM Minutes, SS Seconds """
    if hasattr(src, "timers"):
        assert key is not None, "key must be provided when src is a Timer object"
        time_repr = (
            f"Average: {readable_time(src.average(key))}"
            f" | Elapsed: {readable_time(src.elapsed_total(key))}"
        )
        if left_steps is not None:
            assert isinstance(left_steps, int), "left_steps must be int"
            time_repr += f" | Remain: {readable_time(src.average(key) * left_steps)}"
        return time_repr

    seconds = int(src)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days} Days, {hours} Hours, {minutes} Minutes, {seconds} Seconds"
    if hours > 0:
        return f"{hours} Hours, {minutes} Minutes, {seconds} Seconds"
    if minutes > 0:
        return f"{minutes} Minutes, {seconds} Seconds"
    return f"{seconds} Seconds"


def print_args(title, args):
    print(f'------------------------ {title} ------------------------', flush=True)
    str_list = []
    name_max_length = max([48] + [len(arg) + 3 for arg in vars(args)])
    for arg in vars(args):
        dots = '.' * (name_max_length - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print(f'-------------------- end of {title} ---------------------', flush=True)


@dataclass
class DataClassMixin:

    def __repr__(self):
        max_key_length = max([len(k) for k in asdict(self).keys()])
        return "\n".join(
            [f"{k:>{max_key_length}}: {v}" for k, v in asdict(self).items()]
        )

    def to_dict(self):
        return asdict(self)

# ===== file_utils.py =====
def safe_file(path):
    """Create the parent directory of a file if it does not exist."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def safe_save_file(save_path, *args, save_fn=None, **kwargs):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_save_path = save_path.parent / f"temp_{save_path.name}"
    save_to = None
    try:
        save_to = save_fn(tmp_save_path, *args, **kwargs)
        shutil.copyfile(tmp_save_path, save_path)
        save_to = save_path
        tmp_save_path.unlink()
    except Exception as e:
        print(f"Failed to save to {save_path}. {type(e)}: {e}")
    return save_to


def empty_logger():
    logger = logging.getLogger("rosetta_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def save_to_csv(dataframe, save_path, append=False):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if append:
        dataframe.to_csv(save_path, index=False, mode='a', header=not save_path.exists())
    else:
        dataframe.to_csv(save_path, index=False)


def save_to_json(save_path, results, indent=4):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=indent)
    return save_path

def is_package_version(package: str, operation: str, version: str):
    """
    Compares the current Accelerate version to a given reference with an operation.

    Args:
        package (str): The package name to check.
        operation (str): A string representation of an operator, such as `">"` or `"<="`
        version (str): A version string
    """
    package_available = importlib.util.find_spec(package.replace("-", "_")) is not None
    if not package_available:
        return False

    package_version = importlib_metadata.version(package)

    return compare_versions(parse(package_version), operation, version)


# ===== torch_utils.py =====
PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
}

def set_manual_seed(global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)


def set_reproducibility(enable, global_seed=None, benchmark=None):
    if enable:
        set_manual_seed(global_seed)
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def except_collate_fn(batch, except_keys=None):
    if except_keys is None:
        return default_collate(batch)

    ret = {}
    existing_except_keys = []
    for expect_key in except_keys:
        if expect_key in batch[0]:
            existing_except_keys.append(expect_key)

    except_keys = existing_except_keys

    for expect_key in except_keys:
        ret[expect_key] = []

    for batch_index in range(len(batch)):
        for except_key in except_keys:
            ret[except_key].append(batch[batch_index][except_key])
            del batch[batch_index][except_key]

    ret.update(default_collate(batch))

    return ret


class Timer(object):
    def __init__(self, enabled=False):
        self.timers = {}
        self.enabled = enabled

    def __getitem__(self, key):
        return self.timers[key]

    def start(self, name, sync=False, barrier=False):
        if not self.enabled:
            return
        if name not in self.timers:
            self.timers[name] = {"total_time": 0.0, "count": 0, "sync": sync, "barrier": barrier}
        else:
            if "start_time" in self.timers[name]:
                raise ValueError(f"Timer '{name}' is already running.")
            self.timers[name].update({"sync": sync, "barrier": barrier})
        if sync:
            torch.cuda.synchronize()
        if barrier:
            dist.barrier()
        self.timers[name]["start_time"] = time.time()

    def stop(self, name):
        if not self.enabled:
            return
        if name not in self.timers or "start_time" not in self.timers[name]:
            raise ValueError(f"Timer '{name}' was not started.")
        if self.timers[name]["sync"]:
            torch.cuda.synchronize()
        if self.timers[name]["barrier"]:
            dist.barrier()
        elapsed_time = time.time() - self.timers[name]["start_time"]
        self.timers[name]["elapsed_time"] = elapsed_time
        self.timers[name]["total_time"] += elapsed_time
        self.timers[name]["count"] += 1
        del self.timers[name]["start_time"]

    def elapsed(self, name):
        if not self.enabled:
            return 0.0
        if name not in self.timers or "elapsed_time" not in self.timers[name]:
            raise ValueError(f"Timer '{name}' has no recorded elapsed time.")
        return self.timers[name]["elapsed_time"]

    def elapsed_total(self, name):
        if not self.enabled:
            return 0.0
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' has no recorded times.")
        return self.timers[name]["total_time"]

    def average(self, name):
        if not self.enabled:
            return 0.0
        if name not in self.timers or self.timers[name]["count"] == 0:
            raise ValueError(f"Timer '{name}' has no recorded times.")
        return self.timers[name]["total_time"] / self.timers[name]["count"]

# ===== image_base.py =====
class ImageInfo:
    """ Class to store image information for processing and generation. """
    args: EasyDict | dict

    def __init__(
            self,
            image_type: str = None,
            image_tensor: torch.Tensor = None,
            image_width: int = None,
            image_height: int = None,
            token_width: int = None,
            token_height: int = None,
            image_token_length: int = None,
            base_size: int = None,
            ratio_index: int = None,
            face_image: Optional[Image.Image] = None,
            ori_image_width: int = None,
            ori_image_height: int = None,
    ):
        if self.args is None:
            raise ValueError("ImageInfo requires `args` attribute to be set.")

        self.image_type = image_type
        self.image_tensor = image_tensor
        self.ori_image_width = ori_image_width
        self.image_width = image_width
        self.w = image_width
        self.ori_image_height = ori_image_height
        self.image_height = image_height
        self.h = image_height
        self.token_width = token_width
        self.tk_w = token_width
        self.token_height = token_height
        self.tk_h = token_height
        self.image_token_length = default(
            image_token_length,
            token_width * token_height if (token_width is not None and token_height is not None) else None
        )
        self.base_size = base_size
        self.ratio_index = ratio_index
        self.face_image = face_image

        # args
        self.add_timestep_token = self.args.get("add_timestep_token", False)
        self.add_image_shape_token = self.args.get("add_image_shape_token", False)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to attributes."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __contains__(self, key: str) -> bool:
        """Check if the key exists in the ImageInfo object."""
        return hasattr(self, key)

    def __repr__(self):
        return (f"ImageInfo(image_type={self.image_type}, image_tensor={self.image_tensor}, "
                f"ori_image_width={self.ori_image_width}, ori_image_height={self.ori_image_height}, "
                f"image_width={self.image_width}, image_height={self.image_height}, "
                f"token_width={self.token_width}, token_height={self.token_height}, "
                f"image_token_length={self.image_token_length}, "
                f"base_size={self.base_size}, ratio_index={self.ratio_index}, face_image={self.face_image})")

    @property
    def meta_info(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        if self.image_type in ["vae", "src_image", "gen_image"]:
            return dict(
                token_length=self.image_token_length,
                add_timestep_token=self.add_timestep_token,
                add_image_shape_token=self.add_image_shape_token,
                base_size=self.base_size,
                ratio_idx=self.ratio_index,
                token_height=self.token_height,
                token_width=self.token_width,
                image_height=self.image_height,
                image_width=self.image_width,
                ori_image_width=self.ori_image_width,
                ori_image_height=self.ori_image_height,
            )
        elif self.image_type in ["vision_encoder", "und_image", "qwen3vl"]:
            return dict(
                token_length=self.image_token_length,
                add_image_shape_token=self.add_image_shape_token,
                token_height=self.token_height,
                token_width=self.token_width,
                image_height=self.image_height,
                image_width=self.image_width,
                ori_image_width=self.ori_image_width,
                ori_image_height=self.ori_image_height,
            )
        elif self.image_type == "face":
            return dict(
                token_length=self.image_token_length,
            )
        else:
            raise ValueError(f"Unknown image type '{self.image_type}'")

    def copy(self, copy_image_tensor=True):
        if copy_image_tensor and self.image_tensor is None:
            raise ValueError("image_tensor is None, cannot copy")
        return ImageInfo(
            image_type=self.image_type,
            image_tensor=self.image_tensor.clone() if copy_image_tensor else None,
            image_width=self.image_width,
            image_height=self.image_height,
            ori_image_width=self.ori_image_width,
            ori_image_height=self.ori_image_height,
            token_width=self.token_width,
            token_height=self.token_height,
            image_token_length=self.image_token_length,
            base_size=self.base_size,
            ratio_index=self.ratio_index,
            face_image=self.face_image,     # shared
        )

class ImageTensor(torch.Tensor):
    # This class is just for type hinting purposes. Attribute `i` should be defined
    # as an instance attribute of the torch.Tensor instance, like: tensor.i = ImageInfo(...)
    i: ImageInfo
    section_type: str
    vision_encoder_kwargs: dict[str, torch.Tensor]


class JointImageInfo(object):
    def __init__(self, vae_image_info: ImageInfo, vision_image_info: ImageInfo, vision_encoder_kwargs: dict = None):
        self.vae_image_info = vae_image_info
        self.vision_image_info = vision_image_info
        self.vision_encoder_kwargs = vision_encoder_kwargs

        # Define key attributes to align with ImageInfo for uniformity
        self.image_type = "joint_image"
        self.image_token_length = vae_image_info.image_token_length + vision_image_info.image_token_length

        self.add_timestep_token = vae_image_info.add_timestep_token
        self.add_image_shape_token = vae_image_info.add_image_shape_token

    def __repr__(self):
        return f"JointImageInfo(vae_image={self.vae_image_info}, vision_image={self.vision_image_info})"

    @property
    def meta_info(self):
        # Used for image sections of tkwrapper.encode_general()
        return dict(
            token_length=[self.vae_image_info.image_token_length, self.vision_image_info.image_token_length],
            add_timestep_token=self.add_timestep_token,
            add_image_shape_token=self.add_image_shape_token,
            base_size=self.vae_image_info.base_size,
            ratio_idx=self.vae_image_info.ratio_index,
            token_height=[self.vae_image_info.token_height, self.vision_image_info.token_height],
            token_width=[self.vae_image_info.token_width, self.vision_image_info.token_width],
            image_height=[self.vae_image_info.image_height, self.vision_image_info.image_height],
            image_width=[self.vae_image_info.image_width, self.vision_image_info.image_width],
        )

    def copy(self, copy_image_tensor=True):
        if copy_image_tensor and (self.vae_image_info.image_tensor is None or self.vision_image_info.image_tensor is None):
            raise ValueError("image_tensor is None, cannot copy")
        return JointImageInfo(self.vae_image_info.copy(copy_image_tensor), self.vision_image_info.copy(copy_image_tensor), self.vision_encoder_kwargs)

class CondImage(object):
    def __init__(self, image_type: str, vae_image: ImageTensor, vit_image: ImageTensor):
        self.image_type = image_type
        self.vae_image = vae_image
        self.vit_image = vit_image

        if image_type == "vae":
            self.i = vae_image.i
            self.section_type = "cond_vae_image"

        elif image_type == "vit":
            self.i = vit_image.i
            self.section_type = "cond_vit_image"

        elif image_type == "vae_vit":
            self.i = JointImageInfo(vae_image.i, vit_image.i)
            self.section_type = "cond_joint_image"

        else:
            raise ValueError(f"Unknown image_type: {image_type}")


# Runtime state helpers
"""Runtime helpers shared by training and inference."""

# Global runtime state
# args
_GLOBAL_ARGS: Optional[Namespace] = None

# parallel state
_GLOBAL_PARALLEL_STATE = None

# logger
_GLOBAL_LOGGER = None

# Global Switch
_ensure_initialized: bool = True
_ensure_not_initialized: bool = True


def _ensure_var_is_initialized(var, name):
    if not _ensure_initialized:
        return
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    if not _ensure_not_initialized:
        return
    assert var is None, '{} is already initialized.'.format(name)


def set_args(args):
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = args


def get_args():
    global _GLOBAL_ARGS
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def set_parallel_state(parallel_state):
    global _GLOBAL_PARALLEL_STATE
    _ensure_var_is_not_initialized(_GLOBAL_PARALLEL_STATE, "parallel_state")
    _GLOBAL_PARALLEL_STATE = parallel_state


def get_parallel_state():
    global _GLOBAL_PARALLEL_STATE
    _ensure_var_is_initialized(_GLOBAL_PARALLEL_STATE, "parallel_state")
    return _GLOBAL_PARALLEL_STATE


def set_logger(logger):
    global _GLOBAL_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_LOGGER, 'logger')
    _GLOBAL_LOGGER = logger


def get_logger():
    global _GLOBAL_LOGGER
    _ensure_var_is_initialized(_GLOBAL_LOGGER, 'logger')
    return _GLOBAL_LOGGER



# Parallel state
@dataclass
class ParallelState:
    # Data parallel
    dp_rank: int
    dp_size: int
    dp_group: ProcessGroup = None
    # Tensor parallel
    tp_rank: int = 0
    tp_size: int = 1
    # Pipeline parallel
    pp_rank: int = 0
    pp_size: int = 1
    # Expert parallel
    ep_rank: int = 0
    ep_size: int = 1
    # Context parallel
    cp_rank: int = 0
    cp_size: int = 1

    def __post_init__(self):
        set_parallel_state(self)

    def __repr__(self):
        res = []
        res.append(f"dp: {self.dp_rank}/{self.dp_size}")
        res.append(f"tp: {self.tp_rank}/{self.tp_size}")
        res.append(f"pp: {self.pp_rank}/{self.pp_size}")
        res.append(f"cp: {self.cp_rank}/{self.cp_size}")
        return ", ".join(res)

    @staticmethod
    def from_pure_torch():
        """Initialize ParallelState from native torch.distributed.

        For inference with pure data parallelism: each rank holds the full model,
        dp_rank = global rank, dp_size = world_size.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return ParallelState(
            dp_rank=rank,
            dp_size=world_size,
            dp_group=None,
        )

    def get_tensor_and_data_parallel_group(self, with_context_parallel: bool = False):
        return dist.GroupMember.WORLD
