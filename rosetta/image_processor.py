import random
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.integrate import quad
from scipy.optimize import fsolve
from transformers import BaseImageProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.image_utils import load_image

from rosetta.autoencoder import VAE_META_INFO
from rosetta.visual_encoder import VISION_ENCODER_META_INFO, load_vit_processor
from rosetta.utils import ImageTensor, ImageInfo, CondImage
from rosetta.utils import DataClassMixin

InputImage = Image.Image | str
IMAGE_INPUT_TYPES = (Image.Image, str)


class SliceVocabLogitsWarper:
    def __init__(self, vocab_start: int = None, vocab_end: int = None):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, f"Ensure vocab_start {vocab_start} < {vocab_end}"
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores[:, self.vocab_start: self.vocab_end]

    def __repr__(self):
        return (
            f"SliceVocabLogitsWarper(vocab_start={self.vocab_start}, "
            f"vocab_end={self.vocab_end})"
        )


class DataMixin:
    enable_crypto: bool = False
    cos_base = None

    @staticmethod
    def require_configs(obj, required, obj_name, do_assert=True):
        if isinstance(required, str):
            required = [required]

        # Use tuple for alternatives
        if isinstance(required, tuple):
            passed = DataMixin.require_configs(obj, required[0], None, do_assert=False)
            if not passed:
                for alt_required in required[1:]:
                    passed = DataMixin.require_configs(obj, alt_required, None, do_assert=False)
                    if passed:
                        break
                else:
                    raise KeyError(f"One of {required} is required for {obj_name}.")
            return passed

        else:
            missing_keys = []
            if isinstance(obj, (dict, list, tuple, set)):
                for key in required:
                    if key not in obj:
                        missing_keys.append(key)
            else:
                for key in required:
                    if not hasattr(obj, key) or getattr(obj, key) is None:
                        missing_keys.append(key)
            if do_assert and len(missing_keys) > 0:
                raise KeyError(f"[{', '.join(missing_keys)}] is required for {obj_name}.")
            return len(missing_keys) == 0

ResampleType = dict(
    bilinear=Image.Resampling.BILINEAR,
    bicubic=Image.Resampling.BICUBIC,
    lanczos=Image.Resampling.LANCZOS,
)


class Resolution:
    def __init__(self, height: int, width: int):
        self.h = self.height = height
        self.w = self.width = width
        self.ratio = height / width


class ResolutionGroup:
    def __init__(
            self,
            base_size: int = None,
            step: Optional[int] = None,
            align: int = 16,
            mode: Optional[str] = None,
            preset: Optional[str] = None,
            num_buckets: Optional[int] = None,
            **_,
    ):
        if base_size is None:
            raise ValueError("base_size is required.")
        if base_size % align != 0:
            raise ValueError(f"base_size {base_size} is not divisible by align {align}.")
        if preset is not None and mode is not None:
            raise ValueError("preset and mode cannot be set at the same time.")
        if preset is not None:
            if preset == "sdxl":
                mode = "sdxl"
                step = base_size // 16
            elif preset == "arc33":
                mode = "arc"
                num_buckets = 33
            else:
                raise ValueError(f"preset {preset} is not supported.")
        elif mode is None:
            mode = "sdxl"
        if mode == "sdxl" and step is None:
            step = base_size // 16
        if mode == "arc" and num_buckets is None:
            raise ValueError("num_buckets must be specified for arc mode.")
        if mode != "arc" and num_buckets is not None:
            raise ValueError(f"The `{mode}` mode does not support num_buckets.")
        if step is not None:
            if align > step:
                raise ValueError(f"align {align} must be no larger than step {step}.")
            if step > base_size // 2:
                raise ValueError(f"step must be no larger than base_size // 2, got {step}.")

        self.base_size = base_size
        self.step = step
        self.align = align
        self.mode = mode
        self.preset = preset
        self.num_buckets = num_buckets
        self.data = self._calc()
        self.ratio = np.array([reso.ratio for reso in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _align_size(self, size: int) -> int:
        return size // self.align * self.align

    def _calc(self):
        if self.mode == "sdxl":
            data = self._calc_by_step()
        elif self.mode == "arc":
            data = self._calc_by_arc(self.num_buckets)
        else:
            raise ValueError(f"mode {self.mode} is not supported.")

        return sorted(data, key=lambda reso: reso.ratio)

    def _calc_by_step(self):
        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while cur_height < max_height or cur_width > min_width:
            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(self._align_size(cur_height), self._align_size(cur_width)))

        cur_height, cur_width = self.base_size, self.base_size
        while cur_height > min_height or cur_width < max_width:
            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(self._align_size(cur_height), self._align_size(cur_width)))

        return sorted(resolutions, key=lambda reso: reso.ratio)

    def _calc_by_arc(self, n: int):
        if n % 2 != 1:
            raise ValueError(f"n {n} must be odd.")

        a = self.base_size // 2 // self.align
        b = self.base_size * 2 // self.align

        def integrand(u):
            return np.sqrt(np.cosh(2 * u))

        def integral(t):
            result, _ = quad(integrand, 0, t)
            return result

        def equation(t, target):
            return integral(t) - target

        t0 = 0.5 * np.log(b / a)
        full_integral = integral(t0)
        segment = 2 * full_integral / (n - 1)

        half_ts = []
        for i in range(1, n // 2):
            target = segment * i
            half_ts.extend(fsolve(equation, 1, args=(target,)))
        ts = [t0] + half_ts[::-1] + [0.0] + [-t for t in half_ts] + [-t0]

        resolutions = []
        for t in ts:
            width = np.sqrt(a * b) * np.exp(t)
            height = np.sqrt(a * b) * np.exp(-t)
            resolutions.append(Resolution(int(height) * self.align, int(width) * self.align))
        return resolutions

    def _closest_ratio_index(self, width: int, height: int):
        ratio = height / width
        return int(np.argmin(np.abs(self.ratio - ratio)))

    def get_target_size(self, width: int, height: int):
        reso = self.data[self._closest_ratio_index(width, height)]
        return reso.width, reso.height

    def get_base_size_and_ratio_index(self, width: int, height: int):
        return self.base_size, self._closest_ratio_index(width, height)


def resize_and_crop(
        image,
        target_size,
        crop_type='center',
        resample=Image.Resampling.BICUBIC,
):
    target_width, target_height = target_size
    width, height = image.size
    target_ratio = target_height / target_width
    ratio = height / width

    if crop_type == "resize":
        resized_image = image.resize((target_width, target_height), resample=resample)
        return resized_image, (0, 0)

    if ratio < target_ratio:
        resize_height = target_height
        resize_width = int(round(target_height / height * width))
    else:
        resize_width = target_width
        resize_height = int(round(target_width / width * height))

    if crop_type == 'center':
        crop_top = int(round((resize_height - target_height) / 2.0))
        crop_left = int(round((resize_width - target_width) / 2.0))
    elif crop_type == 'random':
        crop_top = random.randint(0, resize_height - target_height)
        crop_left = random.randint(0, resize_width - target_width)
    else:
        raise ValueError(f'crop_type must be center, random or resize, but got {crop_type}')

    resized_image = image.resize((resize_width, resize_height), resample=resample)
    resized_image = resized_image.crop(
        (crop_left, crop_top, crop_left + target_width, crop_top + target_height)
    )
    return resized_image, (crop_left, crop_top)


@dataclass
class ResolutionGroupConfig(DataClassMixin):
    base_size: int = None
    align: int = 16
    preset: Optional[str] = None

    @classmethod
    def from_args(cls, args, **kwargs):
        config = dict(
            base_size=kwargs.get("base_size", args.reso_base_size),
            align=kwargs.get("align", args.reso_align),
            preset=kwargs.get("preset", args.reso_preset),
        )
        return cls(**config)


@dataclass
class VAEInfo:
    encoder_type: str
    down_h_factor: int = -1
    down_w_factor: int = -1
    h_factor: int = -1
    w_factor: int = -1
    image_type: str = None

    def __post_init__(self):
        self.h_factor = self.down_h_factor
        self.w_factor = self.down_w_factor
        if self.image_type is None:
            self.image_type = "vae"


@dataclass
class ViTInfo:
    encoder_type: str
    h_factor: int = -1
    w_factor: int = -1
    max_token_length: int = 0   # pad to max_token_length
    processor: Callable = field(default_factory=BaseImageProcessor)
    image_type: str = None

    def __post_init__(self):
        if self.image_type is None:
            self.image_type = self.encoder_type.split("-")[0]


class ImageMixin(DataMixin):
    task_kwargs: dict
    index_kwargs: dict
    modality: list[str]
    vae_info: VAEInfo
    vit_info: ViTInfo

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tensor_to_pil_image = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
        )

    def setup_image(self, args):
        ImageInfo.args = dict(
            add_timestep_token=args.add_timestep_token,
            add_image_shape_token=args.add_image_shape_token,
        )
        self.cond_image_section_type = "cond_joint_image"

        if "vae_image" in self.modality:
            self.require_configs(args, ["vae_type", "vae_image_token_length"], "vae_image modality")

            self.vae_image_token_length = self.task_kwargs.get("vae_image_token_length", args.vae_image_token_length)
            self.reso_base_size = args.reso_base_size
            self.reso_group_config = ResolutionGroupConfig.from_args(
                args, **self.index_kwargs.get("reso_bucket_kwargs", {})
            )
            if hasattr(self, "index_manager") and (self.index_kwargs.get("online_bucketing") or not self.index_kwargs.get("multireso", False)):
                self.index_manager.set_resolution_buckets(**self.reso_group_config.to_dict())
            self.vae_reso_group = ResolutionGroup(**self.reso_group_config.to_dict())
            vae_meta_info = VAE_META_INFO[args.vae_type]
            downsample_factor = vae_meta_info["downsample_factor"]
            self.vae_info = VAEInfo(
                encoder_type=args.vae_type,
                down_h_factor=downsample_factor[0], down_w_factor=downsample_factor[1],
            )

        if "vit_image" in self.modality:
            self.require_configs(args, ["vit_type", "vit_image_token_length"], "vit_image modality")

            self.vit_image_token_length = self.task_kwargs.get("vit_image_token_length", args.vit_image_token_length)
            self.min_vit_image_token_length = self.task_kwargs.get("min_vit_image_token_length", args.min_vit_image_token_length)
            if self.min_vit_image_token_length is None:
                self.min_vit_image_token_length = 256
            processor = load_vit_processor(
                args.vit_type,
                min_pixels=self.min_vit_image_token_length * 32 * 32,
                max_pixels=self.vit_image_token_length * 32 * 32,
            )

            self.vit_info = ViTInfo(
                encoder_type=args.vit_type,
                h_factor=processor.patch_size,
                w_factor=processor.patch_size,
                max_token_length=self.vit_image_token_length,
                processor=processor,
            )

        self.uncond_p = self.task_kwargs.get('uncond_p', 0.0)

    def as_image_tensor(self, image, image_type, **kwargs) -> ImageTensor:
        if isinstance(image, Image.Image):
            tensor = self.pil_image_to_tensor(image)
        else:
            tensor = image

        origin_size = kwargs["origin_size"]
        ori_image_width = origin_size[0]
        ori_image_height = origin_size[1]

        if image_type == "vae":
            assert tensor.ndim == 3 or tensor.ndim == 4
            h, w = tensor.shape[-2], tensor.shape[-1]
            assert (h % self.vae_info.h_factor == 0 and w % self.vae_info.w_factor == 0), \
                (f"Image size should be divisible by ({self.vae_info.h_factor}, {self.vae_info.w_factor}), "
                 f"but got ({h} x {w}).")
            tk_height = h // self.vae_info.h_factor
            tk_width = w // self.vae_info.w_factor
            base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(w, h)
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=w, image_height=h, token_width=tk_width, token_height=tk_height,
                base_size=base_size, ratio_index=ratio_idx,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.section_type = "cond_vae_image"
        elif image_type  == "qwen3vl":
            encoder_meta = VISION_ENCODER_META_INFO.get(self.vit_info.encoder_type, {})
            spatial_merge_size = encoder_meta.get("spatial_merge_size", 2)

            grid_height, grid_width = kwargs["image_grid_thw"][1].item(), kwargs["image_grid_thw"][2].item()
            token_height, token_width = grid_height // spatial_merge_size, grid_width // spatial_merge_size
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=grid_height * self.vit_info.w_factor,
                image_height=grid_width * self.vit_info.h_factor,
                token_width=token_width,
                token_height=token_height,
                image_token_length=token_width * token_height,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.section_type = "cond_vit_image"
            tensor.vision_encoder_kwargs = {
                "grid_thw": kwargs["image_grid_thw"],
            }
        else:
            raise ValueError(f"Unknown image type: {image_type}")
        return tensor

    def crop(self, image, target_size):
        tw, th = target_size
        w, h = image.size

        crop_top = int(round((h - th) / 2.0))
        crop_left = int(round((w - tw) / 2.0))
        image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

        return image, (crop_left, crop_top)

    def vae_process_image(self, image, target_size, random_crop: bool | str = False) -> ImageTensor:
        origin_size = image.size
        crop_type = random_crop if isinstance(random_crop, str) else ("random" if random_crop else "center")
        if crop_type == "center_and_no_resize":
            resized_image, _ = self.crop(image, target_size)
        else:
            resized_image, _ = resize_and_crop(
                image, target_size, crop_type=crop_type, resample=ResampleType["bicubic"]
            )
        return self.as_image_tensor(resized_image, image_type=self.vae_info.image_type, origin_size=origin_size)

    def vit_process_image(self, image) -> ImageTensor:
        if not hasattr(self, "vit_info"):
            raise ValueError("'vit_info' is not defined. Please check if 'vit_image' is in 'modality'.")

        origin_size = image.size
        inputs = self.vit_info.processor(image)
        image = inputs["pixel_values"].squeeze(0)   # (C, H, W)

        remain_keys = set(inputs.keys()) - {"pixel_values"}
        remain_kwargs = {}
        for key in remain_keys:
            if isinstance(inputs[key], torch.Tensor):
                remain_kwargs[key] = inputs[key].squeeze(0)
            else:
                remain_kwargs[key] = inputs[key]

        return self.as_image_tensor(image, image_type=self.vit_info.image_type, origin_size=origin_size, **remain_kwargs)

    def get_image_with_size(
            self,
            src: InputImage,
            random_crop: bool | str = False,
            target_size_type: str = "image",
            return_type: str = "vae",
            **kwargs,
    ) -> tuple[ImageTensor | CondImage, bool]:
        assert isinstance(src, IMAGE_INPUT_TYPES), \
            f"`src` must be a PIL.Image or a string path/URL, got {type(src)}."
        image = load_image(src)
        image_flag = "normal"
        img_success = image_flag != "gray"
        origin_size = image.size

        if "vae" in return_type:
            if target_size_type == "index":
                target_size = self.index_manager.get_target_size(src)  # (w_tgt, h_tgt)
            elif target_size_type == "image":
                target_size = self.vae_reso_group.get_target_size(*origin_size)
            else:
                target_size = (self.reso_base_size, self.reso_base_size)
            vae_image_tensor = self.vae_process_image(image, target_size, random_crop=random_crop)
        else:
            vae_image_tensor = None

        if "vit" in return_type:
            vit_image_tensor = self.vit_process_image(image)
        else:
            vit_image_tensor = None

        if return_type == "vae":
            image_tensor = vae_image_tensor
        elif return_type == "vit":
            image_tensor = vit_image_tensor
        elif return_type == "vae_vit":
            image_tensor = CondImage(image_type=return_type, vae_image=vae_image_tensor, vit_image=vit_image_tensor)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

        return image_tensor, img_success

    def prepare_full_attn_slices(self, output, batch_idx=None, with_gen=True):
        if not hasattr(self, "cond_image_section_type"):
            return []

        slices = output.vae_image_slices[batch_idx] if batch_idx is not None else output.vae_image_slices

        if with_gen:
            gen_image_slices = (
                output.gen_image_slices[batch_idx]
                if batch_idx is not None
                else output.gen_image_slices
            )
            slices = slices + gen_image_slices
        return slices


class ImageProcessor(ImageMixin):
    def __init__(self, args: Namespace):
        super().__init__()
        self.modality = args.modality
        self.img_ratio_slice_logits_processor = None
        self.task_kwargs = {}
        self.index_kwargs = {}
        self.setup_image(args)

    def build_gen_image_info(self, image_size) -> ImageInfo:
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif 'x' in image_size:
                image_size = [int(s) for s in image_size.split('x')]
            elif ':' in image_size:
                image_size = [int(s) for s in image_size.split(':')]
                assert len(image_size) == 2, f"`image_size` should be in the format of 'W:H', got {image_size}."
                image_size = [image_size[1], image_size[0]]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'W:H' or <img_ratio_i>, got {image_size}.")
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_size}."
        else:
            raise ValueError(f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                             f"got {image_size}.")
        image_width, image_height = self.vae_reso_group.get_target_size(image_size[1], image_size[0])
        token_height = image_height // self.vae_info.h_factor
        token_width = image_width // self.vae_info.w_factor
        base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
        image_info = ImageInfo(
            image_type="gen_image", image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height, base_size=base_size, ratio_index=ratio_idx,
        )
        return image_info

    def build_cond_images(
            self,
            image_list: Optional[list[InputImage]] = None,
            message_list: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[list[CondImage | ImageTensor]]:
        if image_list is not None and message_list is not None:
            raise ValueError("`image_list` and `message_list` cannot be provided at the same time.")
        if message_list is not None:
            image_list = []
            for message in message_list:
                visuals = [
                    content
                    for content in message["content"]
                    if isinstance(content, dict) and content["type"] in ["image"]
                ]
                image_list.extend([
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["image", "url", "path", "base64"]
                    if key in vision_info and vision_info["type"] == "image"
                ])

        return [
            self.get_image_with_size(
                src, target_size_type="image", random_crop="center", return_type="vae_vit",
            )[0]
            for src in image_list
        ]

    def build_img_ratio_slice_logits_processor(self, tokenizer):
        if self.img_ratio_slice_logits_processor is None:
            self.img_ratio_slice_logits_processor = LogitsProcessorList([
                SliceVocabLogitsWarper(
                    vocab_start=tokenizer.ratio_token_id(0),
                    vocab_end=tokenizer.ratio_token_id(0) + len(self.vae_reso_group),
                )
            ])

    def postprocess_outputs(self, outputs: list[Image.Image], batch_cond_images):
        return outputs
