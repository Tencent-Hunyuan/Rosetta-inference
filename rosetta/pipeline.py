from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union
import random

import pandas as pd
import torch
from PIL import Image
from transformers import TextStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationMixin, GenerateDecoderOnlyOutput, GenerateOutput
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig, GenerationConfig
from transformers.utils import ModelOutput

from rosetta.autoencoder import FlowMatchDiscreteScheduler, MultimodalPipeline
from rosetta.configuration import MultimodalConfig
from rosetta.image_processor import ImageProcessor
from rosetta.modeling import MultimodalModelBase, MultimodalStaticCache
from rosetta.tokenizer import get_conversation_template
from rosetta.utils import CondImage, ImageInfo, ImageTensor, PRECISION_TO_TYPE, default, get_parallel_state

@dataclass
class MultimodalGenerationOutputs:
    batch: Optional[dict[str, Any]] = None
    texts: Optional[GenerateOutput | torch.LongTensor | list[str]] = None
    images: Optional[torch.Tensor | list[Image.Image]] = None

    def is_empty(self):
        return (
            self.texts is None and self.images is None
        ) or len(self.batch["index"]) == 0

    def postprocess_outputs(self, batch: dict[str, Any]):
        batch_size = len(batch["index"])
        valid_indices = torch.where(batch["is_dummy"].logical_not())[0].tolist()
        if len(valid_indices) == 0:
            return MultimodalGenerationOutputs(batch={"index": []})

        valid_batch = {
            k: v[valid_indices]
            if isinstance(v, torch.Tensor)
            else v.iloc[valid_indices].reset_index(drop=True)
            if isinstance(v, pd.DataFrame)
            else [v[i] for i in valid_indices]
            for k, v in batch.items()
        }

        def assert_length(name, data):
            assert len(data) == batch_size, f"Length of {name}({len(data)}) must match batch size({batch_size})."

        if self.texts is not None:
            if isinstance(self.texts, list):
                assert_length("texts", self.texts)
                valid_texts = [self.texts[vi] for vi in valid_indices]
            else:
                self.texts.sequences = [self.texts.sequences[vi] for vi in valid_indices]    # type: ignore
                self.texts.logits = tuple([logit[valid_indices] for logit in self.texts.logits])  # type: ignore
                valid_texts = self.texts
        else:
            valid_texts = None

        if self.images is not None:
            assert_length("images", self.images)
            if isinstance(self.images, torch.Tensor):
                valid_images = self.images[valid_indices]
            else:
                valid_images = [self.images[vi] for vi in valid_indices]
        else:
            valid_images = None

        return MultimodalGenerationOutputs(
            batch=valid_batch,
            texts=valid_texts,
            images=valid_images,
        )

InputImage = Optional[Union[Image.Image, str, bytes]]
IMAGE_INPUT_TYPES = (Image.Image, str, bytes)
Messages = list[dict[str, Any]]


def to_device(data, device):
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data


def map_absolute_to_local_token_indices(
    input_pos: torch.Tensor,
    abs_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if abs_indices.numel() == 0:
        return abs_indices.to(device)
    B = input_pos.size(0)

    if abs_indices.dim() == 1:
        abs_indices = abs_indices.unsqueeze(0).expand(B, -1).to(device)
    else:
        abs_indices = abs_indices.to(device)
    input_pos = input_pos.to(device)

    local_indices_list = []
    for b in range(B):
        in_current = (input_pos[b].unsqueeze(1) == abs_indices[b].unsqueeze(0))  # (cur_len, N)
        local_idx = in_current.float().argmax(0)  # (N,)
        found = in_current.any(0)  # (N,) False for padding (-1) or positions not in current forward
        valid_local = local_idx[found]
        local_indices_list.append(valid_local)
    max_count = max(len(x) for x in local_indices_list)
    if max_count == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=device)
    result_list = []
    for b in range(B):
        valid = local_indices_list[b]
        if len(valid) == 0:
            result_list.append(torch.zeros(max_count, dtype=torch.long, device=device))
        else:
            repeat_times = (max_count + len(valid) - 1) // len(valid)
            padded = (valid.repeat(repeat_times))[:max_count]
            result_list.append(padded)
    return torch.stack(result_list)


class MultimodalGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.diff_infer_steps = kwargs.pop("diff_infer_steps", 50)
        self.diff_guidance_scale = kwargs.pop("diff_guidance_scale", 5.0)
        self.flow_reverse = kwargs.pop("flow_reverse", False)
        self.flow_shift = kwargs.get("flow_shift", 3.0)
        self.bot_task = kwargs.get("bot_task", "image")
        self.sequence_template = kwargs.pop("sequence_template", "pretrain")


class MultimodalHFConfig(PretrainedConfig):
    def __init__(self, hf_config):
        super().__init__()
        for key, value in hf_config.items():
            setattr(self, key, value)


class MultimodalPreTrainedModel(PreTrainedModel):
    config_class = MultimodalConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["MultimodalDecoderLayer", "MultimodalMoTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True


class MultimodalHFModel(MultimodalModelBase, MultimodalPreTrainedModel, GenerationMixin):
    def __init__(
            self,
            args: Namespace,
            config: MultimodalConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            initialize_weights: bool = True,
    ):
        hf_config = MultimodalHFConfig(config.to_hf_config())
        super().__init__(hf_config)
        self.args = args
        self._dtype = dtype
        self.config = hf_config
        self.__post_init__(config, dtype, device, args, initialize_weights)

        self.tie_weights()

        self.image_processor = ImageProcessor(args)

        self._tokenizer = None
        self._diffusion_pipeline = None
        self.vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        self.model_dict = dict(vae=None)

    @property
    def dtype(self):
        return self._dtype

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def diffusion_pipeline(self):
        return self._diffusion_pipeline

    def build_diffusion_pipeline(self, gen_config: MultimodalGenerationConfig):
        scheduler = FlowMatchDiscreteScheduler(
            shift=gen_config.flow_shift,
            reverse=gen_config.flow_reverse,
        )
        assert self.model_dict["vae"] is not None, "VAE must be initialized before building diffusion pipeline."
        self._diffusion_pipeline = MultimodalPipeline(
            model=self, scheduler=scheduler, vae=self.model_dict["vae"],
        )

    def load_generation_config(self, generation_config_path: str | Path):
        generation_config_path = Path(generation_config_path)
        if not generation_config_path.exists():
            raise ValueError(f"Generation config path {generation_config_path} does not exist.")
        if generation_config_path.is_file():
            config_dir = generation_config_path.parent
            config_file_name = generation_config_path.name
        else:
            assert generation_config_path.is_dir() and (generation_config_path / "generation_config.json").exists(), \
                (f"No generation_config.json found in {generation_config_path}. "
                 f"Please check the weight format or provide by --generation-config argument.")
            config_dir = generation_config_path
            config_file_name = None

        config_keys = MultimodalGenerationConfig().to_dict().keys()
        overrides = {}
        for key in config_keys:
            if getattr(self.args, key, None) is not None:
                overrides[key] = getattr(self.args, key)

        self.generation_config = MultimodalGenerationConfig.from_pretrained(
            config_dir, config_file_name=config_file_name, **overrides,
        )

    @staticmethod
    def check_inputs(prompt=None, image=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError("`prompt` and `message_list` cannot be provided at the same time.")
        if message_list is not None:
            if not isinstance(message_list, list):
                raise ValueError(f"`message_list` should be a list of messages, but got {type(message_list)}.")
            assert len(message_list) > 0, "`message_list` should be a non-empty list."
            for message in message_list:
                assert isinstance(message, list) or isinstance(message, dict), \
                    f"Each message should be a list of dicts or a dict, but got {type(message)}."
        if image is not None:
            error_msg = \
                "`image` should be a PIL Image, a string path, a base64 string, bytes, or a list of them, but got {}."
            if isinstance(image, list):
                for im in image:
                    assert isinstance(im, IMAGE_INPUT_TYPES), error_msg.format(type(im))
            else:
                assert isinstance(image, IMAGE_INPUT_TYPES), error_msg.format(type(image))

    @staticmethod
    def _validate_and_batchify_text(text, name, check_batch_size=None):
        if text is None:
            return text
        assert isinstance(text, str) or isinstance(text, list), \
            f"Input `{name}` should be a string or a list of strings, but got {type(text)}."
        if isinstance(text, str):
            text = [text]
        assert len(text) > 0 and all(isinstance(p, str) and len(p) > 0 for p in text), \
            f"Input `{name}` should be a non-empty list of non-empty strings, got {text}."
        if check_batch_size is not None:
            assert len(text) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size}), got {len(text)}."
        return text

    @staticmethod
    def _validate_and_batchify_image(image, name, check_batch_size=None):
        if image is None:
            return image
        if not isinstance(image, list):
            raise ValueError(f"Input `{name}` should be a list of images, but got {type(image)}.")
        batch_image_list = [image] if not isinstance(image[0], list) else image
        for image_list in batch_image_list:
            assert all(isinstance(im, IMAGE_INPUT_TYPES) for im in image_list), \
                (f"Each item in `{name}` should be a PIL Image, a string path, a base64 string, or bytes, "
                 f"got {[type(im) for im in image_list]}.")
        if check_batch_size is not None:
            assert len(batch_image_list) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size})"
        return batch_image_list

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(f"Length of seed must be equal to the batch_size({batch_size}), got {seed}.")
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        return seeds

    def build_batch_rope_image_info(self, output, sections):
        meta_dict: dict[str, dict | list[dict]] = dict(
            gen_image={},
            cond_vae_image={},
            cond_vit_image={},
        )
        meta_dict["cond_joint_image"] = [meta_dict["cond_vae_image"], meta_dict["cond_vit_image"]]

        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            rope_image_slices = []
            rope_image_shapes = []
            rope_image_metas = []
            image_idx = 0

            for section in sections_i:
                if section['type'] in ["gen_image", "cond_vae_image", "cond_vit_image"]:
                    assert image_idx < len(image_slices), \
                        f"Image index {image_idx} out of range for image slices with length {len(image_slices)}."
                    rope_image_slices.append(image_slices[image_idx])
                    rope_image_shapes.append((section['token_height'], section['token_width']))
                    rope_image_metas.append(meta_dict[section['type']])
                    image_idx += 1

                elif section['type'] == "cond_joint_image":
                    assert image_idx + 1 < len(image_slices), \
                        f"Image index {image_idx + 1} out of range for image slices with length {len(image_slices)}."
                    assert len(section['token_height']) == len(section['token_width']), \
                        (f"token_height and token_width should have the same length, "
                         f"but got {len(section['token_height'])} and {len(section['token_width'])}")

                    rope_image_slices.extend([image_slices[image_idx], image_slices[image_idx + 1]])
                    rope_image_shapes.extend(list(zip(section['token_height'], section['token_width'])))
                    rope_image_metas.extend([meta_dict[section['type']][i] for i in range(2)])
                    image_idx += 2

            rope_image_info.append(list(zip(rope_image_slices, rope_image_shapes, rope_image_metas)))

        return rope_image_info

    def vae_encode(self, image, cfg_factor=1):
        config = self.model_dict["vae"].config

        with torch.autocast(
                device_type="cuda", dtype=self.vae_autocast_dtype,  # noqa
                enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32
        ):
            vae_encode_result = self.model_dict["vae"].encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(config, 'shift_factor') and config.shift_factor:
                latents.sub_(config.shift_factor)
            if hasattr(config, 'scaling_factor') and config.scaling_factor:
                latents.mul_(config.scaling_factor)

        t = torch.zeros((latents.shape[0],))

        if cfg_factor > 1:
            t = t.repeat(cfg_factor)
            latents = latents.repeat(cfg_factor, 1, 1, 1)

        return t, latents

    def _encode_cond_image(
            self,
            batch_cond_images: list[list[Union[ImageTensor, CondImage]]],
            cfg_factor: int = 1,
    ):
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None, None, None

        first_image = batch_cond_images[0][0]

        if first_image.section_type in ["cond_vae_image", "cond_joint_image"]:
            batch_cond_vae_images, batch_cond_t = [], []
            for cond_images in batch_cond_images:
                cond_vae_image_list, cond_t_list = [], []
                for cond_image in cond_images:
                    vae_image = (
                        cond_image.vae_image
                        if cond_image.section_type == "cond_joint_image"
                        else cond_image
                    )
                    cond_t_, cond_vae_image_ = self.vae_encode(
                        vae_image[None].to(self.device),
                    )
                    cond_vae_image_list.append(cond_vae_image_.squeeze(0))
                    cond_t_list.append(cond_t_)
                batch_cond_vae_images.append(cond_vae_image_list)
                batch_cond_t.append(cond_t_list)

            if all([len(items) == 1 for items in batch_cond_vae_images]) and all(
                    items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images):
                cond_vae_images = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
                cond_t = torch.cat([items[0] for items in batch_cond_t], dim=0)
                if cfg_factor > 1:
                    cond_t = cond_t.repeat(cfg_factor)
                    cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
            else:
                cond_t = [torch.cat(item, dim=0) for item in batch_cond_t]
                cond_vae_images = []
                for items in batch_cond_vae_images:
                    if all(items[0].shape == item.shape for item in items):
                        cond_vae_images.append(torch.stack(items, dim=0))
                    else:
                        cond_vae_images.append(items)
                if cfg_factor > 1:
                    cond_t = cond_t * cfg_factor
                    cond_vae_images = cond_vae_images * cfg_factor

        else:
            cond_vae_images = None
            cond_t = None

        if first_image.section_type in ["cond_vit_image", "cond_joint_image"]:
            cond_vit_images = []
            for cond_images in batch_cond_images:
                cond_vit_image_list = []
                for cond_image in cond_images:
                    vit_image = (
                        cond_image.vit_image
                        if cond_image.section_type == "cond_joint_image"
                        else cond_image
                    )
                    vit_image = vit_image.to(dtype=torch.float32)
                    cond_vit_image_list.append(vit_image)
                cond_vit_images.append(cond_vit_image_list)


            if cfg_factor > 1:
                cond_vit_images = cond_vit_images * cfg_factor

        else:
            cond_vit_images = None

        return cond_vae_images, cond_t, cond_vit_images

    @staticmethod
    def _prepare_vit_image_kwargs(batch_cond_images):
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None
        first_image = batch_cond_images[0][0]
        if first_image.section_type == "cond_joint_image":
            vit_image = first_image.vit_image
        else:
            vit_image = first_image
        if not hasattr(vit_image, "vision_encoder_kwargs") or len(vit_image.vision_encoder_kwargs) == 0:
            return None

        image_type = vit_image.i.image_type
        if image_type != "qwen3vl":
            raise ValueError(f"Unsupported ViT image type: {image_type}")

        cond_vit_image_kwargs = {"grid_thw": []}
        for cond_images in batch_cond_images:
            cond_vit_image_kwargs["grid_thw"].append(torch.stack([
                (cond_image.vit_image if cond_image.section_type == "cond_joint_image" else cond_image)
                .vision_encoder_kwargs["grid_thw"]
                for cond_image in cond_images
            ]))
        return cond_vit_image_kwargs

    def prepare_message_list(
            self,
            message_list,
            cond_images: list[CondImage] = None,
            gen_image_info: ImageInfo = None,
    ):
        inner_message_list = []
        image_idx = 0
        for message in message_list:
            content = message["content"]
            if isinstance(content, str):
                inner_message_list.append(dict(role=message["role"], type="text", content=content))
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        inner_message_list.append(dict(role=message["role"], type="text", content=item['text']))
                    elif item["type"] == "image":
                        if all(key not in item for key in ["image", "url", "path", "base64"]):
                            continue
                        assert cond_images is not None and image_idx < len(cond_images), \
                            f"Image index {image_idx} out of range for cond images with length {len(cond_images)}."
                        image = cond_images[image_idx]
                        inner_message_list.append(dict(role=message["role"], type=image.section_type, content=image.i))
                        image_idx += 1
                    else:
                        raise NotImplementedError(f"Message content type {item['type']} not supported.")
            else:
                raise ValueError(f"Message content should be str or list, but got {type(content)}.")

        if gen_image_info is not None:
            inner_message_list.append(dict(role="assistant", type="gen_image", content=gen_image_info))

        return inner_message_list

    def _build_batch_gen_image_info(self, image_size, batch_size):
        if isinstance(image_size, list):
            assert len(image_size) == batch_size, \
                f"image_size should have the same length as batch_size, got {len(image_size)} and {batch_size}"
            return [self.image_processor.build_gen_image_info(image_size[i]) for i in range(batch_size)]
        return [self.image_processor.build_gen_image_info(image_size) for _ in range(batch_size)]

    def prepare_model_inputs(
            self,
            prompt: str | list[str] = None,
            image: list[InputImage] = None,
            mode="gen_text",
            image_size: str | list[tuple[int, int]] = "auto",
            message_list: Optional[Messages | list[Messages]] = None,
            device=None,
            max_new_tokens=None,
            bot_task="auto",
            **kwargs,
    ):
        self.check_inputs(prompt, image, message_list)
        device = default(device, self.device)

        batch_message_list = message_list
        batch_prompt = prompt

        batch_cond_images = kwargs.get('batch_cond_images', None)

        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            batch_message_list = deepcopy(batch_message_list)

            if batch_cond_images is None:
                batch_cond_images = [
                    self.image_processor.build_cond_images(message_list=message_list_)
                    for message_list_ in batch_message_list
                ]
            if mode == "gen_image":
                batch_gen_image_info = self._build_batch_gen_image_info(image_size, batch_size)
            else:
                batch_gen_image_info = [None] * batch_size

            batch_message_list = [
                self.prepare_message_list(message_list_, cond_images, gen_image_info)
                for message_list_, cond_images, gen_image_info in zip(
                    batch_message_list, batch_cond_images, batch_gen_image_info
                )
            ]

        else:
            batch_prompt = self._validate_and_batchify_text(batch_prompt, 'prompt')
            batch_size = len(batch_prompt)

            batch_image_list = self._validate_and_batchify_image(image, 'image', batch_size)
            if batch_cond_images is None:
                batch_cond_images = [
                    self.image_processor.build_cond_images(image_list=image_list)
                    for image_list in batch_image_list
                ] if batch_image_list is not None else None

            if mode == "gen_image":
                batch_gen_image_info = self._build_batch_gen_image_info(image_size, batch_size)
            else:
                batch_gen_image_info = [None] * batch_size

        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        cfg_factor = {
            "gen_text": 1,
            "gen_image": 2
        }
        conv_template = get_conversation_template(self.args.model_name.split('.')[-1])
        out = self._tokenizer.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_images=batch_cond_images,
            max_length=kwargs.get('max_length'),
            bot_task=bot_task,
            image_base_size=self.image_processor.vae_reso_group.base_size if bot_task == "img_ratio" else None,
            sequence_template=self.generation_config.sequence_template,
            cfg_factor=cfg_factor[mode],
            conv_template=conv_template,
            und_token_type=self.args.und_token_type if self.args.use_mot else [],
            gen_token_type=self.args.gen_token_type if self.args.use_mot else [],
        )

        output, sections = out['output'], out['sections']
        cond_vae_images, cond_timesteps, cond_vit_images = self._encode_cond_image(
            batch_cond_images, cfg_factor[mode]
        )
        cond_vit_image_kwargs = self._prepare_vit_image_kwargs(batch_cond_images)

        rope_image_info = self.build_batch_rope_image_info(output, sections)

        max_new_tokens = default(
            default(max_new_tokens, self.generation_config.max_new_tokens),
            self.generation_config.max_length,
        )
        if mode == "gen_image":
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + max_new_tokens
        cache = MultimodalStaticCache(
            config=self.config,
            max_batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=self.dtype,
            dynamic=mode == "gen_text",
        )

        batch_input_pos = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device)[None].expand(
            batch_size * cfg_factor[mode], -1)

        tkw = self._tokenizer
        if mode == "gen_image":
            eos_token_id = None
        else:
            if bot_task == "auto":
                stop_token_id = dict(
                    auto=conv_template.stop_token_ids,
                )
            else:
                if image_size == "auto":
                    extra_auto_stops = list(range(
                        tkw.ratio_token_id(0),
                        tkw.ratio_token_id(0) + len(self.image_processor.vae_reso_group)
                    ))
                else:
                    extra_auto_stops = [tkw.boi_token_id]
                stop_token_id = dict(
                    auto=conv_template.stop_token_ids + extra_auto_stops,
                    img_ratio=extra_auto_stops,
                )
            eos_token_id = stop_token_id[bot_task]

        batch_image_sizes = None
        if mode == "gen_image" and all(batch_gen_image_info[i] is not None for i in range(len(batch_gen_image_info))):
            all_same_size = all(
                info.image_height == batch_gen_image_info[0].image_height and
                info.image_width == batch_gen_image_info[0].image_width
                for info in batch_gen_image_info
            )
            if not all_same_size:
                vae_df = self.config.vae_downsample_factor
                batch_image_sizes = []
                for info in batch_gen_image_info:
                    batch_image_sizes.append((info.image_height // vae_df, info.image_width // vae_df))

        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            input_pos=batch_input_pos,
            past_key_values=cache,
            mode=mode,
            rope_image_info=rope_image_info,
            image_mask=to_device(output.gen_image_mask, device),
            timesteps_index=to_device(output.gen_timestep_scatter_index, device),
            cond_vae_images=to_device(cond_vae_images, device),
            cond_vae_image_mask=to_device(output.vae_image_mask, device),
            cond_timesteps=to_device(cond_timesteps, device),
            cond_timesteps_index=to_device(output.cond_timestep_scatter_index, device),
            cond_vit_images=to_device(cond_vit_images, device),
            cond_vit_image_mask=to_device(output.vit_image_mask, device),
            cond_vit_image_kwargs=to_device(cond_vit_image_kwargs, device),
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            batch_cond_images=batch_cond_images,
            batch_image_sizes=batch_image_sizes,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=kwargs.get("return_dict_in_generate", False),
            output_logits=kwargs.get("output_logits", None),
        )
        if mode == "gen_image":
            model_input_kwargs["cfg_factor"] = cfg_factor[mode]
        if self.args.use_mot:
            model_input_kwargs["und_token_indices"] = to_device(output.und_token_indices, device)
            model_input_kwargs["gen_token_indices"] = to_device(output.gen_token_indices, device)

        return model_input_kwargs

    def _prepare_attention_mask_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            generation_config: GenerationConfig,
            model_kwargs: dict[str, Any],
    ) -> Optional[torch.Tensor]:
        bsz, seq_len = inputs_tensor.shape
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_full_attn_slices = [
            self.image_processor.prepare_full_attn_slices(tokenizer_output, i)
            for i in range(bsz)
        ]
        if len(batch_full_attn_slices[0]) == 0:
            return None

        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).tril(
            diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_full_attn_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            tokenizer_output=None, batch_cond_images=None, batch_gen_image_info=None, generator=None,
            **kwargs
    ):
        input_pos = kwargs.get("input_pos", kwargs.get("position_ids"))
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            assert input_pos is not None, "input_pos or position_ids must be provided in kwargs."
            if input_ids is not None and input_ids.shape[1] != input_pos.shape[1]:
                assert input_pos.size(0) == 1, f"Only support batch size 1, got {input_pos.size(0)}"
                if input_pos[0, -1] >= input_ids.shape[1]:
                    input_ids = input_ids[:, -input_pos.shape[1]:]
                    input_pos[0, -1] = input_ids.shape[1] - 1
                else:
                    input_ids = torch.gather(input_ids, dim=1, index=input_pos)
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "input_pos": input_pos,
                "past_key_values": past_key_values,
                "mode": kwargs["mode"],
                "rope_image_info": kwargs["rope_image_info"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timesteps": kwargs.get("timesteps"),
                "timesteps_index": kwargs.get("timesteps_index"),
                "cond_vae_images": kwargs.get("cond_vae_images"),
                "cond_vae_image_mask": kwargs.get("cond_vae_image_mask"),
                "cond_timesteps": kwargs.get("cond_timesteps"),
                "cond_timesteps_index": kwargs.get("cond_timesteps_index"),
                "cond_vit_images": kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": kwargs.get("cond_vit_image_mask"),
                "cond_vit_image_kwargs": kwargs.get("cond_vit_image_kwargs"),
                "batch_image_sizes": kwargs.get("batch_image_sizes"),
            }
        )
        if self.args.use_mot:
            und_abs = kwargs.get("und_token_indices")
            gen_abs = kwargs.get("gen_token_indices")
            if past_key_values is not None and und_abs is not None and gen_abs is not None:
                device = input_pos.device
                model_inputs["und_token_indices"] = map_absolute_to_local_token_indices(
                    input_pos, und_abs, device
                )
                model_inputs["gen_token_indices"] = map_absolute_to_local_token_indices(
                    input_pos, gen_abs, device
                )
            else:
                model_inputs["und_token_indices"] = und_abs
                model_inputs["gen_token_indices"] = gen_abs
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "rope_image_info": model_kwargs["rope_image_info"],
        }

        if "past_key_values" in outputs:
            updated_model_kwargs["past_key_values"] = outputs.past_key_values

        if "tokenizer_output" in model_kwargs:
            if mode == "gen_text":
                real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                updated_model_kwargs["input_pos"] = real_pos
            else:
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)

                batch_image_sizes = model_kwargs.get("batch_image_sizes")
                if batch_image_sizes is not None:
                    img_token_counts = image_mask.sum(dim=1).long()  # (bsz,)
                    max_img_tokens = img_token_counts.max().item()
                    pad_counts = max_img_tokens - img_token_counts  # (bsz,)
                    input_pos_list = []
                    for i in range(bsz):
                        indices = index[i].masked_select(image_mask[i].bool())
                        if pad_counts[i] > 0:
                            indices = torch.cat([indices, index[i][-pad_counts[i]:]])
                        input_pos_list.append(indices)
                    input_pos = torch.stack(input_pos_list, dim=0)
                else:
                    input_pos = index.masked_select(image_mask.bool()).reshape(bsz, -1)

                timestep_position_ids = \
                    index[torch.arange(bsz), model_kwargs["timesteps_index"][:, -1]].unsqueeze(-1)
                updated_model_kwargs["input_pos"] = torch.cat([timestep_position_ids, input_pos], dim=1)

                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                        model_kwargs["attention_mask"], updated_model_kwargs["input_pos"]):
                    mask_list.append(torch.index_select(attention_mask_i, dim=1, index=position_ids_i.reshape(-1)))
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
        else:
            if mode == "gen_text":
                updated_model_kwargs["input_pos"] = model_kwargs["input_pos"] + 1
            else:
                updated_model_kwargs["input_pos"] = model_kwargs["input_pos"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]

        if self.args.use_mot:
            if "und_token_indices" in model_kwargs:
                updated_model_kwargs["und_token_indices"] = model_kwargs["und_token_indices"]
            if "gen_token_indices" in model_kwargs:
                updated_model_kwargs["gen_token_indices"] = model_kwargs["gen_token_indices"]

        if "batch_image_sizes" in model_kwargs:
            updated_model_kwargs["batch_image_sizes"] = model_kwargs["batch_image_sizes"]

        return updated_model_kwargs

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            use_model_defaults: Optional[bool] = None,
            generator: Optional[list[torch.Generator]] = None,
            decode_text: bool = False,
            verbose: int = 0,
            image_output_type: str = "pil",
            skip_special_tokens: bool = False,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        gen_config = default(generation_config, self.generation_config)
        mode = kwargs.get("mode", "gen_text")
        return_dict_in_generate = kwargs.get("return_dict_in_generate", gen_config.return_dict_in_generate)

        if mode == "gen_text":
            if verbose >= 2 and streamer is None:
                streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=False)

            dp_enabled = get_parallel_state().dp_size > 1

            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                results = super().generate(
                    inputs,
                    gen_config,
                    logits_processor,
                    stopping_criteria,
                    prefix_allowed_tokens_fn,
                    synced_gpus or dp_enabled,
                    assistant_model,
                    streamer,
                    negative_prompt_ids,
                    negative_prompt_attention_mask,
                    use_model_defaults,
                    **kwargs,
                )
                if isinstance(results, torch.Tensor):
                    samples = results
                else:
                    samples = results.sequences
                if decode_text:
                    samples = self.decode_text(samples, input_length=kwargs["input_ids"].shape[1], skip_special_tokens=skip_special_tokens)
                samples = MultimodalGenerationOutputs(texts=samples)

        elif mode == "gen_image":
            batch_gen_image_info: list[ImageInfo] = kwargs.get("batch_gen_image_info")
            if batch_gen_image_info is None:
                raise ValueError("`batch_gen_image_info` should be provided when `mode` is `gen_image`.")

            gen_config.update(**kwargs)
            self.build_diffusion_pipeline(gen_config)

            all_same = all(
                info.image_height == batch_gen_image_info[0].image_height and
                info.image_width == batch_gen_image_info[0].image_width
                for info in batch_gen_image_info
            )
            if all_same:
                pipeline_image_size = (batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width)
            else:
                pipeline_image_size = [(info.image_height, info.image_width) for info in batch_gen_image_info]

            results = self.diffusion_pipeline(
                batch_size=len(batch_gen_image_info),
                image_size=pipeline_image_size,
                num_inference_steps=gen_config.diff_infer_steps,
                guidance_scale=gen_config.diff_guidance_scale,
                cfg_factor=kwargs['cfg_factor'],
                generator=generator,
                output_type=image_output_type,
                model_kwargs=kwargs,
            )
            samples = MultimodalGenerationOutputs(images=results)

        else:
            raise ValueError(f"Unknown mode {mode}, only `gen_text` and `gen_image` are supported.")

        kv_cache: MultimodalStaticCache = kwargs.get('past_key_values', None)
        if kv_cache is not None:
            for layer in kv_cache.layers:
                del layer.keys
                del layer.values
            torch.cuda.empty_cache()

        if return_dict_in_generate:
            return MultimodalGenerationOutputs(
                texts=GenerateDecoderOnlyOutput(sequences=samples.texts, logits=results.logits),
            )

        return samples

    def decode_text(self, output: torch.Tensor, input_length: int = None, skip_special_tokens: bool = False):
        if output.ndim == 2:
            assert output.size(0) == 1, "Batch decoding is not supported yet."
            return [self.decode_text(output_i, input_length, skip_special_tokens=skip_special_tokens) for output_i in output]
        elif output.ndim == 1:
            if input_length is not None:
                output = output[input_length:]
            text = self._tokenizer.decode(output, skip_special_tokens=skip_special_tokens)
            return text
        else:
            raise ValueError(f"output should be 1D or 2D tensor, but got {output.ndim}D tensor.")

    def generate_image(
            self,
            prompt=None,
            image=None,
            message_list=None,
            seed=None,
            image_size="auto",
            bot_task=None,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        bot_task = default(bot_task, self.generation_config.bot_task)

        if message_list is not None:
            message_list = deepcopy(message_list)

        batch_cond_images_cache = None
        if image_size == "auto":
            self.image_processor.build_img_ratio_slice_logits_processor(self.tokenizer)
            model_inputs = self.prepare_model_inputs(
                seed=seed, mode="gen_text", bot_task="img_ratio", max_new_tokens=1,
                prompt=prompt, image=image,
                message_list=message_list, batch_cond_images=batch_cond_images_cache,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            outputs = self.generate(
                **model_inputs,
                do_sample=False,
                logits_processor=self.image_processor.img_ratio_slice_logits_processor,
                **kwargs,
            )
            bsz = outputs.texts.size(0)
            if bsz == 1:
                ratio_index = outputs.texts[0, -1].item()
                reso = self.image_processor.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            else:
                image_size = []
                for b in range(bsz):
                    ratio_index = outputs.texts[b, -1].item()
                    reso = self.image_processor.vae_reso_group[ratio_index]
                    image_size.append((reso.height, reso.width))

        model_inputs = self.prepare_model_inputs(
            prompt=prompt, image=image, message_list=message_list,
            seed=seed, image_size=image_size, mode="gen_image", batch_cond_images=batch_cond_images_cache,
        )
        batch_cond_images_cache = model_inputs['batch_cond_images']
        outputs = self.generate(**model_inputs, **kwargs)

        outputs.texts = None
        outputs.images = self.image_processor.postprocess_outputs(outputs.images, batch_cond_images_cache)
        return outputs

