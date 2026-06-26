import importlib
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.utils import BaseOutput
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from rosetta.utils import CondImage, ImageInfo, JointImageInfo, default


@dataclass
class Conversation(object):
    name: str
    roles: Tuple[str, str] = ("User", "Assistant")
    sep: str = "\n"
    sep2: str = None
    sep_sp: str = None
    stop_token_ids: list[int] = None
    pretrain_roles: Tuple[str, str] = ("", "")
    pretrain_sep: str = ""
    pretrain_sep2: str = ""
    pretrain_sep_sp: str = ""
    add_pad: bool = False
    add_bos: bool = True
    add_eos: bool = False

    def get_role_prefix(self, role):
        if role == "":
            return ""
        return f"<|im_start|>{role}\n"

    def empty(self, name=None):
        return Conversation(
            name=name or self.name,
            roles=self.roles,
            sep=self.sep,
            sep2=self.sep2,
            sep_sp=self.sep_sp,
            stop_token_ids=self.stop_token_ids,
            pretrain_roles=self.pretrain_roles,
            pretrain_sep=self.pretrain_sep,
            pretrain_sep2=self.pretrain_sep2,
            pretrain_sep_sp=self.pretrain_sep_sp,
            add_pad=self.add_pad,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
        )


conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    assert template.name not in conv_templates, f"{template.name} has been registered."
    conv_templates[template.name] = template


register_conv_template(Conversation(
    name="qwen-vl-30b-a3b-instruct",
    roles=("user", "assistant"),
    sep="<|im_end|>\n",
    sep2="<|im_end|>",
    sep_sp="\n\n",
    stop_token_ids=[151643, 151645],
    add_bos=False,
))
for _template_name in (
    "qwen3-06b-base-upcycling-moe-lm-deepseek",
    "qwen3-06b-base-upcycling-ours-lm",
    "qwen3-06b-upcycling-moe-mm-deepseek",
    "qwen3-06b-upcycling-ours-mm",
    "qwen3-06b-base-mot-lm",
    "qwen3-06b-mot",
):
    register_conv_template(conv_templates["qwen-vl-30b-a3b-instruct"].empty(name=_template_name))


def get_conversation_template(name: str) -> Conversation:
    return deepcopy(conv_templates[name])


ASSETS_BASE = os.getenv("ASSETS_BASE", "./public_assets").rstrip("/")
TOKENIZER_BASE = os.getenv("TOKENIZER_BASE", f"{ASSETS_BASE}/pretrained_llm").rstrip("/")
TOKENIZER_PATH = {
    "qwen3-0.6b-base": f"{TOKENIZER_BASE}/Qwen3-0.6B-Base",
}


class TokenizerEncodeOutput(BaseOutput):
    tokens: torch.Tensor = None
    text_slices: Optional[list[slice]] = None
    gen_image_slices: Optional[list[slice]] = None
    vae_image_slices: Optional[list[slice]] = None
    vit_image_slices: Optional[list[slice]] = None
    joint_image_slices: Optional[list[slice]] = None
    all_image_slices: Optional[list[slice]] = None
    text_mask: Optional[torch.Tensor] = None
    gen_image_mask: Optional[torch.Tensor] = None
    vae_image_mask: Optional[torch.Tensor] = None
    vit_image_mask: Optional[torch.Tensor] = None
    real_pos: Optional[torch.Tensor] = None
    cond_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_timestep_scatter_index: Optional[torch.Tensor] = None
    und_token_indices: Optional[torch.Tensor] = None
    gen_token_indices: Optional[torch.Tensor] = None


class BaseMultimodalTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        special_tokens = self.special_tokens_map.get('additional_special_tokens', [])
        if len(special_tokens) > 0:
            special_token_ids = self.convert_tokens_to_ids(special_tokens)
            self._sp_dict = dict(zip(special_tokens, special_token_ids))
        else:
            self._sp_dict = dict()

        self.setup_special_tokens()

    def setup_special_tokens(self):
        predefined_name_mapping = {
            "answer": "",
            "end_of_answer": "",
            "boi": "<｜boi｜>",
            "eoi": "<｜eoi｜>",
            "img": "<｜img｜>",
        }
        for name, mapping in predefined_name_mapping.items():
            setattr(self, f"{name}_token", mapping)
            setattr(self, f"{name}_token_id", self.convert_tokens_to_ids(mapping))

        if len(self._sp_dict) > 0:
            name_mapping = dict(
                cfg_token="<｜cfg｜>",
                timestep_token="<｜timestep｜>",
                joint_img_sep_token="<｜joint_img_sep｜>",
            )
            for name, token in name_mapping.items():
                if token in self._sp_dict:
                    setattr(self, name, token)
                    setattr(self, f"{name}_id", self._sp_dict[token])

    def size_token(self, size: int):
        assert len(self._sp_dict) > 0, "Size tokens are not defined in the tokenizer."
        return f"<｜img_size_{size}｜>"

    def size_token_id(self, size: int):
        return self._sp_dict[self.size_token(size)]

    def ratio_token(self, ratio_idx: int):
        assert len(self._sp_dict) > 0, "Ratio tokens are not defined in the tokenizer."
        return f"<｜img_ratio_{ratio_idx}｜>"

    def ratio_token_id(self, ratio_idx: int):
        return self._sp_dict[self.ratio_token(ratio_idx)]

    def encode_text(
            self,
            *texts,
            uncond_enabled: Optional[bool | list[bool]] = None,
            uncond_p: Optional[float] = None,
            max_length: Optional[int] = None,
            pad: Optional[str] = None,
    ):
        if pad is not None:
            assert max_length is not None, "max_length should be provided when pad is not None."

        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        assert len(uncond_enabled) == len(texts), (
            f"Length of uncond_flags should be equal to the number of texts, "
            f"but got {len(uncond_enabled)} and {len(texts)}."
        )

        do_uncond_drop = (uncond_p is not None) and (random.random() < uncond_p)
        text_tokens = []
        cum_length = 0
        for text, uncond_flag in zip(texts, uncond_enabled):
            if max_length is not None and cum_length >= max_length:
                break
            if isinstance(text, str):
                text_token = self.encode(text, add_special_tokens=False)
            else:
                text_token = text
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            if max_length is not None and (cum_length + len(text_token)) > max_length:
                text_token = text_token[:max_length - cum_length]
            text_tokens.extend(text_token)
            cum_length += len(text_token)

        if pad is not None and (pad_length := max_length - len(text_tokens)) > 0:
            if pad == 'left':
                text_tokens = [self.pad_token_id] * pad_length + text_tokens
            elif pad == 'right':
                text_tokens = text_tokens + [self.pad_token_id] * pad_length
            else:
                raise ValueError(f"Unsupported padding method: {pad}.")

        return text_tokens

    @staticmethod
    def _check_key_number_matched(keys, data):
        assert set(keys) == set(data.keys()), (
            f"Keys in the template and token source should be matched, but got {keys} and {list(data.keys())}."
        )
        key_counts = {k: 0 for k in keys}
        for key in keys:
            key_counts[key] += 1
        for key, count in key_counts.items():
            assert len(data[key]) == count, (
                f"Number of `{key}` in the token source should be matched with the template, but got "
                f"{data[key]}({len(data[key])}) and {count}."
            )

    def _add_meta_info_token(
            self,
            token_seq,
            token_count,
            extra_token_pos,
            add_timestep_token: bool = False,
            add_image_shape_token: bool = False,
            base_size=None,
            ratio_idx=None,
            token_height=None,
            token_width=None,
            image_type=None,
            media_type=None,
            und_token_type: list[str] = [],
            gen_token_type: list[str] = [],
            und_token_indices: list[int] = [],
            gen_token_indices: list[int] = [],
            token_count_start: int = 0,
    ):
        add_mot_indices = partial(self.process_mot_indices, und_token_indices=und_token_indices, gen_token_indices=gen_token_indices, und_token_type=und_token_type, gen_token_type=gen_token_type)
        if add_image_shape_token:
            token_seq.extend([self.size_token_id(base_size), self.ratio_token_id(ratio_idx)])
            token_count += 2
            add_mot_indices(token_type="vae_info", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        if add_timestep_token:
            token_seq.extend([self.timestep_token_id])
            extra_token_pos['timestep'].append(token_count)
            if media_type is not None:
                if media_type == "gen_image":
                    extra_token_pos['gen_timestep'].append(token_count)
                elif media_type in ["cond_joint_image", "cond_vae_image"]:
                    extra_token_pos['cond_timestep'].append(token_count)
                else:
                    raise ValueError(f"Unsupported image type: {media_type}.")
            token_count += 1
            add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        return token_count, token_count_start

    def _shorten_text(self, text):
        text = re.sub(f"({self.img_token})+", lambda m: f"[{self.img_token}]{{{len(m.group(0)) // len(self.img_token)}}}", text)
        text = re.sub(f"({self.pad_token})+", lambda m: f"[{self.pad_token}]{{{len(m.group(0)) // len(self.pad_token)}}}", text)
        return text

    @staticmethod
    def process_mot_indices(token_type: str, token_indices: list[int], und_token_indices: list[int], gen_token_indices: list[int], und_token_type: list[str] = [], gen_token_type: list[str] = []):
        if token_type in und_token_type:
            und_token_indices.extend(token_indices)
        elif token_type in gen_token_type:
            gen_token_indices.extend(token_indices)

    def encode_sequence(
            self,
            template: str,
            token_source: dict[str, list[list[int] | dict[str, Any]]],
            total_length=None,
            add_eos=True,
            add_pad=True,
            add_bos=True,
            drop_last: str | bool = 'auto',
            add_image_shape_token=False,
            und_token_type: list[str] = [],
            gen_token_type: list[str] = [],
    ):
        if drop_last is True and total_length is None:
            raise ValueError("total_length should be provided when drop_last is True.")

        keys = template.split('-')
        index_indicator = {k: 0 for k in token_source}
        for k, v in token_source.items():
            assert isinstance(v, (list, tuple)), (
                f"Value of `{k}` in the token source should be a list or tuple, but got {type(v)}."
            )
        self._check_key_number_matched(keys, token_source)

        token_seq = []
        token_count = 0
        extra_token_pos = defaultdict(list)
        und_token_indices = []
        gen_token_indices = []
        add_mot_indices = partial(self.process_mot_indices, und_token_indices=und_token_indices, gen_token_indices=gen_token_indices, und_token_type=und_token_type, gen_token_type=gen_token_type)
        if add_bos and self.bos_token_id is not None:
            token_seq.append(self.bos_token_id)
            add_mot_indices(token_type="bos", token_indices=[token_count])
            token_count += 1
        drop_last_break = False
        for i, key in enumerate(keys):
            source = token_source[key][index_indicator[key]]
            token_count_start = token_count

            if key == "text":
                token_seq.extend(source)
                extra_token_pos["<text>_start"].append(token_count)
                token_count += len(source)
                extra_token_pos["<text>_end"].append(token_count - 1)
                add_mot_indices(token_type="text", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

            elif key == "gen_image":
                extra_count = \
                    2 \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (2 if source['add_image_shape_token'] else 0)
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                token_seq.append(self.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                add_mot_indices(token_type="boi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_image_shape_token=source['add_image_shape_token'],
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    token_height=source.get('token_height'),
                    token_width=source.get('token_width'),
                    image_type=key,
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )

                token_seq.extend(
                    [self.img_token_id] * source['length']
                )
                extra_token_pos["<img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)

                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                token_seq.extend([self.eoi_token_id])
                extra_token_pos["eoi"].append(token_count)
                add_mot_indices(token_type="eoi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

            elif key == "cond_joint_image":
                assert isinstance(source['length'], list) and len(
                    source['length']) == 2, "cond_joint_image length should be a list of two integers"
                extra_count = \
                    2 + 1 \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (2 if source['add_image_shape_token'] else 0)
                if drop_last is True and token_count + extra_count + sum(source['length']) > total_length:
                    drop_last_break = True
                    break
                token_seq.append(self.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                token_count += 1
                add_mot_indices(token_type="boi", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_image_shape_token=source['add_image_shape_token'],
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    token_height=source.get('token_height'),
                    token_width=source.get('token_width'),
                    image_type=key,
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )

                token_seq.extend(
                    [self.img_token_id] * source['length'][0]
                )
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<joint_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source['length'][0]
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                token_seq.extend([self.joint_img_sep_token_id])
                extra_token_pos["joint_img_sep"].append(token_count)
                add_mot_indices(token_type="joint_image_sep", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

                token_seq.extend(
                    [self.img_token_id] * source['length'][1]
                )
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source['length'][1]
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<joint_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)

                add_mot_indices(token_type="vit", token_indices=list(range(token_count_start, token_count)))

                token_seq.extend(
                    [self.eoi_token_id]
                )
                extra_token_pos["eoi"].append(token_count)
                add_mot_indices(token_type="eoi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

            elif key == "cond_vae_image":
                extra_count = \
                    2 \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (2 if source['add_image_shape_token'] else 0)
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                token_seq.append(self.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                add_mot_indices(token_type="boi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_image_shape_token=source['add_image_shape_token'],
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    token_height=source.get('token_height'),
                    token_width=source.get('token_width'),
                    image_type=key,
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )

                token_seq.extend(
                    [self.img_token_id] * source['length']
                )
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))

                token_seq.extend(
                    [self.eoi_token_id]
                )
                extra_token_pos["eoi"].append(token_count)
                add_mot_indices(token_type="eoi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

            elif key == "cond_vit_image":
                extra_count = 2
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                token_seq.append(self.boi_token_id)
                add_mot_indices(token_type="boi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

                token_seq.extend([self.img_token_id] * source['length'])
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                add_mot_indices(token_type="vit", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                token_seq.append(self.eoi_token_id)
                extra_token_pos["eoi"].append(token_count)
                add_mot_indices(token_type="eoi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

            else:
                raise ValueError(f"Not supported key: {key}")
            index_indicator[key] += 1

        if add_eos is True and not drop_last_break:
            token_seq.append(self.eos_token_id)
            extra_token_pos["eos"].append(token_count)
            add_mot_indices(token_type="eos", token_indices=[token_count])
            token_count += 1
            token_count_start = token_count
        elif add_eos == 'auto' and not drop_last_break:
            if token_seq[-1] != self.eos_token_id and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token_id)
                extra_token_pos["eos"].append(token_count)
                add_mot_indices(token_type="eos", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

        if total_length:
            if token_count > total_length and drop_last:
                for start_key, end_key in [
                    ("<img>_start", "<img>_end"), ("<vae_img>_start", "<vae_img>_end"),
                    ("<vit_img>_start", "<vit_img>_end"), ("<joint>_start", "<joint>_end"),
                ]:
                    if start_key in extra_token_pos and end_key in extra_token_pos:
                        assert all(
                            (start > total_length or end + 1 < total_length)
                            for start, end in zip(extra_token_pos[start_key], extra_token_pos[end_key])
                        ), ("Clip position should not be in the middle of the media tokens.\n"
                            f"Below is the text:\n{self._shorten_text(self.decode(token_seq))}")
                token_seq = token_seq[:total_length]
                und_token_indices = [idx for idx in und_token_indices if idx < total_length]
                gen_token_indices = [idx for idx in gen_token_indices if idx < total_length]

            pad_num = max(0, total_length - len(token_seq))
            if add_pad and pad_num:
                token_seq.extend([self.pad_token_id] * pad_num)
                extra_token_pos["first_pad"].append(token_count)
                add_mot_indices(token_type="pad", token_indices=list(range(token_count, token_count + pad_num)))

        if len(und_token_indices) > 0 and len(gen_token_indices) > 0:
            assert und_token_indices[-1] < len(token_seq) and gen_token_indices[-1] < len(token_seq), f"{und_token_indices[-1]=}, {gen_token_indices[-1]=}, {len(token_seq)=}"
        return token_seq, extra_token_pos, und_token_indices, gen_token_indices

    @staticmethod
    def parse_extra_token_pos(extra_token_pos, prefix, tokens, rng=None):
        if rng is None:
            rng = slice(None)
        image_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos[f'<{prefix}>_start'][rng], extra_token_pos[f'<{prefix}>_end'][rng])
        ] if f'<{prefix}>_start' in extra_token_pos and f'<{prefix}>_end' in extra_token_pos else []
        if image_slices:
            image_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for image_slice in image_slices:
                image_mask[image_slice] = True
        else:
            image_mask = None
        return image_slices, image_mask

    def encode_general(
            self,
            sections: Optional[list[dict[str, Any]]] = None,
            max_token_length: Optional[int] = None,
            add_eos: bool | str = 'auto',
            use_text_mask: bool = True,
            add_pad: bool | str = 'auto',
            add_bos: bool = True,
            drop_last: bool | str = 'auto',
            und_token_type: list[str] = [],
            gen_token_type: list[str] = [],
            disable_ignore: bool = False,
    ):
        if sections is None:
            raise ValueError("sections must be provided.")
        template = '-'.join([section['type'] for section in sections])

        sections = deepcopy(sections)
        token_source = defaultdict(list)
        text_mask_specs = []
        for section in sections:
            if section['type'] == 'text':
                text = self.encode_text(
                    section['text'] if 'text' in section else section['tokens'],
                    uncond_enabled=section.get('uncond_enabled'),
                    uncond_p=section.get('uncond_p'),
                    max_length=section.get('max_length'),
                )
                token_source['text'].append(text)
                text_mask_specs.append(dict(
                    ignore=section.get('ignore', False),
                    start_offset=section.get('start_offset', 0),
                    end_offset=section.get('end_offset', 0),
                ))
            elif section['type'] == 'gen_image':
                token_source['gen_image'].append(dict(
                    length=section['token_length'],
                    add_timestep_token=section.get('add_timestep_token', False),
                    add_image_shape_token=section.get('add_image_shape_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                    token_height=section.get('token_height'),
                    token_width=section.get('token_width'),
                ))
            elif section['type'] in ['cond_joint_image', 'cond_vae_image', 'cond_vit_image']:
                token_source[section['type']].append(dict(
                    length=section['token_length'],
                    add_timestep_token=section.get('add_timestep_token', False),
                    add_image_shape_token=section.get('add_image_shape_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                    token_height=section.get('token_height'),
                    token_width=section.get('token_width'),
                ))
            else:
                raise ValueError(f"Invalid section type: {section['type']}")

        full_token_seq, extra_token_pos, und_token_indices, gen_token_indices = self.encode_sequence(
            template=template,
            token_source=dict(token_source),
            total_length=max_token_length,
            add_eos=add_eos,
            add_pad=add_pad,
            add_bos=add_bos,
            drop_last=drop_last,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)
        und_token_indices = torch.tensor(und_token_indices, dtype=torch.long)
        gen_token_indices = torch.tensor(gen_token_indices, dtype=torch.long)

        cond_timestep_scatter_index = torch.tensor(extra_token_pos['cond_timestep'], dtype=torch.long) \
            if 'cond_timestep' in extra_token_pos else None
        gen_timestep_scatter_index = torch.tensor(extra_token_pos['gen_timestep'], dtype=torch.long) \
            if 'gen_timestep' in extra_token_pos else None
        gen_image_slices, gen_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'img', full_seq_token_tensor)
        vae_image_slices, vae_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'vae_img', full_seq_token_tensor)
        vit_image_slices, vit_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'vit_img', full_seq_token_tensor)
        joint_image_slices, _ = self.parse_extra_token_pos(
            extra_token_pos, 'joint_img', full_seq_token_tensor)
        all_image_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<all_img>_start'], extra_token_pos['<all_img>_end'])
        ] if '<all_img>_start' in extra_token_pos and '<all_img>_end' in extra_token_pos else []

        text_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<text>_start'], extra_token_pos['<text>_end'])
        ] if '<text>_start' in extra_token_pos and '<text>_end' in extra_token_pos else []
        assert len(text_slices) <= len(text_mask_specs), \
            (f"Number of text slices ({len(text_slices)}) should be less than or equal to "
             f"number of text mask specs ({len(text_mask_specs)})")
        if use_text_mask:
            text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
            for text_slice, mask_spec in zip(text_slices, text_mask_specs):
                if not mask_spec['ignore'] or disable_ignore:
                    real_slice = slice(
                        text_slice.start + mask_spec['start_offset'],
                        text_slice.stop + mask_spec['end_offset']
                    )
                    text_mask[real_slice] = 1.0
        else:
            text_mask = None

        real_pos = torch.tensor(extra_token_pos.get('first_pad', [full_seq_token_tensor.shape[0]]), dtype=torch.long)

        if len(und_token_type) == 0 and len(gen_token_type) == 0:
            und_token_indices = None
            gen_token_indices = None

        return TokenizerEncodeOutput(
            tokens=full_seq_token_tensor,
            text_slices=text_slices,
            gen_image_slices=gen_image_slices,
            vae_image_slices=vae_image_slices,
            vit_image_slices=vit_image_slices,
            joint_image_slices=joint_image_slices,
            all_image_slices=all_image_slices,
            text_mask=text_mask,
            gen_image_mask=gen_image_mask,
            vae_image_mask=vae_image_mask,
            vit_image_mask=vit_image_mask,
            real_pos=real_pos,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
        )

    def apply_general_template(
            self,
            message_list,
            conv_template,
            max_length=None,
            add_assistant_prefix=False,
            answer="auto",
            bot_task="auto",
            sequence_template="instruct",
            uncond_p=0.0,
            cfg_factor=1,
            batchify=False,
            image_base_size=None,
            und_token_type=None,
            gen_token_type=None,
            use_text_mask=False,
    ):
        if bot_task == "img_ratio":
            assert image_base_size is not None, "image_base_size should be provided for img_ratio task."

        if batchify:
            assert isinstance(message_list[0], list), \
                f"When batchify is True, message_list should be a list of list, but got [{type(message_list[0])}, ...]."
            return self.batch_gen_infer(
                infer_fn=self.apply_general_template,
                infer_fn_kwargs_list=[dict(
                    message_list=message_list_i,
                    conv_template=conv_template,
                    max_length=max_length,
                    add_assistant_prefix=add_assistant_prefix,
                    answer=answer,
                    bot_task=bot_task,
                    sequence_template=sequence_template,
                    image_base_size=image_base_size,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    use_text_mask=use_text_mask,
                ) for message_list_i in message_list],
                do_classifier_free_guidance=cfg_factor > 1,
                uncondition_repeat_times=cfg_factor - 1,
                und_token_type=und_token_type,
                gen_token_type=gen_token_type,
            )

        uncond_kwargs = dict(
            uncond_enabled=uncond_p == 1.0,
            uncond_p=uncond_p,
        )

        def process_successive_message(_message_list, _cur_message_idx, role, prefix, suffix,
                                       answer_prefix="", answer_suffix=""):
            _sub_sections = []
            while _cur_message_idx < len(message_list) and _message_list[_cur_message_idx]['role'] == role:
                message = _message_list[_cur_message_idx]
                if message['type'] == 'text':
                    text = message['content']
                    if role == "system":
                        _sub_sections.append(dict(type="text", text=text))
                    elif role == "assistant":
                        _sub_sections.append(dict(
                            type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs))
                    else:
                        _sub_sections.append(dict(type="text", text=text, **uncond_kwargs))
                elif message['type'] == 'gen_image':
                    info = message['content']
                    assert isinstance(info, ImageInfo), f"Expected ImageInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message['type'] in ['cond_joint_image', 'cond_vae_image', 'cond_vit_image']:
                    info = message['content']
                    assert isinstance(info, (ImageInfo, JointImageInfo)), \
                        f"Expected ImageInfo or JointImageInfo, but got {type(info)}"
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                else:
                    raise ValueError(f"Unknown message type: {message['type']}")
                _cur_message_idx += 1
            if len(_sub_sections) > 0:
                _sub_sections.insert(0, dict(type='text', text=prefix))
                _sub_sections.append(dict(type='text', text=suffix))
            return _sub_sections, _cur_message_idx

        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = self.answer_token, self.end_of_answer_token
        else:
            answer_prefix, answer_suffix = "", ""
        if sequence_template == "pretrain":
            system_suffix = conv_template.pretrain_sep_sp
            user_prefix = conv_template.get_role_prefix(conv_template.pretrain_roles[0])
            user_suffix = conv_template.pretrain_sep
            bot_prefix = conv_template.get_role_prefix(conv_template.pretrain_roles[1])
            bot_suffix = conv_template.pretrain_sep2
        else:
            system_suffix = conv_template.sep_sp
            user_prefix = conv_template.get_role_prefix(conv_template.roles[0])
            user_suffix = f"{conv_template.sep}"
            bot_prefix = conv_template.get_role_prefix(conv_template.roles[1])
            bot_suffix = f"{conv_template.sep2}"

        sections = []
        cur_message_idx = 0
        final_role = None
        while cur_message_idx < len(message_list):
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="system", prefix="", suffix=system_suffix)
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "system"

            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="user", prefix=user_prefix, suffix=user_suffix)
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "user"

            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="assistant", prefix=bot_prefix, suffix=bot_suffix,
                answer_prefix=answer_prefix, answer_suffix=answer_suffix,
            )
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "assistant"

        if add_assistant_prefix:
            if final_role == "assistant":
                _bot_prefix = ""
                if len(sections) > 0 and sections[-1]['type'] == 'text' and sections[-1]['text'] == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            bot_response_prefix = dict(
                auto=lambda: f"{_bot_prefix}{answer_prefix}",
                image=lambda: "",
                img_ratio=lambda: f"{_bot_prefix}{answer_prefix}{self.boi_token}{self.size_token(image_base_size)}",
            )[bot_task]()
            sections.append(dict(type='text', text=bot_response_prefix))

        if und_token_type is None:
            und_token_type = []
        if gen_token_type is None:
            gen_token_type = []

        output = self.encode_general(
            sections=sections,
            use_text_mask=use_text_mask,
            add_eos=conv_template.add_eos,
            add_pad=conv_template.add_pad,
            add_bos=conv_template.add_bos,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
        )

        if max_length is not None:
            if output.tokens.shape[-1] > max_length:
                raise ValueError(
                    f"Encoded token length {output.tokens.shape[-1]} exceeds max_length {max_length}.\n"
                    f"Please set a larger max_length or check the input messages:\n{message_list}"
                )

        return output, sections

    def apply_chat_template(
            self,
            batch_prompt: Optional[list[str]] = None,
            batch_message_list: Optional[list[list[dict[str, Any]]]] = None,
            mode: str = "gen_text",
            batch_gen_image_info: Optional[list[ImageInfo]] = None,
            batch_cond_images: Optional[Union[list[CondImage], list[list[CondImage]]]] = None,
            max_length: Optional[int] = None,
            bot_task: str = "auto",
            image_base_size: Optional[int] = None,
            sequence_template: str = "pretrain",
            cfg_factor: int = 1,
            add_assistant_prefix: Optional[bool] = None,
            conv_template: Optional[Conversation] = None,
            und_token_type: list[str] = None,
            gen_token_type: list[str] = None,
            use_text_mask: bool = False,
            **kwargs,
    ) -> dict[str, Any]:
        allowed_tasks = ["image", "auto", "img_ratio"]
        assert bot_task in allowed_tasks, f"bot_task should be one of {allowed_tasks}, but got {bot_task}."

        if batch_message_list is None:
            batch_size = len(batch_prompt)

            if not isinstance(batch_gen_image_info, list):
                batch_gen_image_info = [batch_gen_image_info] * batch_size
            if batch_cond_images is not None:
                assert len(batch_cond_images) == batch_size, \
                    (f"batch_cond_image_info should have the same length as batch_size ({batch_size}), "
                     f"but got {len(batch_cond_images)}.")
                batch_cond_images = [
                    cond_images if isinstance(cond_images, list) else [cond_images]
                    for cond_images in batch_cond_images
                ]
            else:
                batch_cond_images = [[] for _ in range(batch_size)]

            batch_message_list = []
            for prompt, gen_image_info, cond_images in zip(
                    batch_prompt, batch_gen_image_info,
                    batch_cond_images,
            ):
                message_list = []
                if len(cond_images) > 0:
                    message_list.extend([
                        dict(role="user", type=cond_image.section_type, content=cond_image.i)
                        for cond_image in cond_images
                    ])
                message_list.append(dict(role="user", type="text", content=prompt))
                if mode == "gen_image":
                    message_list.append(dict(
                        role="assistant", type="gen_image", content=gen_image_info))
                batch_message_list.append(message_list)

        output, sections = self.apply_general_template(
            message_list=batch_message_list,
            conv_template=conv_template,
            max_length=max_length,
            add_assistant_prefix=default(add_assistant_prefix, mode == "gen_text"),
            bot_task=bot_task,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor,
            batchify=True,
            image_base_size=image_base_size,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
            use_text_mask=use_text_mask,
            **kwargs,
        )
        return dict(output=output, sections=sections)
    def pad(self, tensor_list, dim=0, pad_val=None, key=None):
        if pad_val is None:
            pad_val = self.pad_token_id
        max_len = max([t.shape[dim] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            if t.shape[dim] < max_len:
                assert pad_val is not False, f"Not allowed/implemented pad for key: {key}"
                t = F.pad(t, (0, max_len - t.shape[dim]), value=pad_val)
            padded_tensor_list.append(t)
        return padded_tensor_list

    def batch_gen_infer(
            self,
            infer_fn,
            infer_fn_kwargs_list: list[dict[str, int]] = None,
            do_classifier_free_guidance=False,
            uncondition_repeat_times: int = 1,
            und_token_type: Optional[list] = None,
            gen_token_type: Optional[list] = None,
    ):
        cond_results_list = None
        uncond_results_list = None

        for infer_fn_kwargs in infer_fn_kwargs_list:
            cond_kwargs = {"uncond_p": 0.0} if do_classifier_free_guidance else {}
            results = infer_fn(
                **infer_fn_kwargs,
                **cond_kwargs,
            )
            assert isinstance(results, tuple), f"Expected tuple output from tokenizer template, got {type(results)}."
            if cond_results_list is None:
                cond_results_list = [[] for _ in results]
                uncond_results_list = [[] for _ in results]
            for i, result in enumerate(results):
                cond_results_list[i].append(result)

            if do_classifier_free_guidance:
                uncond_results = infer_fn(
                    **infer_fn_kwargs,
                    uncond_p=1.0,
                )
                if isinstance(uncond_results, TokenizerEncodeOutput):
                    uncond_results_list.append(uncond_results)
                else:
                    for i, result in enumerate(uncond_results):
                        uncond_results_list[i].append(result)

        def make_batch(batch_cond_item, batch_uncond_item):
            first = batch_cond_item[0]
            if isinstance(first, (list, tuple)):
                stacked_item = batch_cond_item + batch_uncond_item * uncondition_repeat_times

            elif isinstance(first, TokenizerEncodeOutput):
                stacked_item = {}
                for key in list(first.keys()):
                    merged_list = [cond_item[key] for cond_item in batch_cond_item] + \
                        [uncond_item[key] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    if isinstance(first[key], torch.Tensor):
                        if 'mask' in key:
                            pad_val = 0.0
                        elif key == 'tokens':
                            pad_val = self.pad_token_id
                        elif key in ['und_token_indices', 'gen_token_indices']:
                            continue
                        else:
                            pad_val = False
                        if key not in ('und_token_indices', 'gen_token_indices'):
                            stacked_item[key] = torch.stack(self.pad(merged_list, pad_val=pad_val, key=key), dim=0)
                    elif isinstance(first[key], list):
                        stacked_item[key] = merged_list
                    elif first[key] is None:
                        pass
                    else:
                        raise ValueError(f"Unsupported type of {key}: {type(first[key])}.")

                stacked_item = TokenizerEncodeOutput(stacked_item)

                if 'und_token_indices' in first.keys() and first['und_token_indices'] is not None and 'gen_token_indices' in first.keys() and first['gen_token_indices'] is not None:
                    und_token_indices_merged_list = [cond_item['und_token_indices'] for cond_item in batch_cond_item] + [uncond_item['und_token_indices'] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    gen_token_indices_merged_list = [cond_item['gen_token_indices'] for cond_item in batch_cond_item] + [uncond_item['gen_token_indices'] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    sequence_length = stacked_item["tokens"].shape[1]

                    max_index = [max(und_token_indices_merged_list[i].max().item(), gen_token_indices_merged_list[i].max().item()) for i in range(len(und_token_indices_merged_list))]
                    for i, (und_token_indices_item, max_index_item) in enumerate(zip(und_token_indices_merged_list, max_index)):
                        if max_index_item == sequence_length - 1:
                            continue
                        und_token_indices_merged_list[i] = torch.cat([und_token_indices_item, torch.arange(max_index_item + 1, sequence_length)])

                    max_gen_count = max(g.shape[0] for g in gen_token_indices_merged_list)

                    max_extra_needed = 0
                    for i in range(len(gen_token_indices_merged_list)):
                        pad_needed = max_gen_count - gen_token_indices_merged_list[i].shape[0]
                        pad_available = max(0, sequence_length - 1 - max_index[i])
                        max_extra_needed = max(max_extra_needed, pad_needed - pad_available)

                    if max_extra_needed > 0:
                        for key in list(stacked_item.keys()):
                            if key == 'tokens':
                                stacked_item[key] = F.pad(stacked_item[key], (0, max_extra_needed), value=self.pad_token_id)
                            elif 'mask' in key and isinstance(stacked_item[key], torch.Tensor):
                                stacked_item[key] = F.pad(stacked_item[key], (0, max_extra_needed), value=0.0)
                        new_positions = torch.arange(sequence_length, sequence_length + max_extra_needed)
                        for i in range(len(und_token_indices_merged_list)):
                            und_token_indices_merged_list[i] = torch.cat([und_token_indices_merged_list[i], new_positions])
                        sequence_length += max_extra_needed

                    for i in range(len(gen_token_indices_merged_list)):
                        pad_needed = max_gen_count - gen_token_indices_merged_list[i].shape[0]
                        if pad_needed > 0:
                            moved = und_token_indices_merged_list[i][-pad_needed:]
                            und_token_indices_merged_list[i] = und_token_indices_merged_list[i][:-pad_needed]
                            gen_token_indices_merged_list[i] = torch.cat([gen_token_indices_merged_list[i], moved])

                    stacked_item['und_token_indices'] = torch.stack(und_token_indices_merged_list, dim=0)
                    stacked_item['gen_token_indices'] = torch.stack(gen_token_indices_merged_list, dim=0)

                elif ('und_token_indices' in first.keys() and first['und_token_indices'] is not None) or ('gen_token_indices' in first.keys() and first['gen_token_indices'] is not None):
                    raise ValueError(f"Only one of 'und_token_indices' and 'gen_token_indices' exists.")

                stacked_item = TokenizerEncodeOutput(stacked_item)
            else:
                raise TypeError(f"Making batch on type {type(first)} is not supported.")

            return stacked_item

        stacked_outputs = []
        for cond_results, uncond_results in zip(cond_results_list, uncond_results_list):
            stacked_outputs.append(make_batch(cond_results, uncond_results))

        return tuple(stacked_outputs)
class Qwen3BaseTokenizerFast(BaseMultimodalTokenizerFast):
    def setup_special_tokens(self):
        predefined_name_mapping = {
            "bos": "<|im_start|>",
            "eos": "<|im_end|>",
            "answer": "",
            "end_of_answer": "",
            "boi": "<|vision_start|>",
            "eoi": "<|vision_end|>",
            "img": "<|image_pad|>",
        }
        for name, mapping in predefined_name_mapping.items():
            setattr(self, f"{name}_token", mapping)
            setattr(self, f"{name}_token_id", self.convert_tokens_to_ids(mapping))

        if len(self._sp_dict) > 0:
            name_mapping = dict(
                cfg_token="<｜cfg｜>",
                timestep_token="<｜timestep｜>",
                joint_img_sep_token="<｜joint_img_sep｜>",
            )
            for name, token in name_mapping.items():
                setattr(self, name, token)
                setattr(self, f"{name}_id", self._sp_dict.get(token))

def load_tokenizer(
        tokenizer_name: str,
        tokenizer_class: str,
) -> "BaseMultimodalTokenizerFast":
    assert '.' in tokenizer_class, (
        f"Invalid tokenizer class: {tokenizer_class}. A valid tokenizer name should be in the form of "
        f"<module_name>.<tokenizer_cls>."
    )

    if tokenizer_class.startswith("transformers."):
        module_name, tokenizer_cls = tokenizer_class.rsplit('.', 1)
        module_spec = importlib.import_module(module_name)
        TokenizerSpec = getattr(module_spec, tokenizer_cls)     # noqa
    else:
        module_name, tokenizer_cls = tokenizer_class.rsplit('.', 1)
        if module_name == "tokenizer":
            module_spec = importlib.import_module("rosetta.tokenizer")
        else:
            module_spec = importlib.import_module(module_name)
        TokenizerSpec = getattr(module_spec, tokenizer_cls)     # noqa

    if tokenizer_name in TOKENIZER_PATH:
        tokenizer_name = TOKENIZER_PATH[tokenizer_name]

    return TokenizerSpec.from_pretrained(tokenizer_name)

