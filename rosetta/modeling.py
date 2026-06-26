import math
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import Cache, StaticCache
from transformers.utils.generic import ModelOutput

from .autoencoder import (
    TimestepEmbedder,
    normal_weight_reset_parameters,
    project_in_layer,
    project_out_layer,
)
from .configuration import MultimodalConfig, TransformerConfig
from .utils import PRECISION_TO_TYPE, default, is_package_version


@dataclass
class MultimodalModelOutput(ModelOutput):
    losses: Optional[dict] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    diffusion_prediction: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def get_vision_position_ids(
        start_position: int,
        grid_thw: List[int],
        spatial_merge_size: int = 1,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    t, h, w = grid_thw[0], grid_thw[1], grid_thw[2]
    if isinstance(t, torch.Tensor):
        t, h, w = t.item(), h.item(), w.item()
    llm_grid_t = t
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
    vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)
    return vision_position_ids


def get_text_position_ids(
        length: int,
        start_position: int = 0,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.arange(length, device=device).view(1, -1).expand(3, -1) + start_position


def get_interleaved_mrope_index(
        image_infos: List[Optional[List[Tuple[slice, Tuple[int, int], dict]]]],
        seq_len: int,
        spatial_merge_size: int,
        sample_offsets: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    if image_infos is None:
        image_infos = [None]
    batch_size = len(image_infos)
    position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.int64, device=device)

    for i, image_info in enumerate(image_infos):
        llm_pos_ids_list = []
        current_pos = 0
        st = 0

        if image_info is None:
            image_info = []

        for sec_slice, (h, w), _ in image_info:
            img_start = sec_slice.start
            text_len = img_start - st

            if text_len > 0:
                llm_pos_ids_list.append(get_text_position_ids(text_len, current_pos, device))
                current_pos += text_len

            grid_thw = [1, h, w]
            llm_pos_ids_list.append(
                get_vision_position_ids(current_pos, grid_thw, spatial_merge_size=spatial_merge_size, device=device)
            )
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            current_pos += max(llm_grid_h, llm_grid_w)
            st = img_start + llm_grid_h * llm_grid_w

        if st < seq_len:
            llm_pos_ids_list.append(get_text_position_ids(seq_len - st, current_pos, device))

        if len(llm_pos_ids_list) == 0:
            llm_pos_ids_list.append(get_text_position_ids(seq_len, 0, device))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

        if llm_positions.shape[1] < seq_len:
            padding = llm_positions[:, -1:].expand(3, seq_len - llm_positions.shape[1])
            llm_positions = torch.cat([llm_positions, padding], dim=1)
        elif llm_positions.shape[1] > seq_len:
            llm_positions = llm_positions[:, :seq_len]

        position_ids[:, i, :] = llm_positions

        if sample_offsets is not None and sample_offsets[i] is not None:
            offsets = sample_offsets[i].tolist()
            if len(offsets) >= 2:
                assert offsets[0] == 0, "First offset must be 0"
                assert offsets[-1] <= seq_len, "Last offset must be less than or equal to seq_len"
                for start, end in zip(offsets[:-1], offsets[1:]):
                    assert end > start, "End must be greater than start"
                    seg_base = position_ids[:, i, start].clone()
                    position_ids[:, i, start:end] = position_ids[:, i, start:end] - seg_base.view(3, 1)


    return position_ids


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def get_batch_interleaved_mrope(
        image_infos: List[List[Tuple[slice, Tuple[int, int]]]],
        seq_len: int,
        n_elem: int,
        mrope_section: List[int],
        device: Optional[torch.device] = None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        spatial_merge_size: int = 1,
        sample_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    position_ids = get_interleaved_mrope_index(image_infos, seq_len, spatial_merge_size, sample_offsets, device)
    position_ids = position_ids.to(device)

    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))

    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    inv_freq_expanded = theta[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()

    device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    return cos, sin


def apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim=-3,
) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if cos.dim() > 1:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    return (x * cos) + (rotated * sin)

class CachedRoPE(object):
    def __init__(self, config):
        self._config = config
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_media_info = None

    @torch.autocast(device_type='cuda', enabled=False)
    def __call__(self, seq_len, device, rope_media_info=None, input_pos=None, sample_offsets=None):
        if (self.seq_len != seq_len) or (self.rope_media_info != rope_media_info):
            self.cos_cache, self.sin_cache = get_batch_interleaved_mrope(
                image_infos=rope_media_info,
                seq_len=seq_len,
                mrope_section=self._config.mrope_section,
                n_elem=self._config.attention_head_size,
                device=device,
                base=self._config.rope_theta,
                base_rescale_factor=1.0,
                sample_offsets=sample_offsets
            )
        if input_pos is None:
            cos, sin = self.cos_cache, self.sin_cache
        else:
            assert input_pos.dim() == 2, f"{input_pos.shape=}"
            head_size = self.cos_cache.size(-1)
            cos = torch.gather(self.cos_cache, dim=1, index=input_pos.unsqueeze(-1).expand(-1, -1, head_size))
            sin = torch.gather(self.sin_cache, dim=1, index=input_pos.unsqueeze(-1).expand(-1, -1, head_size))

        return cos, sin

# Attention and KV cache layers
flex_attention = torch.compile(flex_attention, dynamic=False)


BatchRaggedMedia = Union[torch.Tensor, list[Union[torch.Tensor, list[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, list[torch.Tensor]]


class MultimodalStaticCache(StaticCache):
    def __init__(self, *args, **kwargs):
        self.dynamic = kwargs.pop("dynamic", False)
        super().__init__(*args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        if is_package_version("transformers", "<=", "4.53.3"):
            if self.key_cache[layer_idx].device != key_states.device:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            k_out = self.key_cache[layer_idx]  # max_batch_size x num_key_value_heads x max_cache_len x head_dim
            v_out = self.value_cache[layer_idx]  # max_batch_size x num_key_value_heads x max_cache_len x head_dim
        else:
            if self.layers[layer_idx].keys is None:
                self.layers[layer_idx].lazy_initialization(value_states)
            k_out = self.layers[layer_idx].keys
            v_out = self.layers[layer_idx].values
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

                if self.dynamic:
                    end = cache_position[-1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]
            else:
                # first step of gen_text or all steps of gen_image
                assert cache_position.dim() == 2, f"multiple batch dims not yet {cache_position.shape=}"
                batch_size, idx_size = cache_position.shape
                assert batch_size == k_out.size(0)
                assert batch_size == v_out.size(0)
                assert batch_size == key_states.size(0)
                assert batch_size == value_states.size(0)
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                    v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

                if self.dynamic:
                    assert len(cache_position) == 1
                    # end = cache_position[0, -1].item() + 1
                    end = int(cache_position[0, -1]) + 1  # Tensor.item()导致图中断
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]

        return k_out, v_out


def get_device(tensor: BatchRaggedMedia):
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    elif isinstance(tensor, list):
        return get_device(tensor[0])
    else:
        raise ValueError(f"Unsupported type for get_device: {type(tensor)}")


class CausalSelfAttention(nn.Module):
    def __init__(
            self,
            config: MultimodalConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.attention_head_size,
            bias=False, **factory_kwargs
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.attention_head_size,
            bias=False, **factory_kwargs
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.attention_head_size,
            bias=False, **factory_kwargs
        )

        self.o_proj = nn.Linear(
            config.attention_head_size * config.num_attention_heads, config.hidden_size,
            bias=False, **factory_kwargs
        )

        self.query_layernorm = config.norm_class(config.attention_head_size, eps=config.norm_eps, **factory_kwargs)
        self.key_layernorm = config.norm_class(config.attention_head_size, eps=config.norm_eps, **factory_kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            past_key_values: Optional[MultimodalStaticCache] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        new_q = self.q_proj(hidden_states).view(bsz, seqlen, n_kv_head, q_per_kv, head_size)
        new_k = self.k_proj(hidden_states).view(bsz, seqlen, n_kv_head, 1, head_size)
        new_v = self.v_proj(hidden_states).view(bsz, seqlen, n_kv_head, 1, head_size)
        q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [new_q, new_k, new_v])

        q = q.reshape(bsz, -1, seqlen, head_size)  # (B, n_q_head, T, hs)
        k = k.reshape(bsz, -1, seqlen, head_size)  # (B, n_kv_head, T, hs)
        v = v.reshape(bsz, -1, seqlen, head_size)  # (B, n_kv_head, T, hs)

        q = self.query_layernorm(q)
        k = self.key_layernorm(k)

        q = apply_rope(q, *rotary_position_embeddings)
        k = apply_rope(k, *rotary_position_embeddings)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if input_pos is not None:
            cache_kwargs = {"cache_position": input_pos}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = k.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if n_kv_head != n_q_head and (input_pos is None or q_per_kv != 1):
            k = k.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            v = v.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        y = self.scaled_dot_product_attention(q, k, v, attention_mask)

        y = y.reshape(bsz, seqlen, head_size * n_q_head)  # re-assemble all head outputs side by side

        # output projection
        return self.o_proj(y)

    def scaled_dot_product_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # q, k, v: (bsz, n_head, seqlen, head_size)
        scale = 1.0 / math.sqrt(self._config.attention_head_size)

        if isinstance(mask, BlockMask):
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            y = flex_attention(q, k, v, block_mask=mask, scale=scale)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale,
                # If q only has one token (typically in AR model decoding stage), we should use full attention.
                is_causal=mask is None and q.size(2) > 1
            )

        return y.transpose(1, 2)


class CausalSelfAttentionMoT(CausalSelfAttention):
    def __init__(
            self,
            config: MultimodalConfig,
            config_mot_gen: MultimodalConfig,
            layer_idx: int,
            mot_und_frozen: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(config, layer_idx, dtype, device)
        self.mot_und_frozen = mot_und_frozen

        self.q_proj_mot_gen = nn.Linear(
            config_mot_gen.hidden_size, config_mot_gen.num_attention_heads * config_mot_gen.attention_head_size,
            bias=False, **factory_kwargs
        )
        self.k_proj_mot_gen = nn.Linear(
            config_mot_gen.hidden_size, config_mot_gen.num_kv_heads * config_mot_gen.attention_head_size,
            bias=False, **factory_kwargs
        )
        self.v_proj_mot_gen = nn.Linear(
            config_mot_gen.hidden_size, config_mot_gen.num_kv_heads * config_mot_gen.attention_head_size,
            bias=False, **factory_kwargs
        )

        self.o_proj_mot_gen = nn.Linear(
            config_mot_gen.attention_head_size * config_mot_gen.num_attention_heads, config_mot_gen.hidden_size,
            bias=False, **factory_kwargs
        )

        self.query_layernorm_mot_gen = config_mot_gen.norm_class(config_mot_gen.attention_head_size, eps=config_mot_gen.norm_eps, **factory_kwargs)
        self.key_layernorm_mot_gen = config_mot_gen.norm_class(config_mot_gen.attention_head_size, eps=config_mot_gen.norm_eps, **factory_kwargs)

        if self.mot_und_frozen:
            self.q_proj.eval()
            self.q_proj.requires_grad_(False)
            self.k_proj.eval()
            self.k_proj.requires_grad_(False)
            self.v_proj.eval()
            self.v_proj.requires_grad_(False)
            self.o_proj.eval()
            self.o_proj.requires_grad_(False)
            self.query_layernorm.eval()
            self.query_layernorm.requires_grad_(False)
            self.key_layernorm.eval()
            self.key_layernorm.requires_grad_(False)

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        past_key_values: Optional[MultimodalStaticCache] = None,
        und_token_indices: Optional[torch.Tensor] = None,
        gen_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        und_hidden_states, gen_hidden_states = hidden_states

        bsz = und_hidden_states.shape[0]
        und_seqlen, gen_seqlen = und_hidden_states.shape[1], gen_hidden_states.shape[1]
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        q = self.q_proj(und_hidden_states).view(bsz, -1, n_kv_head, q_per_kv, head_size)
        k = self.k_proj(und_hidden_states).view(bsz, -1, n_kv_head, 1, head_size)
        v = self.v_proj(und_hidden_states).view(bsz, -1, n_kv_head, 1, head_size)

        gen_q = self.q_proj_mot_gen(gen_hidden_states).view(bsz, -1, n_kv_head, q_per_kv, head_size)
        gen_k = self.k_proj_mot_gen(gen_hidden_states).view(bsz, -1, n_kv_head, 1, head_size)
        gen_v = self.v_proj_mot_gen(gen_hidden_states).view(bsz, -1, n_kv_head, 1, head_size)

        q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [q, k, v])
        gen_q, gen_k, gen_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [gen_q, gen_k, gen_v])

        # [bsz, h, seqlen, head_size]
        q = q.reshape(bsz, n_q_head, und_seqlen, head_size)
        k = k.reshape(bsz, n_kv_head, und_seqlen, head_size)
        v = v.reshape(bsz, n_kv_head, und_seqlen, head_size)
        gen_q = gen_q.reshape(bsz, n_q_head, gen_seqlen, head_size)
        gen_k = gen_k.reshape(bsz, n_kv_head, gen_seqlen, head_size)
        gen_v = gen_v.reshape(bsz, n_kv_head, gen_seqlen, head_size)

        # Scatter understanding and generation tokens for rope
        und_token_indices_q = und_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
        gen_token_indices_q = gen_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
        und_token_indices_kv = und_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
        gen_token_indices_kv = gen_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))

        def _scatter(und_src, gen_src, und_token_indices, gen_token_indices, n_head):
            target = torch.zeros((bsz, n_head, und_seqlen+gen_seqlen, head_size), dtype=und_src.dtype, device=und_src.device)
            target.scatter_(dim=2, index=und_token_indices, src=und_src)
            target.scatter_(dim=2, index=gen_token_indices, src=gen_src)
            return target

        q_merge = _scatter(q, gen_q, und_token_indices_q, gen_token_indices_q, n_q_head)
        k_merge = _scatter(k, gen_k, und_token_indices_kv, gen_token_indices_kv, n_kv_head)
        v_merge = _scatter(v, gen_v, und_token_indices_kv, gen_token_indices_kv, n_kv_head)

        q_ = torch.zeros_like(q_merge)
        k_ = torch.zeros_like(k_merge)

        q_.scatter_(dim=2, index=und_token_indices_q, src=self.query_layernorm(q_merge.gather(2, und_token_indices_q)).to(q_.dtype))
        q_.scatter_(dim=2, index=gen_token_indices_q, src=self.query_layernorm_mot_gen(q_merge.gather(2, gen_token_indices_q)).to(q_.dtype))
        k_.scatter_(dim=2, index=und_token_indices_kv, src=self.key_layernorm(k_merge.gather(2, und_token_indices_kv)).to(k_.dtype))
        k_.scatter_(dim=2, index=gen_token_indices_kv, src=self.key_layernorm_mot_gen(k_merge.gather(2, gen_token_indices_kv)).to(k_.dtype))

        q_merge = q_
        k_merge = k_

        # apply rotary position embeddings
        q_merge = apply_rope(q_merge, *rotary_position_embeddings)
        k_merge = apply_rope(k_merge, *rotary_position_embeddings)

        q_merge = q_merge.to(v_merge.dtype)
        k_merge = k_merge.to(v_merge.dtype)

        # Restore from kv_cache and update
        if input_pos is not None:
            cache_kwargs = {"cache_position": input_pos}
            k_merge, v_merge = past_key_values.update(k_merge, v_merge, self.layer_idx, cache_kwargs)
        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = k_merge.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if n_kv_head != n_q_head and (input_pos is None or q_per_kv != 1):
            k_merge = k_merge.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            v_merge = v_merge.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        y = self.scaled_dot_product_attention(q_merge, k_merge, v_merge, attention_mask)

        # re-assemble all head outputs side by side
        y = y.reshape(bsz, -1, head_size * n_q_head)

        core_attn_out = y.gather(dim=1, index=und_token_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))
        gen_core_attn_out = y.gather(dim=1, index=gen_token_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))

        und_hidden_states = self.o_proj(core_attn_out)
        gen_hidden_states = self.o_proj_mot_gen(gen_core_attn_out)
        return und_hidden_states, gen_hidden_states

class MLP(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: int,
            is_shared_mlp: bool = False,
            is_moe: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.ffn_hidden_size = config.ffn_hidden_size

        if is_shared_mlp or is_moe:
            self.ffn_hidden_size = config.moe_ffn_hidden_size

        self.gate_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=False, **factory_kwargs)
        self.up_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=False, **factory_kwargs)
        self.down_proj = nn.Linear(self.ffn_hidden_size, config.hidden_size, bias=False, **factory_kwargs)
        self.act_fn = config.act_class()

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        out = self.down_proj(up * self.act_fn(gate))
        return out


class DeepSeekMoEGate(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk

        if config.use_modality_routing:
            assert config.num_text_experts > 0 and config.num_vit_experts > 0 and config.num_vae_experts > 0, \
                "num_text_experts / num_vit_experts / num_vae_experts must be set when use_modality_routing=True"
            assert config.num_text_experts + config.num_vit_experts + config.num_vae_experts == self.num_experts, \
                f"num_text({config.num_text_experts}) + num_vit({config.num_vit_experts}) + num_vae({config.num_vae_experts}) " \
                f"must equal num_experts({self.num_experts})"
            self.wg_text = nn.Linear(config.hidden_size, config.num_text_experts, bias=False, **factory_kwargs)
            self.wg_vit  = nn.Linear(config.hidden_size, config.num_vit_experts,  bias=False, **factory_kwargs)
            self.wg_vae  = nn.Linear(config.hidden_size, config.num_vae_experts,  bias=False, **factory_kwargs)
        else:
            self.wg = nn.Linear(config.hidden_size, self.num_experts, bias=False, **factory_kwargs)

    def _score(self, logits):
        return F.softmax(logits.float(), dim=1)

    def _forward_modality_routing(self, flat_hidden, bsz, seqlen, token_modalities):
        N = bsz * seqlen
        cfg = self._config
        n_text = cfg.num_text_experts
        n_vit  = cfg.num_vit_experts
        n_vae  = cfg.num_vae_experts
        vit_offset = n_text
        vae_offset = n_text + n_vit

        flat_mod = token_modalities.reshape(-1)

        topk_weights = torch.zeros(N, self.top_k, dtype=torch.float32, device=flat_hidden.device)
        topk_idx     = torch.zeros(N, self.top_k, dtype=torch.long,    device=flat_hidden.device)

        for mod_id, wg, offset, n_local in (
            (0, self.wg_text, 0,          n_text),
            (1, self.wg_vit,  vit_offset, n_vit),
            (2, self.wg_vae,  vae_offset, n_vae),
        ):
            mask = (flat_mod == mod_id)
            if not mask.any():
                continue
            h_m      = flat_hidden[mask].to(wg.weight.dtype)
            logits_m = wg(h_m)
            scores_m = self._score(logits_m)

            tw_m, ti_m = torch.topk(scores_m, self.top_k, dim=-1)
            topk_weights[mask] = tw_m
            topk_idx[mask]     = ti_m + offset

        if self.top_k > 1:
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp(
                min=torch.finfo(topk_weights.dtype).eps
            )
            topk_weights = topk_weights / denom

        return topk_weights, topk_idx

    def forward(self, hidden_states, token_modalities=None):
        bsz, seqlen, hdim = hidden_states.size()
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))

        if self._config.use_modality_routing:
            if token_modalities is None:
                N = hidden_states.shape[0]
                token_modalities = torch.zeros(N, dtype=torch.long, device=hidden_states.device)
            return self._forward_modality_routing(hidden_states, bsz, seqlen, token_modalities)

        logits = self.wg(hidden_states.to(self.wg.weight.dtype))

        scores = F.softmax(logits, dim=1)
        topk_weights, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        if self.top_k > 1:
            denominator = topk_weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(topk_weights.dtype).eps)
            topk_weights = topk_weights / denominator

        return topk_weights, topk_idx


class DeepSeekMoE(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk
        self.gate = DeepSeekMoEGate(config, layer_idx, device=device, dtype=torch.float32)

        self.experts = nn.ModuleList(
            [MLP(config, layer_idx, is_moe=True, **factory_kwargs)
             for _ in range(self.num_experts)]
        )
        self.shared_mlp = MLP(config, layer_idx=layer_idx, is_shared_mlp=True, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, token_modalities: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, hdim = hidden_states.size()
        input_hidden_states = hidden_states

        with torch.autocast('cuda', enabled=False):
            topk_weights, topk_idx = self.gate(hidden_states, token_modalities)

        topk_weights = topk_weights.to(hidden_states.dtype)

        flat_topk_idx = topk_idx.view(-1)
        hidden_states = hidden_states.view(-1, hdim)
        hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)

        expert_outputs = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        for i in range(self.num_experts):
            expert_mask = (flat_topk_idx == i)
            selected_inputs = hidden_states[expert_mask]
            expert_output = self.experts[i](selected_inputs)
            expert_outputs[expert_mask] = expert_output.to(hidden_states.dtype)

        weighted_outputs = (expert_outputs.view(
            bsz * seqlen, self.top_k, hdim) * topk_weights.unsqueeze(-1)).sum(dim=1)
        weighted_outputs = weighted_outputs.to(hidden_states.dtype).view(bsz, seqlen, hdim)
        shared_out = self.shared_mlp(input_hidden_states)

        cfg = self._config
        if (
            token_modalities is not None
            and getattr(cfg, 'shield_step', 0) > 0
            and getattr(cfg, '_current_training_iter', 0) < cfg.shield_step
        ):
            text_mask = (token_modalities == 0).unsqueeze(-1)
            shared_out = shared_out * text_mask + shared_out.detach() * (~text_mask)

        return weighted_outputs + shared_out

class MultimodalDecoderLayer(nn.Module):
    def __init__(
            self,
            config: MultimodalConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = CausalSelfAttention(config, layer_idx, **factory_kwargs)
        self.input_layernorm = config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs)
        self.post_attention_layernorm = config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs)
        self.mlp = DeepSeekMoE(config, layer_idx, **factory_kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            past_key_values: Optional[MultimodalStaticCache] = None,
            und_token_indices: Optional[torch.Tensor] = None,
            gen_token_indices: Optional[torch.Tensor] = None,
            token_modalities: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            input_pos=input_pos,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_modalities=token_modalities) \
            if token_modalities is not None and hasattr(self.mlp, 'gate') and hasattr(self.mlp.gate, 'wg_text') \
            else self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class MultimodalMoTDecoderLayer(MultimodalDecoderLayer):
    def __init__(
            self,
            config: MultimodalConfig,
            layer_idx: int,
            mot_und_frozen: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            gen_config: Optional[MultimodalConfig] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        gen_config = gen_config if gen_config is not None else config
        super().__init__(config, layer_idx, dtype, device)
        self.mot_und_frozen = mot_und_frozen
        self.self_attn = CausalSelfAttentionMoT(config, gen_config, layer_idx, mot_und_frozen, **factory_kwargs)
        self.input_layernorm_mot_gen = gen_config.norm_class(
            gen_config.hidden_size, eps=gen_config.norm_eps, **factory_kwargs
        )
        self.post_attention_layernorm_mot_gen = gen_config.norm_class(
            gen_config.hidden_size, eps=gen_config.norm_eps, **factory_kwargs
        )

        self.mlp_mot_gen = DeepSeekMoE(gen_config, layer_idx, **factory_kwargs)

        if self.mot_und_frozen:
            self.input_layernorm.eval()
            self.input_layernorm.requires_grad_(False)
            self.post_attention_layernorm.eval()
            self.post_attention_layernorm.requires_grad_(False)
            self.mlp.eval()
            self.mlp.requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        past_key_values: Optional[MultimodalStaticCache] = None,
        und_token_indices: Optional[torch.Tensor] = None,
        gen_token_indices: Optional[torch.Tensor] = None,
    ):
        und_hidden_states, gen_hidden_states = hidden_states
        und_residual, gen_residual = und_hidden_states, gen_hidden_states

        # Pre-attn norm
        und_hidden_states = self.input_layernorm(und_hidden_states)
        gen_hidden_states = self.input_layernorm_mot_gen(gen_hidden_states)

        # Self attention
        core_attn_out = self.self_attn(
            (und_hidden_states, gen_hidden_states),
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            input_pos=input_pos,
            past_key_values=past_key_values,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
        )

        und_hidden_states, gen_hidden_states = core_attn_out
        und_hidden_states = und_residual + und_hidden_states
        gen_hidden_states = gen_residual + gen_hidden_states

        # Pre-mlp norm
        und_residual, gen_residual = und_hidden_states, gen_hidden_states
        und_hidden_states = self.post_attention_layernorm(und_hidden_states)
        gen_hidden_states = self.post_attention_layernorm_mot_gen(gen_hidden_states)

        # Mlp
        if und_hidden_states.nelement() != 0:
            und_hidden_states = self.mlp(und_hidden_states)
        if gen_hidden_states.nelement() != 0:
            gen_hidden_states = self.mlp_mot_gen(gen_hidden_states)

        und_hidden_states = und_residual + und_hidden_states
        gen_hidden_states = gen_residual + gen_hidden_states

        return (und_hidden_states, gen_hidden_states)

class MultimodalModelBase(nn.Module):
    def get_input_embeddings(self):
        return self.model["embed_tokens"]

    def get_output_embeddings(self):
        return getattr(self, "lm_head", None)

    def __post_init__(
            self,
            config: MultimodalConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            args: Namespace = None,
            initialize_weights: bool = True,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Set as protected member to avoid conflict with potential parent classes
        self._config = config
        if self._config.use_mot:
            self._config_mot_gen = self._config.to_mot_gen_config()
        self._dtype = dtype

        # For inference, args can be None
        self.args = args or Namespace()
        self.vit_frozen = getattr(args, "vit_frozen", True)
        self.vit_precision = PRECISION_TO_TYPE[default(getattr(args, "vit_precision", None), dtype)]
        self.mot_und_frozen = getattr(args, "mot_und_frozen", False)
        self.lm_frozen = getattr(args, "lm_frozen", False)
        self.moe_aux_loss_coeff = getattr(args, "moe_aux_loss_coeff", 0.0)

        # ======================================
        #     Define vae projector modules
        # ======================================
        if config.use_vae:
            vae_hidden_size = self._config_mot_gen.hidden_size if config.use_mot else config.hidden_size
            vae_config = self._config_mot_gen if config.use_mot else config
            if config.use_timestep_token:
                self.timestep_emb = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)

            # One for patch_embed and other for final_layer
            self.time_embed = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)
            self.time_embed_2 = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)

            self.patch_embed = project_in_layer(vae_config, **factory_kwargs)
            self.final_layer = project_out_layer(vae_config, **factory_kwargs)

        # ======================================
        #     Define vit and aligner modules
        # ======================================
        if config.use_vit:
            from .visual_encoder import load_vit

            self.vit = load_vit(
                vision_model_type=config.vit_type,
                vision_model_precision=self.vit_precision,
                device=device,
                require_grad=not self.vit_frozen,
                eval_mode=self.vit_frozen,
                vision_model_params=config.vit_config,
                # When vit_no_load_pretrained=False (default for new qwen3vl-vit types),
                # backbone weights are loaded from VISION_ENCODER_META_INFO path via __init__.py.
                # Set vit_no_load_pretrained=True only when VIT weights come from the main ckpt
                # (e.g. loading a full VLM checkpoint that already contains VIT weights).
                no_load_pretrained=getattr(args, "vit_no_load_pretrained", False),
            )
            self.vit_context = torch.no_grad if self.vit_frozen else nullcontext

        # ======================================
        #       Define language modules
        # ======================================

        if config.use_mot:
            self.model = nn.ModuleDict(
                dict(
                    embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, **factory_kwargs),
                    layers=nn.ModuleList([
                        MultimodalMoTDecoderLayer(
                            config, block_idx, self.mot_und_frozen,
                            gen_config=self._config_mot_gen,
                            **factory_kwargs
                        )
                        for block_idx in range(config.num_layers)
                    ]),
                    norm=config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs),
                )
            )
        else:
            self.model = nn.ModuleDict(
                dict(
                    embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, **factory_kwargs),
                    layers=nn.ModuleList([
                        MultimodalDecoderLayer(config, block_idx, **factory_kwargs)
                        for block_idx in range(config.num_layers)
                    ]),
                    norm=config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs),
                )
            )

        if self.lm_frozen:
            # Freeze entire LLM backbone: embed_tokens, all transformer layers, norm.
            # Used for Stage 2.1 where only the vision-language connector is trained.
            self.model.eval()
            self.model.requires_grad_(False)

        # ====================== Finish model building =====================

        # Initialize cached rope, supporting automatic cache update
        self.cached_rope = CachedRoPE(config)
        self.use_rope_sample_offsets = getattr(args, "use_rope_sample_offsets", False)

        # Initialize weights if needed
        self._prepare_reset_parameters()
        if initialize_weights:
            for name, module in self.named_modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def _prepare_reset_parameters(self):
        # Globally set Linear and Embedding init methods to normal
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.reset_parameters = normal_weight_reset_parameters(
                    std=self._config.init_std, bias_type="zeros").__get__(module)
        for name, module in self.named_modules():
            if hasattr(module, "prepare_reset_parameters"):
                module.prepare_reset_parameters()

    @property
    def dtype(self):
        """Get the dtype of the model parameters."""
        if self._dtype is not None:
            return self._dtype
        # Fallback to getting dtype from model parameters
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # If no parameters, try buffers
            try:
                return next(self.buffers()).dtype
            except StopIteration:
                # Default fallback
                return torch.float32

    def scatter_to_hidden_states(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        hidden_states: torch.Tensor,
        gen_hidden_states: Optional[torch.Tensor] = None,
        dim: int = 1,
        slice_idx: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        n_embd = src.shape[-1]
        hidden_target = hidden_states if slice_idx is None else hidden_states[slice_idx:slice_idx + 1]

        if n_embd == hidden_states.shape[-1]:
            hidden_target.scatter_(dim=dim, index=index, src=src)
        else:
            assert gen_hidden_states is not None, "gen_hidden_states is required when hidden dim of src and hidden_states differ"
            assert gen_hidden_states.shape[-1] == n_embd, \
                f"Expect gen_hidden_states and src to have same hidden_size, but got {gen_hidden_states.shape[-1]} and {n_embd}"
            gen_target = gen_hidden_states if slice_idx is None else gen_hidden_states[slice_idx:slice_idx + 1]
            gen_target.scatter_(dim=dim, index=index, src=src)

        return hidden_states, gen_hidden_states

    def instantiate_vae_image_tokens(
            self,
            hidden_states: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
            timesteps: BatchRaggedTensor,
            medias: BatchRaggedMedia,
            media_mask: torch.Tensor,
    ):
        if hidden_states is None:
            hidden_size = self._config_mot_gen.hidden_size if self._config.use_mot else self._config.hidden_size

            if isinstance(medias, list):
                emb_list = []
                for i in range(len(medias)):
                    t_i = timesteps[i:i+1] if isinstance(timesteps, torch.Tensor) else timesteps[i]
                    t_emb_i = self.time_embed(t_i)
                    img_emb_i, _, _ = self.patch_embed(medias[i], t_emb_i)
                    emb_list.append(img_emb_i)

                max_tokens = max(e.size(1) for e in emb_list)
                padded_img_emb = torch.zeros(len(medias), max_tokens, hidden_size, device=emb_list[0].device, dtype=emb_list[0].dtype)
                for i, emb in enumerate(emb_list):
                    padded_img_emb[i, :emb.size(1), :] = emb[0]

                timestep_emb = self.timestep_emb(timesteps).reshape(len(medias), -1, hidden_size)
                hidden_states = torch.cat([timestep_emb, padded_img_emb], dim=1)
                return hidden_states

            t_emb = self.time_embed(timesteps)
            image_emb, _, _ = self.patch_embed(medias, t_emb)
            hidden_size = self._config_mot_gen.hidden_size if self._config.use_mot else self._config.hidden_size
            timestep_emb = self.timestep_emb(timesteps).reshape(medias.size(0), -1, hidden_size)
            hidden_states = torch.cat([timestep_emb, image_emb], dim=1)
            return hidden_states

        if isinstance(hidden_states, tuple):
            hidden_states, gen_hidden_states = hidden_states
        else:
            gen_hidden_states = None

        bsz, seqlen, n_embd = hidden_states.shape
        assert isinstance(medias, (torch.Tensor, list)), f"images should be BatchRaggedMedia, got {type(medias)}"

        if isinstance(medias, torch.Tensor):
            assert medias.ndim in [4, 5], f"images should be a 4-D or 5-D tensor, got {medias.ndim}-D tensor"
            assert isinstance(timesteps, torch.Tensor), f"timesteps should be 1-D tensor, got {type(timesteps)}"

            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
            t_emb = self.time_embed(timesteps)     # (bsz, n_embd)
            media_seq, *_ = self.patch_embed(medias, t_emb)   # (bsz, num_patches, n_embd)
            media_index = index.masked_select(media_mask.bool()).reshape(bsz, -1)   # (bsz, num_patches)
            assert media_seq.size(1) == media_index.size(1), \
                f"image_seq ({list(media_seq.size())}) has inconsistent shape with index ({list(media_index.size())})"
            n_embd = media_seq.shape[-1]
            index_exp = media_index.unsqueeze(-1).repeat(1, 1, n_embd)
            hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                media_seq.to(hidden_states.dtype), index_exp, hidden_states, gen_hidden_states
            )

        else:   # list
            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
            for i in range(len(medias)):
                media_i = medias[i]
                t_i = timesteps[i:i+1] if isinstance(timesteps, torch.Tensor) else timesteps[i]

                t_i_emb = self.time_embed(t_i)      # (n_i, n_embd)

                if isinstance(media_i, torch.Tensor):
                    media_i_seq, *_ = self.patch_embed(media_i, t_i_emb)  # (n_i, num_patches, n_embd)

                elif isinstance(media_i, list):
                    media_i_seq_list = []
                    for j in range(len(media_i)):
                        media_ij = media_i[j].unsqueeze(0)
                        assert media_ij.ndim in [4, 5], \
                            f"image_ij should have size of (1, C, H, W) or (1, C, D, H, W), got {list(media_ij.size())}"
                        media_ij_seq, *_ = self.patch_embed(media_ij, t_i_emb[j:j + 1])  # (1, num_patches, n_embd)
                        media_i_seq_list.append(media_ij_seq)
                    media_i_seq = torch.cat(media_i_seq_list, dim=1)    # (1, Σj num_patches_j, n_embd)

                else:
                    raise TypeError(f"image_i should be a 4-D or 5-D tensor or a list, got {type(media_i)}")

                media_i_index = index[i:i + 1].masked_select(media_mask[i:i + 1].bool()).reshape(1, -1)  # (1, img_seqlen)
                n_embd = media_i_seq.shape[-1]
                media_i_index_exp = media_i_index.unsqueeze(-1).repeat(1, 1, n_embd)
                media_i_seq_flat = media_i_seq.reshape(1, -1, n_embd)
                assert media_i_seq_flat.shape[1] == media_i_index_exp.shape[1], \
                    f"media_i_seq_flat ({list(media_i_seq_flat.size())}) has inconsistent shape with media_i_index_exp ({list(media_i_index_exp.size())})"
                hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                    media_i_seq_flat.to(hidden_states.dtype), media_i_index_exp, hidden_states, gen_hidden_states,
                    slice_idx=i,
                )

        if gen_hidden_states is not None:
            return hidden_states, gen_hidden_states

        return hidden_states

    def _forward_vision_encoder(self, images, **image_kwargs):
        with self.vit_context():
            image_embeds = self.vit(images, **image_kwargs)

        if isinstance(image_embeds, tuple):
            image_embeds, deepstack_image_embeds = image_embeds
        else:
            deepstack_image_embeds = None
            image_embeds = image_embeds.last_hidden_state
        return image_embeds, deepstack_image_embeds

    @staticmethod
    def _accumulate_deepstack_embeds(all_embeds, new_embeds):
        if new_embeds is None:
            return all_embeds
        if all_embeds is None:
            all_embeds = [[] for _ in range(len(new_embeds))]
        for layer_idx, layer_embeds in enumerate(new_embeds):
            all_embeds[layer_idx].append(layer_embeds)
        return all_embeds

    def instantiate_vit_image_tokens(
            self,
            hidden_states: torch.Tensor,
            images: torch.Tensor | list[torch.Tensor],
            image_masks: torch.Tensor,
            image_kwargs: dict[str, torch.Tensor],
    ):
        if isinstance(hidden_states, tuple):
            hidden_states, gen_hidden_states = hidden_states
        else:
            gen_hidden_states = None

        bsz, seqlen, _ = hidden_states.shape
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        if isinstance(images, torch.Tensor):
            assert images.ndim in [3, 4, 5], f"images should be a 3-D, 4-D, or 5-D tensor, got {images.ndim}-D tensor."
            if images.ndim in [4, 5]:
                bsz, n = images.shape[:2]
                images = images.view(bsz * n, *images.shape[2:])
                image_kwargs = image_kwargs if image_kwargs is not None else {}
                for k, v in image_kwargs.items():
                    image_kwargs[k] = v.reshape(bsz * n, *v.shape[2:])
            else:
                n = 1
            image_embeds, deepstack_image_embeds = self._forward_vision_encoder(images, **image_kwargs)

            image_seqlen, n_embd = image_embeds.size(1), image_embeds.size(-1)

            image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
            index = image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd)
            src = image_embeds.reshape(bsz, n * image_seqlen, n_embd)
            assert src.shape[1] == index.shape[1], \
                f"src ({list(src.size())}) has inconsistent shape with index ({list(index.size())})"
            hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                src.to(hidden_states.dtype), index, hidden_states, gen_hidden_states
            )

        elif isinstance(images, list):
            all_deepstack_embeds = None
            for i, (image, image_mask) in enumerate(zip(images, image_masks)):
                start_index = 0
                image_scatter_index = index[i].masked_select(image_mask.bool()).reshape(1, -1)
                for j, singel_image in enumerate(image):

                    cur_kwargs = {k: v[i][j:j+1] for k, v in image_kwargs.items()} if image_kwargs is not None else {}
                    if isinstance(singel_image, list):
                        image_embed_list = []
                        for _single_image in singel_image:
                            image_embed, deepstack_image_embeds = self._forward_vision_encoder(_single_image, **cur_kwargs)
                            image_embed_list.append(image_embed)
                            all_deepstack_embeds = self._accumulate_deepstack_embeds(all_deepstack_embeds, deepstack_image_embeds)
                        image_embed = torch.cat(image_embed_list, dim=1)
                        if image_embed.ndim == 3:
                            n, image_seqlen, n_embd = image_embed.shape
                            image_embed = image_embed.reshape(n * image_seqlen, n_embd)
                        else:
                            n_embd = image_embed.shape[-1]
                    else:
                        image_embed, deepstack_image_embeds = self._forward_vision_encoder(singel_image, **cur_kwargs)
                        all_deepstack_embeds = self._accumulate_deepstack_embeds(all_deepstack_embeds, deepstack_image_embeds)
                        if image_embed.ndim == 3:
                            n, image_seqlen, n_embd = image_embed.shape
                            image_embed = image_embed.reshape(n * image_seqlen, n_embd)
                        else:
                            n_embd = image_embed.shape[-1]

                    image_scatter_index_j = image_scatter_index[:, start_index:start_index + image_embed.shape[0]]
                    image_scatter_index_j = image_scatter_index_j.unsqueeze(-1).repeat(1, 1, n_embd)
                    image_embed = image_embed.reshape(1, -1, n_embd)
                    start_index += image_embed.shape[1]

                    assert image_scatter_index_j.shape[1] == image_embed.shape[1], \
                        f"image_scatter_index_j ({list(image_scatter_index_j.size())}) has inconsistent shape with image_embed ({list(image_embed.size())})"
                    hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                        image_embed.to(hidden_states.dtype), image_scatter_index_j, hidden_states, gen_hidden_states,
                        slice_idx=i,
                    )

            if all_deepstack_embeds is not None:
                deepstack_image_embeds = []
                for layer_embeds_list in all_deepstack_embeds:
                    valid_embeds = [e for e in layer_embeds_list if e is not None]
                    if len(valid_embeds) == 0:
                        continue
                    flattened_embeds = []
                    for e in valid_embeds:
                        if isinstance(e, list):
                            flattened_embeds.extend([item for item in e if isinstance(item, torch.Tensor)])
                        elif isinstance(e, torch.Tensor):
                            flattened_embeds.append(e)
                    if len(flattened_embeds) == 1:
                        deepstack_image_embeds.append(flattened_embeds[0])
                    elif len(flattened_embeds) > 1:
                        deepstack_image_embeds.append(torch.cat(flattened_embeds, dim=0))
            else:
                deepstack_image_embeds = None
        else:
            raise ValueError(f"und_images should be Tensor or List, but got {type(images)}")

        if gen_hidden_states is not None:
            return (hidden_states, gen_hidden_states), deepstack_image_embeds

        return hidden_states, deepstack_image_embeds

    def instantiate_continuous_tokens(
            self,
            hidden_states: torch.Tensor,
            emb_layer: nn.Module,
            scatter_src: Optional[BatchRaggedTensor] = None,
            scatter_index: Optional[BatchRaggedTensor] = None,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states, gen_hidden_states = hidden_states
        else:
            gen_hidden_states = None

        bsz, seqlen, _ = hidden_states.shape

        if isinstance(scatter_src, list):
            for i, scatter_src_i in enumerate(scatter_src):
                src = emb_layer(scatter_src_i)  # (n, n_embd)
                n_embd = src.shape[-1]
                index = scatter_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd)
                src = src.reshape(1, -1, n_embd)

                assert index.shape[1] == src.shape[1], \
                    f"index ({list(index.size())}) has inconsistent shape with src ({list(src.size())})"
                hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                    src.to(hidden_states.dtype), index, hidden_states, gen_hidden_states,
                    slice_idx=i,
                )

        else:
            src = emb_layer(scatter_src.reshape(-1))    # (bsz * n, n_embd)
            n_embd = src.shape[-1]
            index = scatter_index.unsqueeze(-1).repeat(1, 1, n_embd)
            src = src.reshape(bsz, -1, n_embd)

            assert index.shape[1] == src.shape[1], \
                f"index ({list(index.size())}) has inconsistent shape with src ({list(src.size())})"
            hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                src.to(hidden_states.dtype), index, hidden_states, gen_hidden_states
            )

        if gen_hidden_states is not None:
            return hidden_states, gen_hidden_states

        return hidden_states

    def get_image_tokens_hw(self, images: BatchRaggedMedia):
        assert isinstance(images, (torch.Tensor, list)), f"images should be BatchRaggedMedia, got {type(images)}"
        if isinstance(images, torch.Tensor):
            token_h = images.shape[-2]
            token_w = images.shape[-1]
        else:
            token_h, token_w = [], []
            for image_i in images:
                assert isinstance(image_i, (torch.Tensor, list)), \
                    f"image_i should be a tensor or a list of tensors, got {type(image_i)}"
                if isinstance(image_i, torch.Tensor):
                    token_h.append(image_i.shape[-2])
                    token_w.append(image_i.shape[-1])
                else:
                    token_h.append([])
                    token_w.append([])
                    for j in range(len(image_i)):
                        token_h[-1].append(image_i[j].shape[-2])
                        token_w[-1].append(image_i[j].shape[-1])
        return token_h, token_w

    def ragged_final_layer(self, hidden_states, image_mask, timesteps, token_h, token_w, first_step=None, batch_image_sizes=None):
        n_embd = hidden_states.size(-1)

        if batch_image_sizes is not None:
            # For batch multi-resolution inference
            # bsz可能包含cond/uncond大小，而batch_image_sizes只包含实际的image大小
            bsz = hidden_states.size(0)
            actual_bsz = len(batch_image_sizes)
            pred = []
            for i in range(bsz):
                h_lat, w_lat = batch_image_sizes[i % actual_bsz]
                th_i = h_lat
                tw_i = w_lat
                n_tokens_i = th_i * tw_i
                assert n_tokens_i == image_mask[i].sum().item(), \
                    f"n_tokens_i ({n_tokens_i}) has inconsistent shape with image_mask[i].sum().item() ({image_mask[i].sum().item()})"

                if first_step is False:
                    # 非首步：hidden_states = [timestep_emb, img_emb(1D padded)]，因为有效 token 在前，直接按数量截取
                    image_output_i = hidden_states[i:i+1, 1:1+n_tokens_i, :]
                else:
                    # 首步：从完整序列中按 image_mask 布尔索引提取 image token
                    image_output_i = hidden_states[i, image_mask[i].bool(), :].unsqueeze(0)

                t_emb_i = self.time_embed_2(timesteps[i:i+1])
                pred_i = self.final_layer(image_output_i, t_emb_i, th_i, tw_i)
                pred.append(pred_i)
            return pred

        if isinstance(timesteps, torch.Tensor):
            # When timesteps is a tensor, images must be a 4-D tensor (B, C, H, W), which means only one target image
            t_emb = self.time_embed_2(timesteps)
            if first_step is False:
                # only for gen_image non-first-step inference
                image_output = hidden_states[:, 1:, :]
            else:   # first_step is True or None
                image_output = hidden_states.masked_select(
                    image_mask.unsqueeze(-1).bool()).reshape(-1, token_h * token_w, n_embd)
            pred = self.final_layer(image_output, t_emb, token_h, token_w)
        else:
            # When timesteps is a list, images must be a list of 4-D tensors or a list of list of 3-D tensors, and token_h and token_w must be a list of int or a list of list of int.
            # In this case, each line of the image_mask may contain different number of Trues, leading
            # the `reshape(batch_size, ...)` is not possible.
            sections = image_mask.sum(1).tolist()
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()).reshape(-1, n_embd).split(sections)
            pred = []
            for image_output_i, t_i, token_h_i, token_w_i in zip(image_output, timesteps, token_h, token_w):
                t_emb_i = self.time_embed_2(t_i)
                if isinstance(token_h_i, int):
                    # corresponds to image_output as a list of 4-D tensors, image_output_i as a 4-D tensor
                    image_output_i = image_output_i.reshape(-1, token_h_i * token_w_i, n_embd)
                    pred_i = self.final_layer(image_output_i, t_emb_i, token_h_i, token_w_i)
                    pred.append(pred_i)
                else:
                    # corresponds to image_output as a list of list of 3-D tensors, image_output_i as a list of 3-D tensors
                    subsections = [token_h_ij * token_w_ij for token_h_ij, token_w_ij in zip(token_h_i, token_w_i)]
                    assert sum(subsections) == image_output_i.shape[0], \
                        f"sum(subsections) ({sum(subsections)}) has inconsistent shape with image_output_i.shape[0] ({image_output_i.shape[0]})"
                    image_output_i = image_output_i.split(subsections)
                    pred_i = []
                    for j, image_output_ij in enumerate(image_output_i):
                        pred_ij = self.final_layer(image_output_ij[None], t_emb_i[j:j+1], token_h_i[j], token_w_i[j])
                        pred_i.append(pred_ij)
                    pred.append(pred_i)
        return pred

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        if isinstance(visual_embeds, list):
            if len(visual_embeds) == 0:
                raise ValueError("visual_embeds is an empty list")
            elif len(visual_embeds) == 1:
                visual_embeds = visual_embeds[0]
                if not isinstance(visual_embeds, torch.Tensor):
                    raise ValueError(f"visual_embeds list contains non-tensor element: {type(visual_embeds)}")
            else:
                if all(isinstance(e, torch.Tensor) for e in visual_embeds):
                    visual_embeds = torch.cat(visual_embeds, dim=0)
                else:
                    raise ValueError(f"visual_embeds list contains non-tensor elements")
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        bsz = hidden_states.shape[0]
        for i in range(bsz):
            batch_mask = visual_pos_masks[i]
            batch_hidden = hidden_states[i]
            num_visual_tokens = batch_mask.sum().item()
            if num_visual_tokens == 0:
                continue

            visual_hidden = batch_hidden[batch_mask]

            if visual_embeds.shape[0] >= num_visual_tokens:
                batch_visual_embeds = visual_embeds[:num_visual_tokens]
            else:
                batch_visual_embeds = visual_embeds
                if visual_embeds.shape[0] < num_visual_tokens:
                    padding = visual_embeds[-1:].repeat(num_visual_tokens - visual_embeds.shape[0], 1)
                    batch_visual_embeds = torch.cat([visual_embeds, padding], dim=0)

            visual_hidden = visual_hidden + batch_visual_embeds
            batch_hidden[batch_mask] = visual_hidden
            hidden_states[i] = batch_hidden

        return hidden_states

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,  # bsz x seqlen
            attention_mask: Optional[torch.Tensor] = None,  # bsz x 1 x seqlen x seqlen
            rope_image_info: Optional[list[list[tuple[slice, tuple[int, int], dict]]]] = None,
            return_dict: bool = True,
            # for gen images
            images: Optional[BatchRaggedMedia] = None,  # bsz x c x h x w, or bsz x (n_i x (c x h_ij x w_ij))
            image_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
            timesteps: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_i)
            timesteps_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            # for cond images
            cond_vae_images: Optional[BatchRaggedMedia] = None,  # bsz x c x h x w, or bsz x (m_i x (c x h_ij x w_ij))
            cond_vae_image_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
            cond_timesteps: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (m_i)
            cond_timesteps_index: Optional[BatchRaggedTensor] = None,
            cond_vit_images: Optional[BatchRaggedMedia] = None,
            cond_vit_image_mask: Optional[torch.Tensor] = None,
            cond_vit_image_kwargs: Optional[dict[str, Any]] = None,
            # only for inference
            input_pos: Optional[torch.Tensor] = None,  # bsz x seq_len-1, used for KVCache
            past_key_values: Optional[MultimodalStaticCache] = None,
            mode: Optional[str] = None,
            first_step: Optional[bool] = None,
            und_token_indices: Optional[torch.Tensor] = None,
            gen_token_indices: Optional[torch.Tensor] = None,
            sample_offsets: Optional[list[torch.Tensor]] = None,
            batch_image_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> MultimodalModelOutput | tuple:
        # Sanity check
        if input_ids is None and images is None:
            raise ValueError("Either input_ids or images should be provided.")
        if input_ids is not None:
            bsz = input_ids.size(0)
            device = input_ids.device
        else:
            bsz = images.size(0) if isinstance(images, torch.Tensor) else len(images)
            device = get_device(images)
        if self.training:
            seqlen = input_ids.size(1)
        else:
            seqlen = self._config.max_position_embeddings
        assert self._config.max_position_embeddings >= seqlen, (
            f"Cannot forward sequence of length {seqlen}, "
            f"max position embeddings is only {self._config.max_position_embeddings}, "
            f"try set --max-position-embeddings to a larger value."
        )
        cos, sin = self.cached_rope(
            seqlen, device, rope_media_info=rope_image_info, input_pos=input_pos, sample_offsets=sample_offsets if self.use_rope_sample_offsets else None,
        )
        cos = cos.to(dtype=self.dtype)
        sin = sin.to(dtype=self.dtype)

        if input_ids is not None:
            hidden_states = self.model["embed_tokens"](input_ids)     # (bsz, seqlen, n_embd)
            if self._config.use_mot and self._config_mot_gen.hidden_size != self._config.hidden_size:
                und_hidden_states = hidden_states
                gen_hidden_states = torch.zeros(
                    (hidden_states.size(0), hidden_states.size(1), self._config_mot_gen.hidden_size),
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                hidden_states = (und_hidden_states, gen_hidden_states)
        else:
            hidden_states = None    # only for non-first step inference of the image generation

        deepstack_image_embeds = None
        if images is not None:
            hidden_states = self.instantiate_vae_image_tokens(hidden_states, timesteps, images, image_mask)

        if cond_vae_images is not None:
            hidden_states = self.instantiate_vae_image_tokens(hidden_states, cond_timesteps, cond_vae_images, cond_vae_image_mask)

        if cond_vit_images is not None:
            hidden_states, deepstack_image_embeds = self.instantiate_vit_image_tokens(hidden_states, cond_vit_images, cond_vit_image_mask, cond_vit_image_kwargs)

        if timesteps_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, emb_layer=self.timestep_emb, scatter_src=timesteps, scatter_index=timesteps_index)

        if cond_timesteps_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, emb_layer=self.timestep_emb, scatter_src=cond_timesteps, scatter_index=cond_timesteps_index)

        if self._config.use_mot:
            assert und_token_indices is not None and gen_token_indices is not None, \
                "und_token_indices and gen_token_indices must be provided when using MoT"
            if isinstance(hidden_states, tuple):
                und_hidden_states, gen_hidden_states = hidden_states
            else:
                und_hidden_states = (
                    hidden_states.new_zeros(hidden_states.shape[0], 0, self._config.hidden_size)
                    if first_step is False else hidden_states
                )
                gen_hidden_states = hidden_states

            und_token_indices_ = und_token_indices.unsqueeze(-1).expand(-1, -1, und_hidden_states.shape[-1])
            gen_token_indices_ = gen_token_indices.unsqueeze(-1).expand(-1, -1, gen_hidden_states.shape[-1])

            und_hidden_states = und_hidden_states.gather(dim=1, index=und_token_indices_)
            gen_hidden_states = gen_hidden_states.gather(dim=1, index=gen_token_indices_)
            hidden_states = (und_hidden_states, gen_hidden_states)

        if self._config.use_modality_routing:
            _hs_for_seqlen = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
            actual_seqlen = _hs_for_seqlen.size(1)
            token_modalities = torch.zeros(bsz, actual_seqlen, dtype=torch.long, device=device)

            if input_ids is not None:
                if cond_vit_image_mask is not None:
                    token_modalities[cond_vit_image_mask.bool()] = 1
                if image_mask is not None:
                    token_modalities[image_mask.bool()] = 2
                if cond_vae_image_mask is not None:
                    token_modalities[cond_vae_image_mask.bool()] = 2
            else:
                token_modalities[:, 1:] = 2
        else:
            token_modalities = None

        for layer_idx, layer in enumerate(self.model["layers"]):    # noqa
            layer_inputs = [
                hidden_states,
                attention_mask,
                (cos, sin),
                input_pos,
                past_key_values,
                und_token_indices,
                gen_token_indices
            ]
            hidden_states = layer(*layer_inputs, token_modalities=token_modalities)
            if deepstack_image_embeds is not None and layer_idx in range(len(deepstack_image_embeds)):
                if isinstance(hidden_states, tuple):
                    und_hs, gen_hs = hidden_states
                    und_hs = self._deepstack_process(
                        und_hs,
                        cond_vit_image_mask,
                        deepstack_image_embeds[layer_idx],
                    )
                    hidden_states = (und_hs, gen_hs)
                else:
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        cond_vit_image_mask,
                        deepstack_image_embeds[layer_idx],
                    )

        if isinstance(hidden_states, tuple):
            und_hidden_states_, gen_hidden_states_ = hidden_states
            bsz = und_hidden_states_.shape[0]
            und_seqlen = und_hidden_states_.shape[1]
            gen_seqlen = gen_hidden_states_.shape[1]

            und_hidden_states = torch.zeros(
                (bsz, und_seqlen+gen_seqlen, und_hidden_states_.shape[-1]),
                device=und_hidden_states_.device, dtype=und_hidden_states_.dtype
            )
            und_hidden_states.scatter_(dim=1, index=und_token_indices_.to(und_hidden_states_.device), src=und_hidden_states_)

            gen_hidden_states = torch.zeros(
                (bsz, und_seqlen+gen_seqlen, gen_hidden_states_.shape[-1]),
                device=gen_hidden_states_.device, dtype=gen_hidden_states_.dtype
            )
            gen_hidden_states.scatter_(dim=1, index=gen_token_indices_, src=gen_hidden_states_)

        else:
            und_hidden_states, gen_hidden_states = hidden_states, hidden_states

        if images is not None:
            token_h, token_w = self.get_image_tokens_hw(images)
            gen_hidden_states = gen_hidden_states.to(device=get_device(images))
            diff_pred = self.ragged_final_layer(
                gen_hidden_states, image_mask, timesteps, token_h, token_w, first_step, batch_image_sizes=batch_image_sizes)
        else:
            diff_pred = None

        if input_ids is None or mode == "gen_image":
            logits = None
        else:
            und_hidden_states = self.model["norm"](und_hidden_states)
            logits = F.linear(und_hidden_states, self.model.embed_tokens.weight)

        if not return_dict:
            return logits, past_key_values, diff_pred
        return MultimodalModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            diffusion_prediction=diff_pred,
        )

class MultimodalModel(MultimodalModelBase):
    def __init__(
            self,
            args: Namespace,
            config: MultimodalConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            initialize_weights: bool = True,
    ):
        super().__init__()
        self.__post_init__(config, dtype, device, args, initialize_weights)

# Model construction helpers
def build_model(
        args,
        logger=None,
        dtype=None,
        device=None,
        **kwargs,
) -> tuple[torch.nn.Module, Any]:
    # Support cpu, cuda, meta devices
    factor_kwargs = {"device": device, "dtype": dtype}

    if logger is None:
        from loguru import logger

    model_structure = args.model_structure
    logger.info(f"Building model {model_structure} for {args.model_name}")

    if device == 'meta':
        context = torch.device('meta')
    else:
        context = nullcontext()

    with context:
        if model_structure in {"MultimodalModel", "MultimodalHFModel"}:
            model, model_config = _build_multimodal_model(
                args, logger=logger, **kwargs, **factor_kwargs)
        else:
            raise NotImplementedError(f"Model structure {model_structure} not implemented.")

    return model, model_config


def _build_multimodal_model(args, logger=None, dtype=None, device=None, **kwargs):
    factory_kwargs = {"device": device or 'cpu', "dtype": dtype}

    model_structure = args.model_structure
    valid_keywords = {"initialize_weights"}
    valid_kwargs = {key: value for key, value in kwargs.items() if key in valid_keywords}

    if model_structure == "MultimodalModel":
        from .configuration import core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = core_model_config_from_args(args)
        model_config = MultimodalConfig.from_name(model_name, **model_config_dict)
        model = MultimodalModel(args, model_config, **factory_kwargs, **valid_kwargs)

    elif model_structure == "MultimodalHFModel":
        from .pipeline import MultimodalHFModel
        from .configuration import core_model_config_from_args

        model_name = args.model_name.split(".")[-1]
        model_config_dict = core_model_config_from_args(args)
        model_config = MultimodalConfig.from_name(model_name, **model_config_dict)
        model = MultimodalHFModel(args, model_config, **factory_kwargs, **valid_kwargs)

    else:
        raise NotImplementedError(f"Model structure {model_structure} not implemented.")

    logger.info(f"Build Model {model.__class__.__name__} finished.")

    return model, model_config
