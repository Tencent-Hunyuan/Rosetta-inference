import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from rosetta.utils import DataClassMixin, PRECISION_TO_TYPE

MODEL_ZOO: dict[str, "QwenViTConfig"] = {}


def register_model_config(name, base=None, **kwargs):
    if base is not None:
        if base not in MODEL_ZOO:
            raise ValueError(f"Base model {base} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        base_config = MODEL_ZOO[base].to_dict()
        base_config.update({**kwargs, "name": name})
        MODEL_ZOO[name] = QwenViTConfig(**base_config)
    else:
        MODEL_ZOO[name] = QwenViTConfig(name=name, **kwargs)


@dataclass
class QwenViTConfig(DataClassMixin):
    name: str = ""
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    initializer_range: float = 0.02

    @classmethod
    def from_name(cls, model_name: str, **kwargs) -> "QwenViTConfig":
        if model_name not in MODEL_ZOO:
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        model_config = MODEL_ZOO[model_name].to_dict()
        model_config.update(kwargs)
        return cls(**model_config)


register_model_config(
    name="qwen3vl-vit-for-30b-a3b",
    depth=27,
    hidden_size=1152,
    intermediate_size=4304,
    num_heads=16,
    in_channels=3,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=2048,
    num_position_embeddings=2304,
    deepstack_visual_indexes=[8, 16, 24],
)

register_model_config(
    name="qwen3vl-vit-for-0.6b",
    base="qwen3vl-vit-for-30b-a3b",
    out_hidden_size=1024,
)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)


    def reset_parameters(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, patch_size, temporal_patch_size, in_channels, hidden_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_implementation="sdpa") -> None:
        super().__init__()
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self._attn_implementation = attn_implementation
        self.attention_dropout = 0.0
        self.is_causal = False
        # Create a minimal config object for transformers flash attention compatibility
        self.config = type('Config', (), {'_attn_implementation': attn_implementation})()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]

        if self._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class GELUTanh(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act_fn(input)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = GELUTanh()

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads, attn_implementation)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn
        mlp = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config: QwenViTConfig, attn_implementation="sdpa", **kwargs):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config.patch_size,
            config.temporal_patch_size,
            config.in_channels,
            config.hidden_size,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(config.hidden_size, config.num_heads, config.intermediate_size, attn_implementation) for _ in range(config.depth)
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config.hidden_size,
            config.spatial_merge_size,
            config.out_hidden_size,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config.hidden_size,
                    config.spatial_merge_size,
                    config.out_hidden_size,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # Convert tensor to Python int for torch.linspace
            h_int = h.item() if isinstance(h, torch.Tensor) else int(h)
            w_int = w.item() if isinstance(w, torch.Tensor) else int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h_int)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w_int)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def process_single_hidden_states(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)


        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states.unsqueeze(0)

        return hidden_states, deepstack_feature_lists

    def forward(self, hidden_states: torch.Tensor, grid_thw: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
            **kwargs: additional keyword arguments.
        Returns:
            `torch.Tensor`: hidden_states, (bsz, seqlen, n_embd).
            `list[torch.Tensor]`: deepstack_feature_lists, [[], [], ...]
        """

        if hidden_states.ndim == 3 and hidden_states.shape[0] > 1:
            # 如果输入的hidden_states是(batch, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = hidden_states.shape

            # 处理**kwargs将其也拆分成和hidden_states_i = hidden_states[i]
            all_kwargs = []
            for i in range(batch_size):
                kwargs_i = {k: v[i:i+1] for k, v in kwargs.items()}
                all_kwargs.append(kwargs_i)

            all_hidden_states = []
            all_deepstack_feature_lists = []
            for i in range(batch_size):
                hidden_states_i = hidden_states[i]
                grid_thw_i = grid_thw[i:i+1]
                kwargs_i = all_kwargs[i]
                hidden_states_i, deepstack_feature_lists_i = self.process_single_hidden_states(hidden_states_i, grid_thw_i, **kwargs_i)
                all_hidden_states.append(hidden_states_i)
                all_deepstack_feature_lists.append(deepstack_feature_lists_i)

            # 合并所有hidden_states和deepstack_feature_lists
            hidden_states = torch.cat(all_hidden_states, dim=0)
            deepstack_feature_lists = all_deepstack_feature_lists
        else:
            if isinstance(grid_thw, list) and len(grid_thw) == 1:
                grid_thw = grid_thw[0]
            elif isinstance(grid_thw, list) and len(grid_thw) > 1:
                assert False, "process_single_hidden_states should be called with a single grid_thw"
            hidden_states, deepstack_feature_lists = self.process_single_hidden_states(hidden_states, grid_thw, **kwargs)

        if not len(self.deepstack_visual_indexes) > 0:
            deepstack_feature_lists = None

        return hidden_states, deepstack_feature_lists

ASSETS_BASE = os.getenv("ASSETS_BASE", "./public_assets").rstrip("/")
VISION_ENCODER_BASE = os.getenv("VISION_ENCODER_BASE", f"{ASSETS_BASE}/vision_encoder").rstrip("/")
VISION_ENCODER_META_INFO = {
    "qwen3vl-vit-for-0.6b": {
        "path": f"{VISION_ENCODER_BASE}/Qwen3-VL-30B-A3B-Instruct",
        "downsample_factor": [32, 32],
        "spatial_merge_size": 2,
        "patch_dim": 1536,
        "dummy_number": 1,
    },
}


def load_vision_model(
        vision_model_type=None,
        vision_model_precision=None,
        device=None,
        logger=None,
        require_grad=False,
        eval_mode=True,
        no_load_pretrained=False,
        vision_model_params=None,
        config=None,
):
    if logger is None:
        from loguru import logger

    if config is not None:
        vision_model_type = config["vision_model_type"]
        vision_model_precision = config.get("vision_model_precision", vision_model_precision)
        require_grad = not config.get("vision_model_freeze", not require_grad)
        eval_mode = config.get("vision_model_freeze", eval_mode)
        no_load_pretrained = config.get("no_load_pretrained_vision_model", False)
        vision_model_params = config.get("vision_model_params", vision_model_params)

    if vision_model_params is None:
        vision_model_params = {}

    if vision_model_type.startswith("qwen3vl-vit"):
        model_config = QwenViTConfig.from_name(vision_model_type)
        vision_model = Qwen3VLVisionModel(model_config, **vision_model_params)
        if not no_load_pretrained and vision_model_type in VISION_ENCODER_META_INFO:
            meta = VISION_ENCODER_META_INFO[vision_model_type]
            if "path" in meta:
                from loguru import logger as _logger
                from safetensors.torch import load_file as st_load_file
                import json
                hf_path = Path(meta["path"])
                index_file = hf_path / "model.safetensors.index.json"
                if index_file.exists():
                    with open(index_file) as _f:
                        shard_map = json.load(_f)["weight_map"]
                    # Only load keys that belong to visual encoder (prefix "visual.")
                    vit_shards = set(v for k, v in shard_map.items() if k.startswith("visual."))
                    vit_state = {}
                    for shard in sorted(vit_shards):
                        sd = st_load_file(hf_path / shard)
                        vit_state.update({k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")})
                    missing, unexpected = vision_model.load_state_dict(vit_state, strict=False)
                    _logger.info(
                        f"[qwen3vl-vit] Loaded backbone from {hf_path.name}: "
                        f"{len(vit_state)-len(missing)} keys loaded, "
                        f"{len(missing)} missing (e.g. shape-mismatch merger.linear_fc2 → random init), "
                        f"{len(unexpected)} unexpected."
                    )
    else:
        raise NotImplementedError(f"vision_model_type {vision_model_type} not implemented")

    if vision_model_precision is not None:
        logger.warning(f"You are transforming the Vision Encoder to {vision_model_precision}. Please make sure this is what you want.")
        if isinstance(vision_model_precision, str):
            vision_model_precision = PRECISION_TO_TYPE[vision_model_precision]
        vision_model = vision_model.to(dtype=vision_model_precision)

    if device is not None:
        vision_model = vision_model.to(device=device)

    if not require_grad:
        vision_model.requires_grad_(False)

    if eval_mode:
        vision_model.eval()

    return vision_model


def load_vision_model_processor(vision_model_type, **kwargs):
    if vision_model_type.startswith("qwen3vl-vit"):
        from transformers import AutoImageProcessor
        vision_model_meta_info = VISION_ENCODER_META_INFO[vision_model_type]
        vision_model_path = Path(vision_model_meta_info["path"])
        processor = AutoImageProcessor.from_pretrained(vision_model_path, **kwargs)
    else:
        raise NotImplementedError(f"vision_model_type {vision_model_type} not implemented")

    return processor


load_vit = load_vision_model
load_vit_processor = load_vision_model_processor
