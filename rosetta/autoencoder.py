import math
import copy
import os
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.init as init
from PIL import Image
from einops import rearrange
from torch import Tensor, nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from rosetta.utils import PRECISION_TO_TYPE


def normal_weight_reset_parameters(std=0.02, bias_type="default"):
    def _wrap_fn(_self):
        init.normal_(_self.weight, std=std)
        if hasattr(_self, "bias") and _self.bias is not None:
            if bias_type == "default":
                fan_in, _ = init._calculate_fan_in_and_fan_out(_self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(_self.bias, -bound, bound)
            elif bias_type == "zeros":
                init.zeros_(_self.bias)
            else:
                raise ValueError(f"Unsupported bias_init_type: {bias_type}")
    return _wrap_fn


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size,
        act_layer=nn.GELU,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
        config=None,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.init_std = config.init_std if config is not None and hasattr(config, 'init_std') else 0.02
        out_size = hidden_size if out_size is None else out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=self.init_std)
        nn.init.normal_(self.mlp[2].weight, std=self.init_std)

    def prepare_reset_parameters(self):
        self.mlp[0].reset_parameters = normal_weight_reset_parameters(std=self.init_std).__get__(self.mlp[0])
        self.mlp[2].reset_parameters = normal_weight_reset_parameters(std=self.init_std).__get__(self.mlp[2])

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


def zero_reset_parameters(_self):
    init.zeros_(_self.weight)
    if hasattr(_self, 'bias') and _self.bias is not None:
        init.zeros_(_self.bias)


def conv_nd(dims, *args, **kwargs):
    if dims != 2:
        raise ValueError(f"unsupported dimensions: {dims}")
    return nn.Conv2d(*args, **kwargs)


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, **kwargs):
    return nn.GroupNorm(32, channels, **kwargs)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        dims=2,
        device=None,
        dtype=None,
        kernel_size=3,
        padding=1,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        if dims != 2:
            raise ValueError("Only 2D image projector blocks are supported.")
        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, kernel_size, padding=padding, **factory_kwargs),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels, **factory_kwargs),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, **factory_kwargs)),
        )
        self.skip_connection = (
            nn.Identity()
            if self.out_channels == self.in_channels
            else conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)
        )

    def reset_parameters(self):
        self.out_layers[3].reset_parameters = zero_reset_parameters.__get__(self.out_layers[3])

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1. + scale) + shift
        h = out_rest(h)
        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    def __init__(
        self, in_channels, emb_channels, hidden_channels, out_channels, dropout=0.0,
        device=None, dtype=None, kernel_size=3, padding=1,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.model = nn.ModuleList([
            conv_nd(2, in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding, **factory_kwargs),
            ResBlock(
                in_channels=hidden_channels, emb_channels=emb_channels, out_channels=out_channels,
                dropout=dropout, dims=2, kernel_size=kernel_size, padding=padding, **factory_kwargs,
            ),
        ])

    def forward(self, x, t):
        assert x.ndim == 4, f"image latents should be 4D [B, C, H, W], got {list(x.shape)}"
        for module in self.model:
            x = module(x, t) if isinstance(module, ResBlock) else module(x)
        _, _, *token_sizes = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, *token_sizes


class UNetUp(nn.Module):
    def __init__(
        self, in_channels, emb_channels, hidden_channels, out_channels, dropout=0.0,
        device=None, dtype=None, out_norm=False, kernel_size=3, padding=1,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.model = nn.ModuleList([
            ResBlock(
                in_channels=in_channels, emb_channels=emb_channels, out_channels=hidden_channels,
                dropout=dropout, dims=2, kernel_size=kernel_size, padding=padding, **factory_kwargs,
            )
        ])
        if out_norm:
            self.model.append(nn.Sequential(
                normalization(hidden_channels, **factory_kwargs),
                nn.SiLU(),
                conv_nd(2, in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, **factory_kwargs),
            ))
        else:
            self.model.append(conv_nd(
                2, in_channels=hidden_channels, out_channels=out_channels,
                kernel_size=kernel_size, padding=padding, **factory_kwargs,
            ))

    def forward(self, x, t, *token_sizes):
        token_h, token_w = token_sizes
        x = rearrange(x, 'b (h w) c -> b c h w', h=token_h, w=token_w)
        for module in self.model:
            x = module(x, t) if isinstance(module, ResBlock) else module(x)
        return x


def project_in_layer(config, **kwargs):
    return UNetDown(
        emb_channels=config.hidden_size,
        in_channels=config.vae_latent_dim,
        hidden_channels=config.patch_embed_hidden_dim,
        out_channels=config.hidden_size,
        **kwargs,
    )


def project_out_layer(config, **kwargs):
    return UNetUp(
        emb_channels=config.hidden_size,
        in_channels=config.hidden_size,
        hidden_channels=config.patch_embed_hidden_dim,
        out_channels=config.vae_latent_dim,
        out_norm=True,
        **kwargs,
    )


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            reverse: bool = True,
            start_sigma: float = 1.0,
            end_sigma: float = 0.0,
    ):
        sigmas = torch.linspace(start_sigma, end_sigma, num_train_timesteps + 1)
        if not reverse:
            sigmas = sigmas.flip(0)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        sigmas = torch.linspace(self.config.start_sigma, self.config.end_sigma, num_inference_steps + 1)
        if self.config.shift != 1.:
            sigmas = (self.config.shift * sigmas) / (1 + (self.config.shift - 1) * sigmas)
        if not self.config.reverse:
            sigmas = 1 - sigmas
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self._step_index = None

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        self._step_index = indices[1 if len(indices) > 1 else 0].item()

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
    ) -> Tuple[torch.Tensor]:
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError("Pass a scheduler timestep value, not an integer step index.")
        if self._step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        return (sample + model_output * (sigma_next - sigma),)


class ClassifierFreeGuidance:
    def __call__(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor, guidance_scale: float, step: int):
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


class MultimodalPipeline(DiffusionPipeline):
    def __init__(
            self,
            model,
            scheduler: SchedulerMixin,
            vae,
            vae_autocast_dtype: Optional[torch.dtype] = None,
            progress_bar_config: Dict[str, Any] = None,
    ):
        super().__init__()
        self._progress_bar_config = getattr(self, "_progress_bar_config", {})
        self._progress_bar_config.update(progress_bar_config or {})
        self.vae_autocast_dtype = vae_autocast_dtype
        self.register_modules(model=model, scheduler=scheduler, vae=vae)
        self.latent_scale_factor = self.model.config.vae_downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)
        self.cfg_operator = ClassifierFreeGuidance()

    def prepare_latents(self, batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
        if isinstance(image_size, list):
            assert len(image_size) == batch_size
            return self._prepare_latents_variable_res(
                batch_size, latent_channel, image_size, dtype, device, generator, latents,
            )
        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(image_size)
        elif isinstance(self.latent_scale_factor, (tuple, list)):
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(f"Unsupported latent_scale_factor: {self.latent_scale_factor}")
        latents_shape = (
            batch_size, latent_channel,
            *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generator list length must match batch size.")
        latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype) if latents is None else latents.to(device)
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _prepare_latents_variable_res(
        self, batch_size, latent_channel, image_sizes, dtype, device, generator, latents=None,
    ):
        ndim = len(image_sizes[0])
        if self.latent_scale_factor is None:
            scale_factors = (1,) * ndim
        elif isinstance(self.latent_scale_factor, int):
            scale_factors = (self.latent_scale_factor,) * ndim
        else:
            scale_factors = tuple(self.latent_scale_factor)
        latents_list = []
        for i, img_size in enumerate(image_sizes):
            shape = (1, latent_channel, *tuple(int(s) // f for s, f in zip(img_size, scale_factors)))
            gen = generator[i] if isinstance(generator, list) else generator
            lat = randn_tensor(shape, generator=gen, device=device, dtype=dtype) if latents is None else latents[i:i + 1].to(device)
            if hasattr(self.per_sample_schedulers[i], "init_noise_sigma"):
                lat = lat * self.per_sample_schedulers[i].init_noise_sigma
            latents_list.append(lat)
        return latents_list

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0 and self._cfg_factor > 1

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        image_size: tuple[int, int] | list[tuple[int, int]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        model_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> list[Image.Image]:
        self._guidance_scale = guidance_scale
        cfg_factor = kwargs.pop('cfg_factor', None)
        self._cfg_factor = cfg_factor if cfg_factor is not None else (2 if guidance_scale > 1.0 else 1)
        device = self.model.device
        is_variable_res = isinstance(image_size, list)
        if is_variable_res:
            self.per_sample_schedulers = [copy.deepcopy(self.scheduler) for _ in range(batch_size)]
        latents = self.prepare_latents(
            batch_size=batch_size,
            latent_channel=self.model.config.vae_latent_dim,
            image_size=image_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if is_variable_res:
            per_sample_timesteps = []
            for b_idx in range(batch_size):
                self.per_sample_schedulers[b_idx].set_timesteps(num_inference_steps, device=device)
                per_sample_timesteps.append(self.per_sample_schedulers[b_idx].timesteps)
            timesteps = per_sample_timesteps[0]
            scheduler_order = self.per_sample_schedulers[0].order
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            scheduler_order = self.scheduler.order

        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(device)
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler_order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if is_variable_res:
                    latent_model_input = [lat.clone() for lat in latents] * self._cfg_factor
                    per_t = [per_sample_timesteps[j][i] for j in range(batch_size)]
                    t_expand = torch.stack(per_t * self._cfg_factor)
                else:
                    latent_model_input = torch.cat([latents] * self._cfg_factor)
                    t_expand = t.repeat(latent_model_input.shape[0])
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids, images=latent_model_input, timesteps=t_expand, **model_kwargs,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model_output = self.model(**model_inputs, first_step=(i == 0))
                    pred = model_output["diffusion_prediction"]
                if is_variable_res:
                    pred = [p.to(dtype=torch.float32) for p in pred]
                    if self.do_classifier_free_guidance:
                        half = len(pred) // 2
                        pred = [
                            self.cfg_operator(pc, pu, self.guidance_scale, step=i)
                            for pc, pu in zip(pred[:half], pred[half:])
                        ]
                    for j in range(batch_size):
                        latents[j] = self.per_sample_schedulers[j].step(
                            pred[j], per_sample_timesteps[j][i], latents[j],
                        )[0]
                else:
                    pred = pred.to(dtype=torch.float32)
                    if pred.ndim == 5 and pred.size(2) == 1 and latents.ndim == 4:
                        pred = pred.squeeze(2)
                    if self.do_classifier_free_guidance:
                        pred_cond, pred_uncond = pred.chunk(2)
                        pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)
                    latents = self.scheduler.step(pred, t, latents)[0]
                if i != len(timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(model_output, model_kwargs)
                    input_ids = None
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler_order == 0):
                    progress_bar.update()
        return self.decode_latents(latents, is_variable_res, output_type, generator)

    def decode_latents(self, latents, is_variable_res, output_type, generator):
        if is_variable_res:
            images = []
            for lat in latents:
                images.extend(self.decode_latents(lat, False, output_type, generator))
            return images
        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor
        with torch.autocast(
                device_type="cuda", dtype=self.vae_autocast_dtype,
                enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32,
        ):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
        return self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=[True] * image.shape[0],
        )

@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32

    @classmethod
    def from_json(cls, json_path: str | Path) -> "AutoEncoderParams":
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor, return_dict: bool = True, generator=None) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @classmethod
    def from_pretrained(cls, path: str):
        path = Path(path)

        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            params = AutoEncoderParams.from_json(config_path)
            config = FrozenDict(config_dict)
        else:
            print(f"Warning: config.json not found at {config_path}, using default params")
            params = AutoEncoderParams()
            config = FrozenDict(params.to_dict())

        model = cls(params=params)
        model.config = config
        
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
        return model
ASSETS_BASE = os.getenv("ASSETS_BASE", "./public_assets").rstrip("/")
VAE_BASE = os.getenv("VAE_BASE", f"{ASSETS_BASE}/image_encoder").rstrip("/")
VAE_META_INFO = {
    "16x16-128c-flux2": {
        "path": f"{VAE_BASE}/flux2-vae",
        "downsample_factor": [16, 16],
        "trans_type": "-11",
    },
}


def load_vae(
    vae_type,
    vae_precision=None,
    device=None,
    logger=None,
    args=None,
):
    if logger is None:
        from loguru import logger

    vae_meta_info = VAE_META_INFO[vae_type]
    vae_path = Path(vae_meta_info["path"])

    config_file = vae_path / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    if "_class_name" in config:
        classname = config.pop("_class_name")
    else:
        raise ValueError(f"Cannot find the _class_name in {config_file}")
    logger.info(f"Load VAE with class {classname} from {config_file}")
    logger.info(f"Load vae_type: {vae_type} from path: {vae_path}")

    if classname != "AutoencoderKLFlux2" and "flux" not in vae_type:
        raise NotImplementedError(f"VAE class {classname} is not supported.")
    vae = AutoEncoder.from_pretrained(vae_path)

    vae._downsample_factor = vae_meta_info["downsample_factor"]
    if not hasattr(vae, 'downsample_factor'):
        vae.downsample_factor = vae_meta_info["downsample_factor"][0]
    vae._trans_type = vae_meta_info["trans_type"]

    if args is not None:
        vae.autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]

    if vae_precision is not None:
        logger.warning(f"You are transforming VAE to {vae_precision} precision! Please make sure this is what you want.")
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    if device is not None:
        vae = vae.to(device=device)
    vae.requires_grad_(False)
    vae.eval()

    return vae
