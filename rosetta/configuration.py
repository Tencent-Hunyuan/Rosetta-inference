import argparse
import json
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from rosetta.utils import DataClassMixin, default


@dataclass
class TransformerConfig(DataClassMixin):
    name: str = ""

    num_layers: int = 0
    hidden_size: int = 0
    max_position_embeddings: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int | None = None
    attention_head_size: int | None = None
    ffn_hidden_size: int | None = None
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_mixed_mlp: int = 0
    moe_topk: int = 1
    capacity_factor: float = 1.0
    moe_aux_loss: bool = True
    use_modality_routing: bool = False
    num_text_experts: int = 0
    num_vit_experts: int = 0
    num_vae_experts: int = 0
    shield_step: int = 0
    _current_training_iter: int = 0
    use_mot: bool = False
    init_std: float = 0.02
    norm_eps: float = 1e-6
    rope_theta: float = 2000.0
    mrope_section: list[int] | None = None

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads

        if self.attention_head_size is None:
            self.attention_head_size = self.hidden_size // self.num_attention_heads

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def to_hf_config(self) -> dict[str, Any]:
        return self.to_dict()

MODEL_ZOO: dict[str, dict[str, Any]] = {}


def register_model_config(name, base=None, **kwargs):
    if base is not None:
        if base not in MODEL_ZOO:
            raise ValueError(f"Base model {base} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        base_config = deepcopy(MODEL_ZOO[base])
        base_config.update(deepcopy(kwargs))
        base_config["name"] = name
        MODEL_ZOO[name] = base_config
    else:
        MODEL_ZOO[name] = {"name": name, **deepcopy(kwargs)}


@dataclass
class MultimodalConfig(TransformerConfig):
    name: str = ""

    vocab_size: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    max_position_embeddings: int = 0
    tie_word_embeddings: bool = False
    num_attention_heads: int = 0
    num_kv_heads: int | None = None
    attention_head_size: int | None = None
    ffn_hidden_size: int | None = None
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_mixed_mlp: int = 0
    moe_topk: int = 1
    capacity_factor: float = 1.0
    moe_aux_loss: bool = True
    init_std: float = 0.02
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    mrope_section: list[int] | None = None

    use_vae: bool = False
    vae_type: str = "16x16x4-32c-hy-image"
    vae_latent_dim: int = 32
    vae_downsample_factor: int = 16
    vae_precision: str = "fp32"
    vae_autocast_dtype: str = "fp16"
    patch_embed_hidden_dim: int = 1024
    use_timestep_token: bool = True
    use_vit: bool = False
    vit_type: str = "qwen3vl-vit-for-0.6b"
    vit_config: dict[str, Any] = field(default_factory=dict)

    use_mot: bool = False
    num_experts_mot_gen: int = None

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads

        if self.attention_head_size is None:
            self.attention_head_size = self.hidden_size // self.num_attention_heads

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def to_hf_config(self) -> dict[str, Any]:
        hf_configs = dict(
            vocab_size=self.vocab_size,
            org_vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.ffn_hidden_size,
            moe_intermediate_size=self.moe_ffn_hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_kv_heads,
            attention_head_dim=self.attention_head_size,
            head_dim=self.attention_head_size,
            hidden_act="silu",
            max_position_embeddings=self.max_position_embeddings,
            tie_word_embeddings=self.tie_word_embeddings,
            initializer_range=self.init_std,
            rms_norm_eps=self.norm_eps,
            rope_type="interleaved_mrope",
            rope_theta=self.rope_theta,
            use_rotary_pos_emb=True,
            num_experts=self.num_experts,
            use_mixed_mlp_moe=self.moe_mixed_mlp > 0,
            num_shared_expert=self.moe_mixed_mlp,
            moe_topk=self.moe_topk,
            use_mot=self.use_mot,
            use_vae=self.use_vae,
            vae_type=self.vae_type,
            vae_latent_dim=self.vae_latent_dim,
            vae_precision=self.vae_precision,
            vae_autocast_dtype=self.vae_autocast_dtype,
            vae_downsample_factor=self.vae_downsample_factor,
            patch_embed_hidden_dim=self.patch_embed_hidden_dim,
            use_timestep_token=self.use_timestep_token,
            use_vit=self.use_vit,
            vit_type=self.vit_type,
            vit_config=self.vit_config,
        )
        return hf_configs

    @classmethod
    def from_name(cls, model_name: str, **kwargs) -> "MultimodalConfig":
        if model_name not in MODEL_ZOO:
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        model_config = deepcopy(MODEL_ZOO[model_name])
        model_config.update(deepcopy(kwargs))
        return cls(**model_config)

    @property
    def norm_class(self):
        from .modeling import RMSNorm
        return RMSNorm

    @property
    def act_class(self):
        from torch.nn import SiLU
        return SiLU

    def to_mot_gen_config(self) -> "MultimodalConfig":
        assert self.use_mot, "use_mot must be True when using mot_gen config"

        config_mot_gen = deepcopy(self)
        if self.num_experts_mot_gen:
            config_mot_gen.num_experts = self.num_experts_mot_gen

        return config_mot_gen

def core_model_config_from_args(args: Namespace) -> dict[str, Any]:
    """ Convert training args to model config dict. """
    model_config = dict(
        use_timestep_token=args.add_timestep_token,
    )
    model_keys = [
        "vocab_size", "num_layers", "hidden_size", "max_position_embeddings",
        "num_attention_heads", "ffn_hidden_size",
        "num_experts",
        "use_vae", "vae_type", "vae_latent_dim", "vae_precision", "vae_autocast_dtype",
        "patch_embed_hidden_dim",
        "use_vit", "vit_type", "vit_config",
        dict(model_key="mrope_section", config_key="rope_dim_list"),
        "use_mot",
        "num_experts_mot_gen",
    ]
    for key_mapping in model_keys:
        if isinstance(key_mapping, dict):
            model_key = key_mapping["model_key"]
            config_key = key_mapping["config_key"]
        else:
            model_key = config_key = key_mapping
        if hasattr(args, config_key) and getattr(args, config_key) is not None:
            value = getattr(args, config_key)
            if isinstance(value, dict):
                model_config[model_key] = dict(value)
            else:
                model_config[model_key] = value
    return model_config


register_model_config(
    name="qwen3-06b-base-upcycling-ours-lm",
    vocab_size=151936,
    num_layers=28,
    hidden_size=1024,
    max_position_embeddings=32768,
    tie_word_embeddings=True,
    num_attention_heads=16,
    num_kv_heads=8,
    attention_head_size=128,
    ffn_hidden_size=3072,
    moe_ffn_hidden_size=3072,
    num_experts=3,
    moe_mixed_mlp=1,
    moe_topk=2,
    init_std=0.02,
    norm_eps=1e-6,
    rope_theta=1000000,
    mrope_section=[24, 20, 20],
)


register_model_config(
    name="qwen3-06b-base-upcycling-moe-lm-deepseek",
    vocab_size=151936,
    num_layers=28,
    hidden_size=1024,
    max_position_embeddings=32768,
    tie_word_embeddings=True,
    num_attention_heads=16,
    num_kv_heads=8,
    attention_head_size=128,
    ffn_hidden_size=3072,
    moe_ffn_hidden_size=3072,
    num_experts=12,
    moe_mixed_mlp=1,
    moe_topk=2,
    init_std=0.02,
    norm_eps=1e-6,
    rope_theta=1000000,
    mrope_section=[24, 20, 20],
)


register_model_config(
    name="qwen3-06b-upcycling-moe-mm-deepseek",
    base="qwen3-06b-base-upcycling-moe-lm-deepseek",
    vocab_size=157420,
    use_vae=True,
    vae_type="16x16-128c-flux2",
    vae_latent_dim=128,
    vae_precision="fp32",
    vae_autocast_dtype="fp32",
    use_vit=True,
    vit_type="qwen3vl-vit-for-0.6b",
    vit_config=dict(spatial_merge_size=2),
)


register_model_config(
    name="qwen3-06b-upcycling-ours-mm",
    base="qwen3-06b-base-upcycling-ours-lm",
    vocab_size=157420,
    num_experts=12,
    use_modality_routing=True,
    num_text_experts=3,
    num_vit_experts=3,
    num_vae_experts=6,
    use_vae=True,
    vae_type="16x16-128c-flux2",
    vae_latent_dim=128,
    vae_precision="fp32",
    vae_autocast_dtype="fp32",
    use_vit=True,
    vit_type="qwen3vl-vit-for-0.6b",
    vit_config=dict(spatial_merge_size=2),
)

register_model_config(
    name="qwen3-06b-base-mot-lm",
    base="qwen3-06b-base-upcycling-ours-lm",
    num_experts=7,
)

register_model_config(
    name="qwen3-06b-mot-mm",
    base="qwen3-06b-base-mot-lm",
    vocab_size=157420,
    use_vae=True,
    vae_type="16x16-128c-flux2",
    vae_latent_dim=128,
    vae_precision="fp32",
    vae_autocast_dtype="fp32",
    use_vit=True,
    vit_type="qwen3vl-vit-for-0.6b",
    vit_config=dict(spatial_merge_size=2),
)

register_model_config(
    name="qwen3-06b-mot",
    base="qwen3-06b-mot-mm",
    use_mot=True,
    num_experts_mot_gen=6,
)


def parse_argv_from_yaml(
        yaml_path: str,
        allow_frozen: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
    from omegaconf import OmegaConf  # type: ignore[reportMissingImports]
    from omegaconf.dictconfig import DictConfig  # type: ignore[reportMissingImports]
    from omegaconf.listconfig import ListConfig  # type: ignore[reportMissingImports]
    OmegaConf.register_new_resolver("add", lambda *args: sum(int(x) for x in args), replace=True)
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y), replace=True)
    OmegaConf.register_new_resolver("div", lambda x, y: int(x) // int(y), replace=True)

    config_data = OmegaConf.load(yaml_path)
    argv = []
    frozen_args = {}

    def _flatten_config_to_args(
        config_field: Union[Dict[str, Any], DictConfig], prefix: str = ""
    ):
        for key, value in config_field.items():
            full_key = f"{prefix}.{key}" if prefix != "" else key

            if full_key.startswith("__GLOBAL_VARS__"):
                continue

            if "__FROZEN__" in full_key:
                if not isinstance(value, DictConfig):
                    raise ValueError(f"__FROZEN__* node must be a DictConfig, got {type(value)}")
                for frozen_key, frozen_value in value.items():
                    if isinstance(frozen_value, (DictConfig, ListConfig)):
                        frozen_args[frozen_key] = OmegaConf.to_container(frozen_value, resolve=True)
                    else:
                        frozen_args[frozen_key] = frozen_value
                continue

            if isinstance(value, DictConfig):
                _flatten_config_to_args(value, prefix=full_key)
            elif isinstance(value, ListConfig):
                if key == "enabled":
                    for item in value:
                        argv.append(f"--{item}")
                else:
                    argv.append(f"--{key}")
                    for item in value:
                        if isinstance(item, DictConfig):
                            # Dict behind a list can not be handled by argument parser,
                            # so we serialize it as a JSON string.
                            resolved_dict = OmegaConf.to_container(item, resolve=True)
                            argv.append(json.dumps(resolved_dict))
                        else:
                            assert item is not None, f"null is not allowed in yaml config."
                            argv.append(str(item))
            elif isinstance(value, bool):
                if value:
                    argv.append(f"--{key}")
                else:
                    argv.append(f"--no-{key}")
            else:
                real_value = str(OmegaConf.select(config_data, full_key))
                assert real_value != "None", f"null is not allowed in yaml config for key: {full_key}"
                argv.append(f"--{key}")
                argv.append(real_value)

    _flatten_config_to_args(config_data)

    if allow_frozen:
        return argv, frozen_args
    return argv


PRECISIONS = ["fp32", "fp16", "bf16", "fp8"]


def add_data_core_args(parser):
    parser.add_argument("--tokenizer-name", type=str, help="Tokenizer name.")
    parser.add_argument("--tokenizer-class", type=str, default="tokenizer.Qwen3BaseTokenizerFast",
                        help="Tokenizer class to use.")

    # vae resolution
    parser.add_argument("--vae-image-token-length", type=int, help="Maximum image token length of vae latents.")
    parser.add_argument("--reso-base-size", type=int, help="Resolution group base size.")
    parser.add_argument("--reso-align", type=int, default=16, help="Resolution alignment.")
    parser.add_argument("--reso-preset", type=str, help="Resolution group preset.")

    # vit
    parser.add_argument("--vit-image-token-length", type=int, help="Image token length of vit features.")
    parser.add_argument("--min-vit-image-token-length", type=int, help="Minimum image token length of vit features.")

    # image token flags
    parser.add_argument("--add-timestep-token", action="store_true", help="Add timestep token.")
    parser.add_argument("--add-image-shape-token", action="store_true",
                        help="Add image shape token before the image token sequence.")
    # sequence
    parser.add_argument("--modality", type=str, nargs="*",
                        choices=["text", "vae_image", "vit_image"])
    parser.add_argument("--sequence-template", type=str, default="instruct",
                        choices=["pretrain", "instruct"])

    return parser


def add_model_core_args(parser):
    # precision
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision.")
    parser.add_argument("--autocast-dtype", type=str, choices=PRECISIONS)

    # reproducibility
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-benchmark", action="store_false", dest="benchmark")

    # model structure
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--max-position-embeddings", type=int)

    # mot
    parser.add_argument("--use-mot", action="store_true", help="Use MoT.")
    parser.add_argument("--und-token-type", type=str, nargs="*",
                        default=["bos", "text", "boi", "vit", "eoi", "joint_image_sep", "eoi", "eos", "pad"])
    parser.add_argument("--gen-token-type", type=str, nargs="*", default=["vae_info", "vae"])
    parser.add_argument("--num-experts-mot-gen", type=int)

    # device / loading
    parser.add_argument("--init-device", type=str, choices=["cuda", "cpu", "meta"],
                        help="Device to initialize the model on.")
    parser.add_argument("--local-rank", type=int, help="Local rank for distributed training.")

    # basic
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-structure", type=str)

    return parser


def add_extra_models_core_args(parser: argparse.ArgumentParser):
    # vae
    parser.add_argument("--use-vae", action="store_true")
    parser.add_argument("--vae-type", type=str)
    parser.add_argument("--vae-precision", type=str, default="fp32", choices=PRECISIONS)
    parser.add_argument("--vae-autocast-dtype", type=str, default="fp16", choices=PRECISIONS)
    parser.add_argument("--vae-latent-dim", type=int)

    # vit
    parser.add_argument("--use-vit", action="store_true")
    parser.add_argument("--vit-type", type=str)
    parser.add_argument("--vit-frozen", action="store_true")

    return parser


def add_denoise_core_args(parser: argparse.ArgumentParser):
    parser.add_argument("--flow-snr-type", type=str, default="lognorm", choices=["uniform", "lognorm"])
    parser.add_argument("--flow-shift", type=float, default=1.0)
    parser.add_argument("--flow-reverse", action="store_true")
    return parser


def add_evaluation_core_args(parser):
    # checkpoint
    parser.add_argument("--ckpt", type=str, help="Model checkpoint path.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")

    # testsets
    parser.add_argument("--testsets", type=str, nargs="*")
    parser.add_argument("--eval-metrics", type=str, nargs="*")
    parser.add_argument("--sample-save-base", type=str)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--eval-save-images", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--generation-config", type=str)

    # run
    parser.add_argument("--prompt", type=str, nargs="*")
    parser.add_argument("--image", type=str, nargs="*")
    parser.add_argument("--image-size", type=str, default="auto")
    parser.add_argument("--bot-task", type=str,
                        choices=["auto", "image"])

    return parser


def add_core_args(parser):
    parser.add_argument("--task-id", type=str, required=True)

    parser = add_data_core_args(parser)
    parser = add_model_core_args(parser)
    parser = add_extra_models_core_args(parser)
    parser = add_denoise_core_args(parser)
    parser = add_evaluation_core_args(parser)
    return parser


def preprocess_args(args):
    if not args.use_mot:
        args.und_token_type = []
        args.gen_token_type = []

    return args


def validate_args(args, defaults=None):
    defaults = default(defaults, {})
    for key in defaults:
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print(f"WARNING: overriding default argument {key} with {defaults[key]}", flush=True)
        else:
            setattr(args, key, defaults[key])
    args = preprocess_args(args)
    return args
