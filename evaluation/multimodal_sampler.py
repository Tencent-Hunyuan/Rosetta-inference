import inspect
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, List, Iterator, TypeVar

import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers.generation.utils import GenerateOutput

from evaluation.entry import create_sampler_for_pipeline
from rosetta.utils import get_args, get_logger, get_parallel_state
from rosetta.utils import ParallelState
from evaluation.sampling_dataset import MessageListDataset
from evaluation.metrics import load_metric
from rosetta.modeling import build_model
from rosetta.utils import default, readable_time
from rosetta.utils import safe_save_file, save_to_csv, save_to_json
from rosetta.utils import Timer


T_co = TypeVar('T_co', covariant=True)


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, Image.Image):
        return f"<PIL.Image {obj.size}>"
    return obj


def _serialize_output_data(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().tolist()
    if isinstance(data, pd.DataFrame):
        return None
    if isinstance(data, list):
        return [
            json.dumps(_make_json_serializable(d), ensure_ascii=False)
            if isinstance(d, (list, dict)) else d
            for d in data
        ]
    raise ValueError(f"Unsupported data type: {type(data)}")


def _postprocess_images(outputs, save_images: bool):
    if not save_images:
        outputs.images = None
    if outputs.images is not None and isinstance(outputs.images, torch.Tensor):
        images_np = outputs.images.cpu().permute(0, 2, 3, 1).float().numpy()
        images_np = (images_np * 255).round().astype("uint8")
        outputs.images = [Image.fromarray(img) for img in images_np]


def _save_generation_outputs(outputs, save_base: Path, summary_file_name: str):
    if outputs.is_empty():
        return
    response = [dict(role="assistant", content=[]) for _ in outputs.batch["index"]]
    if outputs.texts is not None:
        for i, text in enumerate(outputs.texts):
            response[i]["content"].append(dict(type="text", text=text))
    if outputs.images is not None:
        save_image_base = save_base / "images"
        save_image_base.mkdir(parents=True, exist_ok=True)
        outputs.batch["gen_images"] = []
        for i, image in enumerate(outputs.images):
            image_path = save_image_base / f"{outputs.batch['index'][i]}_0.png"
            image.save(image_path)
            outputs.batch["gen_images"].append(str(image_path))
            response[i]["content"].append(dict(type="image", image_path=str(image_path)))
    outputs.batch["response"] = response
    serialized = {k: _serialize_output_data(v) for k, v in outputs.batch.items()}
    serialized = {k: v for k, v in serialized.items() if v is not None}
    save_to_csv(pd.DataFrame(serialized), save_base / summary_file_name, append=True)


def _load_pretrained_model(model, dtype, ckpt_path):
    import re
    from accelerate import dispatch_model
    from transformers.modeling_utils import (
        PreTrainedModel,
        _get_device_map, _get_resolved_checkpoint_files,    # noqa
    )
    from transformers.quantizers.quantizers_utils import get_module_from_name
    from rosetta.utils import is_package_version

    keep_in_fp32_regex = re.compile(r"\.gate\.wg")
    get_device_map_kwargs = dict(keep_in_fp32_regex=keep_in_fp32_regex)
    if is_package_version("transformers", ">=", "4.56"):
        get_device_map_kwargs["dtype"] = dtype
    else:
        get_device_map_kwargs["torch_dtype"] = dtype
    model.device_map = _get_device_map(
        model,
        device_map="auto" if torch.cuda.device_count() > 1 else "sequential",
        max_memory=None,
        hf_quantizer=None,
        **get_device_map_kwargs,
    )
    print(f"Device map: \n{json.dumps(model.device_map, indent=4)}", flush=True)

    kwargs = {}
    if is_package_version("transformers", ">=", "4.53"):
        kwargs["is_remote_code"] = False
    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=ckpt_path,
        subfolder='',
        variant=None,
        gguf_file=None,
        from_tf=False,
        from_flax=False,
        use_safetensors=None,   # noqa
        cache_dir=None,     # noqa
        force_download=False,
        proxies=None,
        local_files_only=False,
        token=False,
        user_agent={'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': False},
        revision='main',
        commit_hash=None,
        **kwargs,
    )
    (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,
        error_msgs,
    ) = PreTrainedModel._load_pretrained_model(     # noqa
        model,
        None,
        checkpoint_files,
        ckpt_path,
        sharded_metadata=sharded_metadata,
        device_map=model.device_map,
        dtype=dtype,
        keep_in_fp32_regex=keep_in_fp32_regex,
        key_mapping=model.get_key_mapping(),
        weights_only=True,
    )
    if len(missing_keys) > 0:
        for key in missing_keys:
            module, _ = get_module_from_name(model, key)
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    model.tie_weights()

    print(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}", flush=True)
    if model.device_map is not None:
        dispatch_model(model, device_map=model.device_map)


def build_tkwrapper():
    from rosetta.tokenizer import load_tokenizer

    args = get_args()
    tkwrapper = load_tokenizer(args.tokenizer_name, args.tokenizer_class)
    return tkwrapper


def build_vae(dp_rank=None):
    from rosetta.autoencoder import load_vae

    args = get_args()
    logger = get_logger()
    device = torch.device("cuda", args.local_rank)

    logger.info("Building VAE...")
    vae = load_vae(
        args.vae_type,
        args.vae_precision,
        device=device,
        logger=logger,
        args=args,
    )
    if dp_rank is None:
        dp_rank = get_parallel_state().dp_rank

    generator = torch.Generator(device).manual_seed(args.seed + dp_rank)
    vae.generator = generator

    return vae


class DistributedSamplerFix(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, add_extra_samples: bool | str = False,
                 ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            raise RuntimeError('Using `dist.get_world_size()` is dangerous.')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            raise RuntimeError('Using `dist.get_rank()` is dangerous.')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.add_extra_samples = add_extra_samples
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
            self.total_size = self.num_samples * self.num_replicas
        elif self.add_extra_samples:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.num_replicas
        else:
            total_size = len(self.dataset)  # type: ignore[arg-type]
            self.num_samples = total_size // self.num_replicas + int(
                rank < total_size % self.num_replicas
            )
            self.total_size = total_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        dataset_length = len(self.dataset)  # type: ignore[arg-type]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(dataset_length, generator=g).tolist()
        else:
            indices = list(range(dataset_length))

        if not self.drop_last:
            if not self.add_extra_samples:
                pass
            elif self.add_extra_samples == "extend":
                padding_size = self.total_size - len(indices)
                indices += [index + dataset_length for index in range(padding_size)]
            else:
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

class MultimodalSampler(object):
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: int = 0,
        logger=None,
        extra_args: Optional[List[str]] = None,
    ):
        """Load sampler for pipeline/inference without distributed.

        Reuses evaluation.entry parsing (parse_argv_from_yaml + add_core_args) so that
        Gradio and run_sample.sh share the same config/arg path.
        extra_args: optional CLI args (e.g. ["--framework", "hf", "--sequence-template", "pretrain"]).
        """
        ckpt_path = Path(ckpt_path)
        config_path = Path(config_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}.")
        if logger is None:
            from loguru import logger as _logger
            logger = _logger
        return create_sampler_for_pipeline(
            config_path=str(config_path),
            ckpt_path=str(ckpt_path),
            extra_args=extra_args,
            sampler_name="multimodal_sampler.MultimodalSampler",
            framework="hf",
            device=device,
            logger_instance=logger,
        )

    def __init__(self, init_args, rank, world_size):
        super().__init__()
        self.init_args = init_args
        self.device = rank % torch.cuda.device_count()
        self.rank = rank
        self.world_size = world_size
        self.parallel_state: ParallelState = get_parallel_state()

        self.logger = get_logger()
        self.setup_models(init_args.framework)
        self.setup_extra_models()
        self.pure_text_tasks = ["auto"]

    def setup_models(self, framework):
        args = get_args()

        # Initialize model.
        # For inference we always honour --bf16 so ViT flash_attention_2 gets bf16/fp16.
        dtype = torch.bfloat16 if args.bf16 else torch.float32
        # Use args.init_device as-is (may be "meta" for FSDP training configs).
        # For the fsdp inference path we follow the standard PyTorch pattern:
        #   build on meta → to_empty(cuda) → load_state_dict
        # This avoids OOM from keeping a full CPU copy before moving to CUDA.
        self.model, self.model_config = build_model(
            args,
            dtype=dtype,
            device=args.init_device,
            initialize_weights=False,
        )
        assert args.ckpt is not None, "Checkpoint path `--ckpt` must be provided for sampling."
        if framework == "hf":
            _load_pretrained_model(self.model, dtype=dtype, ckpt_path=args.ckpt)
            self.model.requires_grad_(False)
            self.model.eval()
            self._cast_model_for_inference(dtype)

        elif framework == "fsdp":
            from pathlib import Path as _Path
            _ckpt = _Path(args.ckpt)
            if (_ckpt / "model.safetensors.index.json").exists():
                self._load_hf_checkpoint(str(args.ckpt))
            elif (_ckpt / "model.safetensors").exists():
                self._load_single_safetensors_checkpoint(str(args.ckpt))
            elif (_ckpt / ".metadata").exists():
                self._load_dcp_checkpoint(str(args.ckpt))
            else:
                raise ValueError(f"Unsupported checkpoint format for FSDP inference: {args.ckpt}")

            cuda_device = f"cuda:{self.device}"
            try:
                first_param = next(self.model.parameters())
                if first_param.device.type == 'meta':
                    self.model = self.model.to_empty(device=cuda_device)
                else:
                    self.model = self.model.cuda()
            except StopIteration:
                self.model = self.model.cuda()
            self.model.requires_grad_(False)
            self.model.eval()
            self.model_engine = self.model
            self._cast_model_for_inference(dtype)

        else:
            raise NotImplementedError(f"Framework {framework} not supported.")

        self.model.load_generation_config(default(args.generation_config, args.ckpt))

    def _cast_model_for_inference(self, dtype: torch.dtype):
        if dtype == torch.float32:
            return
        self.model = self.model.to(dtype=dtype)
        self._keep_moe_router_fp32()
        if hasattr(self.model, "_dtype"):
            self.model._dtype = dtype
        if hasattr(self.model, "vit_precision"):
            self.model.vit_precision = dtype

    def _keep_moe_router_fp32(self):
        for module in self.model.modules():
            for attr in ("wg", "wg_text", "wg_vit", "wg_vae"):
                linear = getattr(module, attr, None)
                if isinstance(linear, torch.nn.Linear):
                    linear.to(dtype=torch.float32)

    def _load_dcp_checkpoint(self, load_dir: str, strict: bool = True):
        import torch
        import torch.distributed.checkpoint as DCP
        from torch.distributed.checkpoint import FileSystemReader

        if self.rank == 0:
            print(f"[DCP] Loading checkpoint from {load_dir} ...", flush=True)

        cuda_device = f"cuda:{self.device}"
        try:
            first_param = next(self.model.parameters())
            if first_param.device.type == 'meta':
                if self.rank == 0:
                    print(f"[DCP] Materialising meta tensors to {cuda_device} ...", flush=True)
                self.model = self.model.to_empty(device=cuda_device)
        except StopIteration:
            pass

        model_state_dict = dict(self.model.state_dict())
        wrapped_state_dict = {"model": model_state_dict}

        try:
            from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
            planner = DefaultLoadPlanner(flatten_sharded_tensors=True)
        except (ImportError, TypeError):
            planner = None

        DCP.load(
            state_dict=wrapped_state_dict,
            storage_reader=FileSystemReader(load_dir),
            planner=planner,
        )

        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=strict)
        if self.rank == 0:
            if missing:
                print(f"[DCP] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}", flush=True)
            if unexpected:
                print(f"[DCP] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}", flush=True)
            print(f"[DCP] Done.", flush=True)

    def _load_hf_checkpoint(self, load_dir: str, strict: bool = True):
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        load_dir = Path(load_dir)
        index_file = load_dir / "model.safetensors.index.json"

        if self.rank == 0:
            print(f"[HF] Loading checkpoint from {load_dir} ...", flush=True)

        # Materialise meta tensors to CUDA before loading weights.
        cuda_device = f"cuda:{self.device}"
        try:
            first_param = next(self.model.parameters())
            if first_param.device.type == "meta":
                if self.rank == 0:
                    print(f"[HF] Materialising meta tensors to {cuda_device} ...", flush=True)
                self.model = self.model.to_empty(device=cuda_device)
        except StopIteration:
            pass

        # All ranks load from the shared filesystem in parallel.
        with index_file.open() as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shard_files = sorted(set(weight_map.values()))

        full_state_dict = {}
        for shard_fname in shard_files:
            if self.rank == 0:
                print(f"[HF] Loading shard {shard_fname} ...", flush=True)
            shard = load_file(str(load_dir / shard_fname), device=cuda_device)
            full_state_dict.update(shard)

        missing, unexpected = self.model.load_state_dict(full_state_dict, strict=strict)
        if self.rank == 0:
            if missing:
                print(f"[HF] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}", flush=True)
            if unexpected:
                print(f"[HF] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}", flush=True)
            print(f"[HF] Done.", flush=True)

    def _load_single_safetensors_checkpoint(self, load_dir: str, strict: bool = True):
        from pathlib import Path
        from safetensors.torch import load_file

        load_dir = Path(load_dir)
        checkpoint_file = load_dir / "model.safetensors"

        if self.rank == 0:
            print(f"[safetensors] Loading checkpoint from {checkpoint_file} ...", flush=True)

        cuda_device = f"cuda:{self.device}"
        try:
            first_param = next(self.model.parameters())
            if first_param.device.type == "meta":
                if self.rank == 0:
                    print(f"[safetensors] Materialising meta tensors to {cuda_device} ...", flush=True)
                self.model = self.model.to_empty(device=cuda_device)
        except StopIteration:
            pass

        state_dict = load_file(str(checkpoint_file), device=cuda_device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("timestep_emb.",)
        disallowed_missing = [
            key for key in missing
            if not key.startswith(allowed_missing_prefixes)
        ]
        if strict and (disallowed_missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict for single-file safetensors checkpoint:\n"
                f"\tMissing key(s): {disallowed_missing}\n"
                f"\tUnexpected key(s): {unexpected}"
            )
        if self.rank == 0:
            if missing:
                print(
                    f"[safetensors] Missing keys ({len(missing)}): "
                    f"{missing[:3]}{'...' if len(missing)>3 else ''}",
                    flush=True,
                )
            if unexpected:
                print(
                    f"[safetensors] Unexpected keys ({len(unexpected)}): "
                    f"{unexpected[:3]}{'...' if len(unexpected)>3 else ''}",
                    flush=True,
                )
            print(f"[safetensors] Done.", flush=True)

    def setup_extra_models(self):
        args = get_args()
        # Initialize vae, tokenizer
        self.model.tokenizer = build_tkwrapper()
        if args.use_vae:
            self.model.model_dict["vae"] = build_vae(dp_rank=self.rank)

    def run(self):
        args = get_args()
        dp_rank = self.parallel_state.dp_rank
        assert args.prompt is not None, "'prompt' must be provided for generation."
        bot_task = self.model.generation_config.bot_task

        if bot_task in self.pure_text_tasks:
            # Pure text generation
            inputs = self.model.prepare_model_inputs(prompt=args.prompt, image=args.image, bot_task=bot_task)
            self.model.generate(**inputs, verbose=2)

        else:
            # Hybrid text-image generation
            generation_outputs = self.model.generate_image(
                prompt=args.prompt[self.rank % len(args.prompt)],
                image=args.image,
                seed=args.seed,
                image_size=args.image_size,
                bot_task=bot_task,
                verbose=2,
            )
            texts, images = generation_outputs.texts, generation_outputs.images
            if texts is not None:
                print(f"[rank {dp_rank}] Generated Text: {texts}")
            for i, image in enumerate(images):
                image.save(f"image_{dp_rank}_{i}.png")
                print(f"Image saved to image_{dp_rank}_{i}.png")

    @staticmethod
    def postprocess_results(save_base):
        results_dir = save_base / "results"
        if not results_dir.exists():
            return

        if torch.distributed.get_rank() == 0:
            all_dfs = []
            for csv_file in sorted(results_dir.glob("results_*.csv"), key=lambda x: int(x.stem.split("_")[-1])):
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
            merged_df = pd.concat(all_dfs, ignore_index=True)
            if 'index' in merged_df.columns:
                merged_df = merged_df.sort_values(by='index')
            merged_save_path = save_base / "results/all_results.csv"
            merged_df.to_csv(merged_save_path, index=False)

    def runtime_media_generation_config(self, run_task_kwargs):
        args = get_args()
        runtime_config = dict(
            bot_task=run_task_kwargs.get("bot_task", self.model.generation_config.bot_task),
            image_size=run_task_kwargs.get("image_size", args.image_size),
        )
        return runtime_config

    @staticmethod
    def per_batch_runtime_config(batch, runtime_config):
        if "height" in batch and "width" in batch:
            image_size = [(int(h), int(w)) for h, w in zip(batch["height"], batch["width"])]
        else:
            image_size = runtime_config["image_size"]
        per_batch_runtime_config = {**runtime_config, "image_size": image_size}
        return per_batch_runtime_config

    def build_dataset(self, testset):
        args = get_args()
        return MessageListDataset(testset, args.sample_save_base, tokenizer=self.model.tokenizer)

    def run_testsets(self):
        args = get_args()
        assert args.testsets is not None or args.eval_metrics is not None, \
            "'testsets' or 'eval_metrics' must be provided for run_testsets."
        assert args.sample_save_base is not None, "'sample_save_base' must be provided for run_testsets."

        run_tasks = [("sample", testset) for testset in (args.testsets or [])] + [
            ("eval_metric", testset) for testset in (args.eval_metrics or [])
        ]

        timer = Timer(enabled=True)
        timer.start("global")

        for task_idx, (run_task_type, testset) in enumerate(run_tasks):
            generate_task_specific_kwargs = dict()

            dataset = self.build_dataset(testset)
            run_task_kwargs = dataset.task_kwargs
            sampler = DistributedSamplerFix(dataset, num_replicas=self.parallel_state.dp_size,
                                            rank=self.parallel_state.dp_rank, shuffle=False, drop_last=False,
                                            add_extra_samples="extend")
            dataloader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False, sampler=sampler,
                                    drop_last=False, collate_fn=getattr(dataset, "collate_fn", None))
            save_base = dataset.save_dir
            self.logger.info(f"=" * 80)
            self.logger.info(f"Running task {testset}({task_idx + 1} / {len(run_tasks)})")
            self.logger.info(f"Save directory: {save_base}")
            self.logger.info(f"=" * 80)

            metric_instances = []
            if run_task_type == "eval_metric":
                kwargs = {}
                LOGIT_BASED_TESTSETS = {"mmlu_bench", "arc_challenge", "mmmlu"}
                if dataset.testset in LOGIT_BASED_TESTSETS:
                    kwargs = {"tokenizer": self.model.tokenizer}
                    generate_task_specific_kwargs["return_dict_in_generate"] = True
                    generate_task_specific_kwargs["output_logits"] = True
                assert "metric" in run_task_kwargs, f"'metric' must be specified in testset for eval_metric task."
                metric_types = run_task_kwargs["metric"].split("+")

                for metric_type in metric_types:
                    metric = load_metric(f"{metric_type}@{dataset.testset}", **kwargs)
                    metric.load_model(self.logger)
                    metric_instances.append((metric, metric_type))

            bot_task = run_task_kwargs.get("bot_task", self.model.generation_config.bot_task)
            generate_task_specific_kwargs["bot_task"] = bot_task

            verbose = args.verbose if args.sample_batch_size == 1 else min(1, args.verbose)
            if bot_task in self.pure_text_tasks:
                max_new_tokens = int(run_task_kwargs.get("max_new_tokens", self.model.generation_config.max_new_tokens))
                generate_task_specific_kwargs["max_new_tokens"] = max_new_tokens
                model_type = run_task_kwargs.get("model_type", None)
                skip_special_tokens = False if model_type else True

                runtime_config = {**self.model.generation_config.to_dict(), **generate_task_specific_kwargs}
                timer.start(f"Task {task_idx}")
                timer.start(f"Batch")
                for batch_idx, batch in enumerate(dataloader):
                    batch: dict[str, Any]
                    self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")

                    inputs = self.model.prepare_model_inputs(
                        message_list=batch[dataset.name_mapper("message_list")],
                        mode="gen_text",
                        **generate_task_specific_kwargs,
                    )
                    outputs = self.model.generate(
                        **inputs, decode_text=True, verbose=verbose, skip_special_tokens=skip_special_tokens
                    )
                    outputs = outputs.postprocess_outputs(batch)

                    if run_task_type == "eval_metric" and outputs is not None:
                        start_time = time.time()
                        for metric, _ in metric_instances:
                            proc_inputs = {k: v for k, v in batch.items()}
                            proc_inputs["answers"] = outputs.texts
                            metric.process(**proc_inputs, model_type=model_type)
                        if isinstance(outputs.texts, GenerateOutput):
                            outputs.texts = outputs.texts.sequences
                        gen_time = time.time() - start_time
                        self.logger.info(f"Metric process time: {gen_time}")

                    _save_generation_outputs(
                        outputs,
                        save_base=save_base,
                        summary_file_name=f"results/results_{self.parallel_state.dp_rank}.csv",
                    )

                    timer.stop(f"Batch")
                    self.logger.info(f"Task {testset}({task_idx + 1}/{len(run_tasks)})"
                                     f"[{batch_idx + 1} / {len(dataloader)}] "
                                     f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)}")
                    timer.start(f"Batch")
                timer.stop("Batch")
                timer.stop(f"Task {task_idx}")
                self.logger.info(f"Save directory: {save_base}")

            else:
                runtime_config = self.runtime_media_generation_config(run_task_kwargs)
                _image_output = dict(sample="pil", eval_metric="pt")
                output_type = _image_output[run_task_type]

                timer.start(f"Task {task_idx}")
                timer.start(f"Batch")
                for batch_idx, batch in enumerate(dataloader):
                    batch: dict[str, Any]
                    self.logger.info(f"Generating batch {batch_idx + 1} / {len(dataloader)} ...")
                    batch_config = self.per_batch_runtime_config(batch, runtime_config)
                    outputs = self.model.generate_image(
                        message_list=batch[dataset.name_mapper("message_list")],
                        seed=batch["seed"],
                        **batch_config,
                        image_output_type=output_type,
                        verbose=verbose,
                    )
                    outputs = outputs.postprocess_outputs(batch)

                    if run_task_type == "eval_metric" and not outputs.is_empty():
                        start_time = time.time()
                        for metric, _ in metric_instances:
                            proc_inputs = {k: v for k, v in outputs.batch.items()}
                            proc_inputs["images"] = outputs.images.float()
                            proc_inputs.setdefault("ids", outputs.batch["index"])
                            metric.process(**proc_inputs)
                        gen_time = time.time() - start_time
                        _postprocess_images(outputs, args.eval_save_images)
                        self.logger.info(f"Metric process time: {gen_time}")

                    _save_generation_outputs(
                        outputs,
                        save_base=save_base,
                        summary_file_name=f"results/results_{self.parallel_state.dp_rank}.csv",
                    )

                    timer.stop(f"Batch")
                    self.logger.info(f"[Task {testset}({task_idx + 1}/{len(run_tasks)})] "
                                     f"[{batch_idx + 1} / {len(dataloader)}] "
                                     f"| {readable_time(timer, 'Batch', len(dataloader) - batch_idx - 1)}")
                    timer.start(f"Batch")
                timer.stop("Batch")
                timer.stop(f"Task {task_idx}")
                self.logger.info(f"Save directory: {save_base}")

            torch.distributed.barrier()
            self.postprocess_results(save_base)

            if run_task_type == "eval_metric":
                dist.barrier()
                self.logger.info(f"All processes have finished the evaluation. Start all gathering results...")

                metric_gather_results = [metric.all_gather_results() for metric, _ in metric_instances]
                self.logger.info(f"All gather results for all metrics finished")

                if self.rank == 0:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    for (metric, metric_type), gathered_results in zip(metric_instances, metric_gather_results):
                        metric_kwargs = {}
                        accept_save_file = 'save_file' in list(inspect.signature(metric.compute_metrics).parameters.keys())
                        if accept_save_file and hasattr(metric, "save_file_template"):
                            metric_kwargs['save_file'] = save_base / f"metric_temp/{metric_type}" \
                                                         / metric.save_file_template.format(timestamp.replace('-', '_'))
                            self.logger.info(f"[{metric_type}] Temp results will be saved to {metric_kwargs['save_file']}")

                        output = metric.compute_metrics(gathered_results, **metric_kwargs)
                        if isinstance(output, tuple):
                            output = {"_": output}

                        results = []
                        for key, (value, count, *extra_outputs) in output.items():
                            suffix = "" if key == "_" else f"_{key}"
                            results.append({
                                "timestamp": timestamp,
                                "metric": f"{metric_type}{suffix}",
                                "testset": dataset.testset,
                                "value": value,
                                "count": count,
                                "runtime_config": {
                                    **self.model.generation_config.to_dict(),
                                    **runtime_config,
                                },
                            })
                            if len(extra_outputs) > 0:
                                # Metrics like VQAv2 return a dict to save some extra information
                                if 'extra_info_dict' in extra_outputs[0]:
                                    results[-1]["extra_metric_stats"] = extra_outputs[0]['extra_info_dict']

                        self.logger.info(results)

                        accumulated_results = results[:]
                        save_path = save_base / f"metric_results/{metric_type}.json"
                        if save_path.exists():
                            with open(save_path, "r") as f:
                                ori_results = json.load(f)
                            accumulated_results = ori_results + results

                        save_to = safe_save_file(save_path, accumulated_results, save_fn=save_to_json)
                        self.logger.info(f"Evaluation results saved to {save_to}")

                dist.barrier()

        timer.stop("global")
        self.logger.info(f"Total time cost: {readable_time(timer.elapsed('global'))}.")

    def exit(self):
        torch.cuda.empty_cache()

        if torch.distributed.is_initialized():
            dist.barrier()
            print(f"[rank {self.rank}] Sampling is complete. Exiting now.", flush=True)
            dist.destroy_process_group()
