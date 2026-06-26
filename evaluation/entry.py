import argparse
import importlib
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, List, Any

import torch.cuda
import torch.distributed as dist
from loguru import logger

from rosetta.configuration import parse_argv_from_yaml
from rosetta.utils import set_args, set_logger
from rosetta.utils import ParallelState
from rosetta.configuration import add_core_args, validate_args
from rosetta.utils import print_args
from rosetta.utils import set_reproducibility


def get_sampler(name):
    assert '.' in name, (
        f"Invalid sampler name: {name}. A valid sampler name should be in the form of "
        f"<module_name>.<sampler_cls>."
    )
    module_name, trainer_cls = name.rsplit('.', 1)
    module_spec = importlib.import_module(f"evaluation.{module_name}")
    return getattr(module_spec, trainer_cls)


def create_sampler_for_pipeline(
    config_path: str,
    ckpt_path: str,
    extra_args: Optional[List[str]] = None,
    sampler_name: str = "multimodal_sampler.MultimodalSampler",
    framework: str = "hf",
    device: int = 0,
    logger_instance: Any = None,
):
    """Create a sampler instance using the same parsing and env setup as entry.run().

    Used by Gradio/pipeline so that config parsing (parse_argv_from_yaml + add_core_args)
    is shared with the torchrun/run_sample.sh path. Returns a sampler ready for inference.
    """
    config_path = str(Path(config_path).resolve())
    ckpt_path = str(Path(ckpt_path).resolve())
    argv = [
        "--config-path", config_path,
        "--sampler", sampler_name,
        "--framework", framework,
        "--ckpt", ckpt_path,
        "--task-id", "pipeline",  # required by add_core_args; use fixed id for Gradio/pipeline
    ]
    if extra_args:
        argv.extend(extra_args)

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + argv
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config-path", type=str, required=True)
        parser.add_argument("--sampler", type=str, required=True)
        parser.add_argument("--framework", type=str, choices=["hf", "fsdp"], default="hf")
        known_args, remaining_argv = parser.parse_known_args(argv)

        config_argv, frozen_args = parse_argv_from_yaml(known_args.config_path, allow_frozen=True)
        combined_argv = config_argv + remaining_argv
        sys.argv = [saved_argv[0]] + combined_argv

        parser = argparse.ArgumentParser(description="Multimodal Pure Torch Sampler Launcher")
        parser = add_core_args(parser)
        args, _ = parser.parse_known_args()
        args = validate_args(args, frozen_args)

        if args.model_structure == "MultimodalModel":
            args.model_structure = "MultimodalHFModel"

        if known_args.framework == "hf":
            rank = 0
            world_size = 1
            torch.cuda.set_device(device)
            ParallelState(dp_rank=0, dp_size=1)
        else:
            dist.init_process_group("nccl", timeout=timedelta(seconds=3600 * 24 * 365))
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            torch.cuda.set_device(rank % torch.cuda.device_count())
            if known_args.framework == "fsdp":
                ParallelState.from_pure_torch()
            else:
                raise NotImplementedError(f"Framework {known_args.framework} not supported yet.")

        set_args(args)
        if logger_instance is not None:
            set_logger(logger_instance)
        elif rank == 0:
            set_logger(logger)
        else:
            from rosetta.utils import empty_logger
            set_logger(empty_logger())
        set_reproducibility(getattr(args, "reproduce", False), getattr(args, "seed", 1234), getattr(args, "benchmark", True))

        sampler_cls = get_sampler(sampler_name)
        return sampler_cls(known_args, rank, world_size)
    finally:
        sys.argv = saved_argv


def run():
    # Parse config yaml
    original_argv = sys.argv.copy()[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path", type=str, required=True, help="Config yaml file path")
    parser.add_argument("--sampler", type=str, required=True, help="Name of the sampler to use for inference.")
    parser.add_argument("--framework", type=str, choices=["hf", "fsdp"], default="hf")
    known_args, remaining_argv = parser.parse_known_args(original_argv)

    # config_argv will be handled by argparse later, frozen_args will be passed to args directly
    config_argv, frozen_args = parse_argv_from_yaml(known_args.config_path, allow_frozen=True)
    original_argv = config_argv + remaining_argv
    sys.argv = [sys.argv[0]] + original_argv

    # parse args
    parser = argparse.ArgumentParser(description="Multimodal Pure Torch Sampler Launcher")
    parser = add_core_args(parser)
    args, _ = parser.parse_known_args()
    args = validate_args(args, frozen_args)

    if args.model_structure == "MultimodalModel":
        args.model_structure = "MultimodalHFModel"

    if known_args.framework == "hf":
        rank = 0
        world_size = 1
        torch.cuda.set_device(0)
        ParallelState(dp_rank=0, dp_size=1)

    else:
        # Initialize distributed process group
        dist.init_process_group("nccl", timeout=timedelta(seconds=3600 * 24 * 365))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())

        if rank == 0:
            print_args("arguments", args)

    if known_args.framework == "fsdp":
        ParallelState.from_pure_torch()
    # Setup global vars
    set_args(args)
    if rank == 0:
        set_logger(logger)
    else:
        from rosetta.utils import empty_logger
        set_logger(empty_logger())

    # Control reproducibility
    set_reproducibility(args.reproduce, args.seed, args.benchmark)

    # Invoke the specified sampler
    sampler = get_sampler(known_args.sampler)(known_args, rank, world_size)

    if args.prompt:
        sampler.run()

    elif args.testsets or args.eval_metrics:
        sampler.run_testsets()

    else:
        raise ValueError("One of the --prompt, --testsets, --eval-metrics must be provided for sampling.")

    sampler.exit()


if __name__ == "__main__":
    run()
