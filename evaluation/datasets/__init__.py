DATASETS = {
    "coco30k", "coco6k", "coco3k",
    "t2i_compbench",
    "mmmu",
    "MMBench_DEV_EN", "MMBench_DEV_CN", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11",
    "mmlu_bench",
    "pope",
    "ai2d_test",
    "realworldqa",
    "arc_challenge",
    "bbh",
    "mbpp",
}


def load_dataset(dataset_name, **kwargs):
    if dataset_name == "coco30k":
        from .coco import COCODataset
        from evaluation.constants import COCO30K_PATH
        dataset = COCODataset(COCO30K_PATH)

    elif dataset_name == "coco6k":
        from .coco import COCODataset
        from evaluation.constants import COCO6K_PATH
        dataset = COCODataset(COCO6K_PATH)

    elif dataset_name == "coco3k":
        from .coco import COCODataset
        from evaluation.constants import COCO3K_PATH
        dataset = COCODataset(COCO3K_PATH)

    elif dataset_name == "t2i_compbench":
        from .t2i_compbench import T2ICompBenchDataset
        from evaluation.constants import T2I_COMPBENCH_PATH
        dataset = T2ICompBenchDataset(T2I_COMPBENCH_PATH)

    elif dataset_name in ["mmmu", "mmmu_pro"]:
        from .mmmu import MMMUDataset
        from evaluation.constants import MMMU_PATH
        dataset = MMMUDataset(dataset_name=dataset_name,
                              data_path=MMMU_PATH[dataset_name]["path"],
                              split=MMMU_PATH[dataset_name]["split"],
                              target_size=512)

    elif dataset_name in ["MMBench_DEV_EN", "MMBench_DEV_CN", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]:
        from .mmbench import MMBenchDataset
        from evaluation.constants import LMUDataRoot
        dataset = MMBenchDataset(LMUDataRoot=LMUDataRoot,
                                 dataset_name=dataset_name,
                                 target_size=512)

    elif dataset_name == "mmlu_bench":
        from .mmlu_bench import MMLUBenchDataset
        from evaluation.constants import MMLU_BENCH_DATA
        dataset = MMLUBenchDataset(MMLU_BENCH_DATA, **kwargs)

    elif dataset_name == "pope":
        from .pope import POPEDataset
        from evaluation.constants import POPE_DATA_ROOT, POPE_IMAGE_ROOT
        dataset = POPEDataset(data_root=POPE_DATA_ROOT, image_root=POPE_IMAGE_ROOT)

    elif dataset_name == "ai2d_test":
        from .ai2d import AI2DDataset
        from evaluation.constants import AI2D_PATH
        dataset = AI2DDataset(data_path=AI2D_PATH, split="test")

    elif dataset_name == "realworldqa":
        from .realworldqa import RealWorldQADataset
        from evaluation.constants import REALWORLDQA_PATH
        dataset = RealWorldQADataset(data_path=REALWORLDQA_PATH)

    elif dataset_name == "arc_challenge":
        from .arc_c import ARCCDataset
        from evaluation.constants import ARC_C_PATH
        dataset = ARCCDataset(data_path=ARC_C_PATH, **kwargs)

    elif dataset_name == "bbh":
        from .bbh import BBHDataset
        from evaluation.constants import BBH_PATH
        dataset = BBHDataset(data_path=BBH_PATH, **kwargs)

    elif dataset_name == "mbpp":
        from .mbpp import MBPPDataset
        from evaluation.constants import MBPP_PATH
        dataset = MBPPDataset(data_path=MBPP_PATH, **kwargs)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset
