import os


ASSETS_BASE = os.getenv("ASSETS_BASE", "./public_assets").rstrip("/")
EVAL_BASE = os.getenv("EVAL_BASE", f"{ASSETS_BASE}/evaluation").rstrip("/")

TESTSET_TEMPLATE = os.environ.get("TESTSET_TEMPLATE", "evaluation/testsets/test/{}.csv")

MMLU_BENCH_DATA = f"{EVAL_BASE}/MMLU/data"
LMUDataRoot = f"{EVAL_BASE}/MMBench"
MMMU_PATH = {
    "mmmu": {
        "path": f"{EVAL_BASE}/MMMU",
        "split": "validation",
    },
}

COCO30K_PATH = os.getenv("COCO30K_PATH", f"{EVAL_BASE}/COCO/coco30k.csv")
COCO6K_PATH = os.getenv("COCO6K_PATH", f"{EVAL_BASE}/COCO/coco6k.csv")
COCO3K_PATH = os.getenv("COCO3K_PATH", f"{EVAL_BASE}/COCO/coco3k.csv")
COCO_VAL2014 = f"{EVAL_BASE}/dataset/COCO/val2014"

POPE_DATA_ROOT = os.getenv("POPE_DATA_ROOT", f"{EVAL_BASE}/POPE")
POPE_IMAGE_ROOT = COCO_VAL2014
AI2D_PATH = os.getenv("AI2D_PATH", f"{EVAL_BASE}/AI2D")
REALWORLDQA_PATH = os.getenv("REALWORLDQA_PATH", f"{EVAL_BASE}/RealWorldQA")
ARC_C_PATH = os.getenv("ARC_C_PATH", f"{EVAL_BASE}/ARC_C")
BBH_PATH = os.getenv("BBH_PATH", f"{EVAL_BASE}/BBH")
MBPP_PATH = os.getenv("MBPP_PATH", f"{EVAL_BASE}/MBPP")

T2I_COMPBENCH_PATH = [
    "evaluation/testsets/t2i_compbench/color_val.csv",
    "evaluation/testsets/t2i_compbench/shape_val.csv",
    "evaluation/testsets/t2i_compbench/texture_val.csv",
]
VQA_MODEL_PATH = os.getenv("VQA_MODEL_PATH", f"{EVAL_BASE}/T2I-CompBench")

FID_INCEPTION_PATH = os.getenv("FID_INCEPTION_PATH", f"{EVAL_BASE}/fid/inception_v3_google-0cc3c7bd.pth")
FID_TARGET_PATH = {
    "coco30k": os.getenv("FID_TARGET_COCO30K", f"{EVAL_BASE}/fid/coco30k.npz"),
    "coco6k": os.getenv("FID_TARGET_COCO6K", f"{EVAL_BASE}/fid/coco6k.npz"),
    "coco3k": os.getenv("FID_TARGET_COCO3K", f"{EVAL_BASE}/fid/coco3k.npz"),
}
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", f"{EVAL_BASE}/clip")
HPSV2_MODEL_PATH = os.getenv("HPSV2_MODEL_PATH", f"{EVAL_BASE}/hpsv2")
