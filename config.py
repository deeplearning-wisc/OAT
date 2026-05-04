"""Configuration for OAT failure attribution."""

from __future__ import annotations

from pathlib import Path

OAT_DIR = Path(__file__).resolve().parent
DATASET_DIR = OAT_DIR / "dataset"

WHO_AND_WHEN_DATA_DIR = DATASET_DIR / "who_and_when" / "Who&When"
MCP_ATLAS_DATA_DIR = DATASET_DIR / "MCP-atlas" / "Qwen3.5-27B"

DATASETS = ("mcp_atlas", "who_and_when")
DEFAULT_DATASET = "mcp_atlas"
MAX_TOOL_CONTENT_LENGTH = 4096

OUTPUT_DIR = OAT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
HIDDEN_STATES_ROOT = OUTPUT_DIR / "hidden_states"
HIDDEN_STATES_MEAN_ROOT = OUTPUT_DIR / "hidden_states_mean"

MODEL_NAME = "Qwen/Qwen3.5-27B"
MAX_SEQ_LENGTH = 262144
HIDDEN_STATE_AGGREGATION = "mean"
DEFAULT_LAYER = -1
MODEL_STEPS_ONLY = True

# Latent projection is intentionally fixed to PCA.
LATENT_DIM = 64
ENCODER_TYPE = "pca"

# OAT uses a TorchCDE-based neural CDE core.
OAT_HIDDEN = 64
OAT_DEPTH = 3
OAT_INTERPOLATION = "cubic"
OAT_SOLVER = "euler"
OAT_ADJOINT = False
OAT_CONTROL_GATE = True
OAT_CONTROL_GATE_HIDDEN = 12
OAT_CONTROL_GATE_DEPTH = 4
OAT_CONTROL_GATE_EPS = 1e-6
OAT_USE_QUESTION_H0 = True
OAT_H0_NORM_REG = 0.0
OAT_H1_BRIDGE_HIDDEN = OAT_HIDDEN
OAT_H1_BRIDGE_DEPTH = OAT_DEPTH
OAT_H1_BRIDGE_SOLVER = "euler"

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 20
WEIGHT_DECAY = 1e-5
N_SEED_RUNS = 5
RANDOM_SEED = 42

DETECTION_TOP_K = 3
CONFORMAL_ALPHA = 0.2
CONFORMAL_MIN_DETECTIONS = 1
REPORT_METRICS = (
    "topk_detection_precision",
    "topk_detection_recall",
    "topk_detection_f1",
    "topk_detection_hit_rate",
    "conformal_detection_precision",
    "conformal_detection_recall",
    "conformal_detection_f1",
    "conformal_detection_hit_rate",
    "auroc",
    "auprc",
)
OOD_ALIGN = "coral"
OOD_ALIGN_EPS = 1e-4

LLM_JUDGE_BACKEND = "vllm"
LLM_JUDGE_MODEL_NAME = MODEL_NAME
LLM_JUDGE_MAX_NEW_TOKENS = 2048
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_MODEL = "gpt-5"
AZURE_OPENAI_API_KEY_ENV = "AZURE_OPENAI_API_KEY"

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        DTYPE = torch.float32
except ImportError:  # pragma: no cover
    DEVICE = "cpu"
    DTYPE = None


def get_dataset_dir(dataset: str) -> Path:
    if dataset == "mcp_atlas":
        return MCP_ATLAS_DATA_DIR
    if dataset == "who_and_when":
        return WHO_AND_WHEN_DATA_DIR
    raise ValueError(f"Unsupported dataset: {dataset}. Choose from {DATASETS}.")


def get_hidden_states_dir(
    aggregation: str = HIDDEN_STATE_AGGREGATION,
    dataset: str = DEFAULT_DATASET,
) -> Path:
    root = HIDDEN_STATES_MEAN_ROOT if aggregation == "mean" else HIDDEN_STATES_ROOT
    return root / dataset
