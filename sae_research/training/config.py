import importlib
import itertools
from enum import Enum
from pathlib import Path

import torch as t
import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------


def resolve_class(import_path: str) -> type:
    """Resolve a dotted import path to the actual class.

    Example:
        resolve_class("dictionary_learning.trainers.batch_top_k.BatchTopKTrainer")
        → <class BatchTopKTrainer>
    """
    module_path, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).parent / "configs"

_DTYPE_MAP = {
    "float16": t.float16,
    "float32": t.float32,
    "bfloat16": t.bfloat16,
}


def _load_yaml(name: str) -> dict:
    with open(_CONFIGS_DIR / name) as f:
        return yaml.safe_load(f)


def _load_models() -> dict:
    raw = _load_yaml("models.yaml")
    configs = {}
    for model_name, v in raw.items():
        activault = None
        if "activault" in v:
            a = v["activault"]
            activault = ActivaultConfig(
                s3_prefix=a["s3_prefix"],
                s3_bucket=a.get("s3_bucket", "activations"),
                s3_buffer_size=a.get("s3_buffer_size", 2),
                s3_workers=a.get("s3_workers", 2),
            )
        configs[model_name] = LLMConfig(
            llm_batch_size=v["llm_batch_size"],
            context_length=v["context_length"],
            sae_batch_size=v["sae_batch_size"],
            dtype=_DTYPE_MAP[v["dtype"]],
            activault=activault,
        )
    return configs


def _load_architectures() -> dict:
    return _load_yaml("architectures.yaml")


# ---------------------------------------------------------------------------
# Enums & config models
# ---------------------------------------------------------------------------


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    Matryoshka_BATCH_TOP_K = "matryoshka_batch_top_k"
    THRESHOLDING_TOP_K = "thresholding_topk"
    NESTED_THRESHOLDING_TOP_K = "nested_thresholding_topk"
    STIEFEL_NESTED_THRESHOLDING_TOP_K = "stiefel_nested_thresholding_topk"
    MATCHING_PURSUIT = "matching_pursuit"
    TEMPORAL_MATRYOSHKA_BATCH_TOP_K = "temporal_matryoshka_batch_top_k"
    TEMPORAL_BATCH_TOP_K = "temporal_batch_top_k"


class ActivaultConfig(BaseModel):
    s3_prefix: str
    s3_bucket: str = "activations"
    s3_buffer_size: int = 2
    s3_workers: int = 2
    s3_concurrency: int = 5


class LLMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype
    activault: ActivaultConfig | None = None


class SparsityPenalties(BaseModel):
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


# ---------------------------------------------------------------------------
# Load configs from YAML at import time (replaces module-level globals)
# ---------------------------------------------------------------------------

_arch_cfg = _load_architectures()

LLM_CONFIG = _load_models()

SPARSITY_PENALTIES = SparsityPenalties(**_arch_cfg["sparsity_penalties"])
TARGET_L0s = _arch_cfg["target_l0s"]

_training_params = _arch_cfg["training_params"]
WARMUP_STEPS = _training_params["warmup_steps"]
SPARSITY_WARMUP_STEPS = _training_params["sparsity_warmup_steps"]
DECAY_START_FRACTION = _training_params["decay_start_fraction"]
K_ANNEAL_END_FRACTION = _training_params["k_anneal_end_fraction"]

TEMPORAL_TEMP_ALPHAS = _arch_cfg.get("temporal_temp_alphas", [0.1])
TEMPORAL_CONTRASTIVE = _arch_cfg.get("temporal_contrastive", [False])


# ---------------------------------------------------------------------------
# Trainer config models — string import paths instead of Type[Any]
# ---------------------------------------------------------------------------


class BaseTrainerConfig(BaseModel):
    activation_dim: int
    device: str
    layer: int
    lm_name: str
    submodule_name: str
    trainer: str
    dict_class: str
    warmup_steps: int
    steps: int
    decay_start: int | None


class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: int | None
    resample_steps: int | None = None


class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: int | None


class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: int | None
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: int | None = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: int | None = None


class NestedThresholdingTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k_values: list[int]
    k_weights: list[float] | None = None
    auxk_alpha: float = 1 / 32


class MatchingPursuitTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    s: int
    auxk_alpha: float = 1 / 32
    s_anneal_steps: int | None = None


class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = Field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: list[float] | None = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: int | None = None


class TemporalMatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    temporal: str
    contrastive: bool
    temp_alpha: float = 0.1
    group_fractions: list[float] = Field(default_factory=lambda: [0.2, 0.8])
    group_weights: list[float] | None = Field(default_factory=lambda: [0.2, 0.8])
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000


class TemporalBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    temporal: str
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000


class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: int | None


class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: int | None
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


# ---------------------------------------------------------------------------
# Architecture resolution (YAML-driven, no Python registry)
# ---------------------------------------------------------------------------

_arch_defs: dict[str, dict[str, str]] = _arch_cfg.get("definitions", {})


def resolve_architecture(architecture: str) -> tuple[str, str]:
    """Look up (trainer_path, dict_class_path) for an architecture name.

    The mapping lives in architectures.yaml under ``definitions:``.
    New architectures = new YAML entries, no Python code changes.
    Callers that already have import path strings don't need this.
    """
    if architecture not in _arch_defs:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. Known: {sorted(_arch_defs)}"
        )
    d = _arch_defs[architecture]
    return d["trainer"], d["dict_class"]


def build_trainer_config(
    trainer: str,
    dict_class: str,
    activation_dim: int,
    dict_size: int,
    seed: int,
    lr: float,
    steps: int,
    device: str,
    layer: int,
    model_name: str,
    submodule_name: str,
    warmup_steps: int = WARMUP_STEPS,
    decay_start_fraction: float = DECAY_START_FRACTION,
    **arch_params,
) -> dict:
    """Build a single trainer config dict.

    ``trainer`` and ``dict_class`` are dotted import path strings resolved
    at training time by ``trainSAE()``. Architecture-specific params
    (k, l1_penalty, temporal, etc.) go in **arch_params.
    No Pydantic validation here -- the trainer constructor validates.
    """
    config = {
        "trainer": trainer,
        "dict_class": dict_class,
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": int(steps * decay_start_fraction),
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    config.update(arch_params)
    return config


def get_architecture_sweep_params(architecture: str, steps: int) -> list[dict]:
    """Return architecture-specific parameter sets for sweeping.

    Each dict contains all params specific to one point in the architecture's
    sweep dimension (e.g. one k value for top-k), including step-dependent
    computed params (k_anneal_steps, sparsity_warmup_steps).

    Cross-product these with (seeds x dict_sizes x learning_rates) to get
    the full sweep grid.
    """
    anneal_end = int(steps * K_ANNEAL_END_FRACTION)

    if architecture == TrainerType.STANDARD.value:
        return [
            {"l1_penalty": p, "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS}
            for p in SPARSITY_PENALTIES.standard
        ]
    elif architecture == TrainerType.STANDARD_NEW.value:
        return [
            {"l1_penalty": p, "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS}
            for p in SPARSITY_PENALTIES.standard_new
        ]
    elif architecture == TrainerType.GATED.value:
        return [
            {"l1_penalty": p, "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS}
            for p in SPARSITY_PENALTIES.gated
        ]
    elif architecture == TrainerType.P_ANNEAL.value:
        return [
            {
                "initial_sparsity_penalty": p,
                "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS,
            }
            for p in SPARSITY_PENALTIES.p_anneal
        ]
    elif architecture in (
        TrainerType.TOP_K.value,
        TrainerType.BATCH_TOP_K.value,
        TrainerType.Matryoshka_BATCH_TOP_K.value,
        TrainerType.THRESHOLDING_TOP_K.value,
    ):
        return [{"k": k, "k_anneal_steps": anneal_end} for k in TARGET_L0s]
    elif architecture == TrainerType.TEMPORAL_BATCH_TOP_K.value:
        return [{"k": k, "temporal": "p"} for k in TARGET_L0s]
    elif architecture == TrainerType.JUMP_RELU.value:
        return [
            {"target_l0": l0, "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS}
            for l0 in TARGET_L0s
        ]
    elif architecture == TrainerType.MATCHING_PURSUIT.value:
        return [{"s": s, "s_anneal_steps": anneal_end} for s in TARGET_L0s]
    elif architecture in (
        TrainerType.NESTED_THRESHOLDING_TOP_K.value,
        TrainerType.STIEFEL_NESTED_THRESHOLDING_TOP_K.value,
    ):
        k_values = sorted(TARGET_L0s)
        k_weights = [1.0 / len(k_values)] * len(k_values)
        return [{"k_values": k_values, "k_weights": k_weights}]
    elif architecture == TrainerType.TEMPORAL_MATRYOSHKA_BATCH_TOP_K.value:
        return [
            {"k": k, "temp_alpha": ta, "contrastive": c, "temporal": "p"}
            for k, ta, c in itertools.product(
                TARGET_L0s, TEMPORAL_TEMP_ALPHAS, TEMPORAL_CONTRASTIVE
            )
        ]
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")


# ---------------------------------------------------------------------------
# Sweep generation
# ---------------------------------------------------------------------------


def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: int,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
    anneal_end_fraction=K_ANNEAL_END_FRACTION,
) -> list[dict]:
    trainer_configs = []

    for architecture in architectures:
        trainer_path, dict_class_path = resolve_architecture(architecture)
        sweep = get_architecture_sweep_params(architecture, steps)

        for seed, dict_size, lr, arch_params in itertools.product(
            seeds,
            dict_sizes,
            learning_rates,
            sweep,
        ):
            config = build_trainer_config(
                trainer=trainer_path,
                dict_class=dict_class_path,
                activation_dim=activation_dim,
                dict_size=dict_size,
                seed=seed,
                lr=lr,
                steps=steps,
                device=device,
                layer=layer,
                model_name=model_name,
                submodule_name=submodule_name,
                warmup_steps=warmup_steps,
                decay_start_fraction=decay_start_fraction,
                **arch_params,
            )
            trainer_configs.append(config)

    return trainer_configs
