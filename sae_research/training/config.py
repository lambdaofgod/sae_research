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


def _load_defaults() -> dict:
    return _load_yaml("defaults.yaml")


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
_defaults_cfg = _load_defaults()

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

num_tokens = _defaults_cfg["num_tokens"]
print(f"NOTE: Training on {num_tokens} tokens")
eval_num_inputs = _defaults_cfg["eval_num_inputs"]
random_seeds = _defaults_cfg["random_seeds"]
dictionary_widths = _defaults_cfg["dictionary_widths"]
learning_rates = _defaults_cfg["learning_rates"]
remove_bos = _defaults_cfg["remove_bos"]
max_activation_norm_multiple = _defaults_cfg["max_activation_norm_multiple"]
mlflow_experiment = _defaults_cfg["mlflow_experiment"]


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
    decay_start = int(steps * decay_start_fraction)
    anneal_end = int(steps * anneal_end_fraction)

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    if TrainerType.P_ANNEAL.value in architectures:
        for seed, dict_size, learning_rate, sparsity_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.p_anneal.PAnnealTrainer",
                dict_class="dictionary_learning.dictionary.AutoEncoder",
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.standard.StandardTrainer",
                dict_class="dictionary_learning.dictionary.AutoEncoder",
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.STANDARD_NEW.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new
        ):
            config = StandardNewTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.standard.StandardTrainerAprilUpdate",
                dict_class="dictionary_learning.dictionary.AutoEncoder",
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.GATED.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated
        ):
            config = GatedTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.gdm.GatedSAETrainer",
                dict_class="dictionary_learning.dictionary.GatedAutoEncoder",
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.top_k.TopKTrainer",
                dict_class="dictionary_learning.trainers.top_k.AutoEncoderTopK",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.batch_top_k.BatchTopKTrainer",
                dict_class="dictionary_learning.trainers.batch_top_k.BatchTopKSAE",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.Matryoshka_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.matryoshka_batch_top_k.MatryoshkaBatchTopKTrainer",
                dict_class="dictionary_learning.trainers.matryoshka_batch_top_k.MatryoshkaBatchTopKSAE",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.JUMP_RELU.value in architectures:
        for seed, dict_size, learning_rate, target_l0 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = JumpReluTrainerConfig(
                **base_config,
                trainer="dictionary_learning.trainers.jumprelu.JumpReluTrainer",
                dict_class="dictionary_learning.dictionary.JumpReluAutoEncoder",
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.THRESHOLDING_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer="sae_research.thresholding_sae.ThresholdingTopKTrainer",
                dict_class="sae_research.thresholding_sae.ThresholdingAutoEncoderTopK",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.MATCHING_PURSUIT.value in architectures:
        for seed, dict_size, learning_rate, s in itertools.product(
            seeds,
            dict_sizes,
            learning_rates,
            TARGET_L0s,
        ):
            config = MatchingPursuitTrainerConfig(
                **base_config,
                trainer="sae_research.matching_pursuit_sae.MatchingPursuitTrainer",
                dict_class="sae_research.matching_pursuit_sae.MatchingPursuitAutoEncoder",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                s=s,
                s_anneal_steps=anneal_end,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.NESTED_THRESHOLDING_TOP_K.value in architectures:
        for seed, dict_size, learning_rate in itertools.product(
            seeds, dict_sizes, learning_rates
        ):
            k_values = sorted(TARGET_L0s)
            k_weights = [1.0 / len(k_values)] * len(k_values)

            config = NestedThresholdingTopKTrainerConfig(
                **base_config,
                trainer="sae_research.thresholding_sae.NestedThresholdingTopKTrainer",
                dict_class="sae_research.thresholding_sae.NestedThresholdingAutoEncoderTopK",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k_values=k_values,
                k_weights=k_weights,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.STIEFEL_NESTED_THRESHOLDING_TOP_K.value in architectures:
        for seed, dict_size, learning_rate in itertools.product(
            seeds, dict_sizes, learning_rates
        ):
            k_values = sorted(TARGET_L0s)
            k_weights = [1.0 / len(k_values)] * len(k_values)

            config = NestedThresholdingTopKTrainerConfig(
                **base_config,
                trainer="sae_research.stiefel_sae.StiefelNestedThresholdingTopKTrainer",
                dict_class="sae_research.stiefel_sae.StiefelNestedThresholdingAutoEncoderTopK",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k_values=k_values,
                k_weights=k_weights,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.TEMPORAL_MATRYOSHKA_BATCH_TOP_K.value in architectures:
        for (
            seed,
            dict_size,
            learning_rate,
            k,
            temp_alpha,
            contrastive,
        ) in itertools.product(
            seeds,
            dict_sizes,
            learning_rates,
            TARGET_L0s,
            TEMPORAL_TEMP_ALPHAS,
            TEMPORAL_CONTRASTIVE,
        ):
            config = TemporalMatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer="sae_research.temporal_sae.TemporalMatryoshkaBatchTopKTrainer",
                dict_class="sae_research.temporal_sae.TemporalMatryoshkaBatchTopKSAE",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                temporal="p",
                contrastive=contrastive,
                temp_alpha=temp_alpha,
            )
            trainer_configs.append(config.model_dump())

    if TrainerType.TEMPORAL_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds,
            dict_sizes,
            learning_rates,
            TARGET_L0s,
        ):
            config = TemporalBatchTopKTrainerConfig(
                **base_config,
                trainer="sae_research.temporal_sae.TemporalBatchTopKTrainer",
                dict_class="sae_research.temporal_sae.TemporalBatchTopKSAE",
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                temporal="p",
            )
            trainer_configs.append(config.model_dump())

    return trainer_configs
