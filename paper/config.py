"""Configurations of the paper's runs.

``paper_config(benchmark, system, model, seed)`` returns the configuration of
one run: one of the six models of Table 1 (or, on the multibasin benchmark, one
of the five ablations in ``ABLATIONS``) on one of the 25 systems, with one
seed. ``paper_runs()`` lists every run of the paper and ``run_dir`` gives the
directory a run writes to.

Settings shared by every run are module constants below; settings that differ
between runs are fields of :class:`TrainConfig`, :class:`EvalConfig` and
:class:`skae.config.ModelConfig`. The defaults of all three are the settings of
the multibasin LISTA runs. The code reads the module constants when it runs (it
keeps no copies), so a script can change them by patching this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator, Tuple

from skae.config import ModelConfig
from paper.systems import DYSTS_SYSTEMS, MULTIBASIN_SYSTEMS

BENCHMARKS = ("multibasin", "dysts")
MODELS = ("lista", "lista_bd", "lista_sb", "sparse_mlp", "sparse_mlp_bd", "dense_mlp")
# Multibasin ablations: name -> base model and its ModelConfig / TrainConfig
# overrides. An ablation has its own run directory and pairs with the base
# model's run of the same system and seed.
ABLATIONS = {
    # Sparse MLP without the L1 penalty: ReLU sparsity only.
    "sparse_mlp_l1_0": {"base": "sparse_mlp", "model": {"sparsity_weight": 0.0}},
    # Training windows of 64 states (L = 63 predicted steps) instead of 9.
    "lista_w64": {"base": "lista", "train": {"window": 63}},
    "sparse_mlp_w64": {"base": "sparse_mlp", "train": {"window": 63}},
    "dense_mlp_w64": {"base": "dense_mlp", "train": {"window": 63}},
    # Hidden-activation control: ReLU hidden layers, linear code and no L1, as for Dense MLP.
    "dense_mlp_relu": {"base": "dense_mlp", "model": {"activation": "relu"}},
}
WINDOW64_ABLATIONS = ("lista_w64", "sparse_mlp_w64", "dense_mlp_w64")
SEEDS = tuple(range(15))
WINDOW64_SEEDS = tuple(range(5))  # the 64-state-window ablation uses seeds 0-4

# Updates per run.
PAPER_STEPS = {"multibasin": 200_000, "dysts": 12_000}

# Evaluation (both benchmarks): forecasts from TEST_STARTS test starts; each run's
# re-encoding period is the one with the lowest MSE at SELECTION_HORIZON on
# VALIDATION_STARTS validation starts.
TEST_STARTS = 100
VALIDATION_STARTS = 100
SELECTION_HORIZON = 500
# A start diverges at its first non-finite prediction or prediction farther from
# the origin than DIVERGENCE_RADIUS x the largest training-state norm; it is then
# forecast by the training-mean state.
DIVERGENCE_RADIUS = 10.0
# Training batches cycle through BATCH_STREAMS seeded streams.
BATCH_STREAMS = 8
# A code coordinate is active when |z_i| > SUPPORT_THRESHOLD (supports and their analyses, and the
# frozen-support re-encoding ablation).
SUPPORT_THRESHOLD = 1e-3

# Generator seeds. Every random draw of a run with seed s comes from a generator
# seeded as below; i counts the draws of one kind. On Dysts, one generator per
# training stream, data split or set of starts draws all of its windows or
# initial conditions, seeded without the "+ i".
#
#   draw                                    seed                                     code (paper/)
#   model initialization                    torch.manual_seed(s)                     training/train.py
#   multibasin training stream k, reset i   s + k * batch_size + i                   data.py
#   Dysts training stream k                 s + k * batch_size                       training/data.py
#   hard-initial-condition candidate i      s + HARD_INIT_SEED_OFFSET + i            data.py
#   hard-initial-condition perturbations    s + HARD_INIT_NOISE_SEED_OFFSET          data.py
#   test start i                            s + TEST_SEED_OFFSET + i                 data.py
#   validation start i                      s + VALIDATION_SEED_OFFSET + i           data.py
#   support validation trajectory i         s + ANALYSIS_VALIDATION_SEED_OFFSET + i  analysis/support_transfer.py
#   support and sparsity test trajectory i  s + ANALYSIS_TEST_SEED_OFFSET + i        analysis/support_transfer.py,
#                                                                                    analysis/sparsity_mechanisms.py
#   intervention trajectory i               42 + i                                   analysis/interventions.py
#   intervention starts                     42 + s                                   analysis/interventions.py
#   intervention random support r           123 + r                                  analysis/interventions.py
#   Dysts data splits                       train 0, val 1, test 2                   systems/dysts.py
#   classical-baseline RBF centers          s (Dysts selection fits: 9000)           baselines/
#   Dysts baseline selection starts         9000 + VALIDATION_SEED_OFFSET            baselines/dysts.py
TEST_SEED_OFFSET = 12_345
VALIDATION_SEED_OFFSET = 54_321
HARD_INIT_SEED_OFFSET = 2_000_000
HARD_INIT_NOISE_SEED_OFFSET = 3_000_000
ANALYSIS_VALIDATION_SEED_OFFSET = 3_000_000
ANALYSIS_TEST_SEED_OFFSET = 4_000_000

# Dysts runs (their trajectory data are defined in paper.systems.dysts).
DYSTS_BLOCKS = 2                  # blocks of K on Dysts (used by LISTA-BD, LISTA-SB and Sparse MLP-BD)
DYSTS_SPARSITY_WEIGHT = 0.03
DYSTS_BATCH_SIZE = 512
DYSTS_WINDOW = 63
DYSTS_LR = 1e-3
DYSTS_KOOPMAN_LR = 2e-4
DYSTS_METRICS_EVERY = 64
# "cosine_tail" learning-rate schedule (paper.training.train.lr_factor): constant
# for the first 75% of the updates, then a cosine decay to 1% of the base rate.
LR_TAIL_FRACTION = 0.25
LR_FINAL_FACTOR = 0.01
LR_SCHEDULES = ("constant", "cosine_tail")


@dataclass
class TrainConfig:
    """Optimization settings of a run; ``steps`` counts updates, one batch each."""

    steps: int = 200_000
    batch_size: int = 256
    window: int = 8                       # L: predicted steps per training window
    lr: float = 5e-5                      # encoder/decoder learning rate
    koopman_lr: float = 5e-6              # learning rate of K
    weight_decay: float = 1e-4            # AdamW weight decay (not applied to K)
    lr_schedule: str = "constant"         # "constant" or "cosine_tail"
    metrics_every: int = 100              # training-log cadence (updates)
    hard_init: bool = True                # oversample hard initial states (multibasin)

    def __post_init__(self):
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"TrainConfig.lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")


@dataclass
class EvalConfig:
    """Forecast horizons of the reported test MSE and candidate re-encoding periods, both in stored steps."""

    horizons: Tuple[int, ...] = tuple(range(100, 1001, 100))  # horizons H of the reported test MSE
    periods: Tuple[int, ...] = (10, 25, 50, 100)              # candidate re-encoding periods


@dataclass
class RunConfig:
    """One run: its names, seed and settings; saved as the run's ``config.json``."""

    benchmark: str
    system: str
    model_name: str
    seed: int
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_name(self) -> str:
        return f"{self.benchmark}/{self.system}/{self.model_name}/seed_{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        """The inverse of :meth:`to_dict`; also reads JSON, whose lists become the configurations' tuples."""
        model = dict(payload["model"])
        model["hidden"] = tuple(model["hidden"])
        evaluation = dict(payload["eval"])
        for key in ("horizons", "periods"):
            evaluation[key] = tuple(evaluation[key])
        return cls(
            benchmark=payload["benchmark"],
            system=payload["system"],
            model_name=payload["model_name"],
            seed=int(payload["seed"]),
            model=ModelConfig(**model),
            train=TrainConfig(**payload["train"]),
            eval=EvalConfig(**evaluation),
        )

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def systems_for(benchmark: str) -> Tuple[str, ...]:
    """The systems of a benchmark, in paper order."""
    if benchmark == "multibasin":
        return tuple(MULTIBASIN_SYSTEMS)
    if benchmark == "dysts":
        return tuple(DYSTS_SYSTEMS)
    raise ValueError(f"unknown benchmark {benchmark!r}; choose from {BENCHMARKS}")


def models_for(benchmark: str, ablations: bool = True) -> Tuple[str, ...]:
    """The models of a benchmark; multibasin ablations are included when ``ablations``."""
    return MODELS + tuple(ABLATIONS) if benchmark == "multibasin" and ablations else MODELS


def seeds_for(model: str) -> Tuple[int, ...]:
    """The paper's seeds of a model."""
    return WINDOW64_SEEDS if model in WINDOW64_ABLATIONS else SEEDS


def paper_runs(benchmarks=BENCHMARKS, ablations: bool = True) -> Iterator[Tuple[str, str, str, int]]:
    """Every paper run as ``(benchmark, system, model, seed)``, in benchmark/system/model/seed order."""
    for benchmark in benchmarks:
        for system in systems_for(benchmark):
            for model in models_for(benchmark, ablations):
                for seed in seeds_for(model):
                    yield benchmark, system, model, seed


def _multibasin_model(model: str, num_basins: int) -> ModelConfig:
    """The ModelConfig of a model of ``MODELS`` on a multibasin system with ``num_basins`` basins."""
    if model in ("lista", "lista_bd", "lista_sb"):
        cfg = ModelConfig()
    elif model in ("sparse_mlp", "sparse_mlp_bd", "dense_mlp"):
        dense = model == "dense_mlp"
        cfg = ModelConfig(
            encoder="mlp",
            activation="tanh" if dense else "relu",
            mlp_output_relu=not dense,
            sparsity_weight=0.0 if dense else 3e-3,
        )
    else:
        raise ValueError(f"no multibasin model configuration for {model!r}")
    if model.endswith("_bd"):
        cfg.koopman = "block_diagonal"
        cfg.num_blocks = num_basins
    if model == "lista_sb":
        cfg.num_blocks = num_basins
        cfg.soft_block_weight = 1e-4
    return cfg


def _dysts_model(model: str) -> ModelConfig:
    """The ModelConfig of a model of ``MODELS`` on a Dysts system."""
    common = dict(input_normalization=True, reconstruction_weight=1.0, loss_time_reduction="mean",
                  num_blocks=DYSTS_BLOCKS)
    if model in ("lista", "lista_bd", "lista_sb"):
        cfg = ModelConfig(
            lista_threshold="nonnegative",
            lista_output="none",
            lista_learn_thresholds=True,
            sparsity_weight=DYSTS_SPARSITY_WEIGHT,
            **common,
        )
    elif model in ("sparse_mlp", "sparse_mlp_bd", "dense_mlp"):
        dense = model == "dense_mlp"
        cfg = ModelConfig(
            encoder="mlp",
            activation="tanh" if dense else "relu",
            mlp_output_relu=not dense,
            sparsity_weight=0.0 if dense else DYSTS_SPARSITY_WEIGHT,
            **common,
        )
    else:
        raise ValueError(f"no Dysts model configuration for {model!r}")
    if model.endswith("_bd"):
        cfg.koopman = "block_diagonal"
        cfg.koopman_rollout = "assembled"
    if model == "lista_sb":
        cfg.soft_block_weight = 1e-4
    return cfg


def paper_config(benchmark: str, system: str, model: str, seed: int) -> RunConfig:
    """The configuration of one paper run.

    ``benchmark`` is one of ``BENCHMARKS``; ``system`` one of ``systems_for(benchmark)``;
    ``model`` one of ``MODELS`` or, on the multibasin benchmark, of ``ABLATIONS``; ``seed`` any
    integer (the paper's are ``seeds_for(model)``). Raises ``ValueError`` for an unknown
    benchmark, system or model.
    """
    if system not in systems_for(benchmark):
        raise ValueError(f"unknown {benchmark} system {system!r}; choose from {systems_for(benchmark)}")
    if model not in models_for(benchmark):
        raise ValueError(f"unknown {benchmark} model {model!r}; choose from {models_for(benchmark)}")
    ablation = ABLATIONS.get(model, {})
    base = ablation.get("base", model)
    if benchmark == "multibasin":
        cfg = RunConfig(
            benchmark, system, model, int(seed),
            model=_multibasin_model(base, MULTIBASIN_SYSTEMS[system].num_basins),
        )
    else:
        cfg = RunConfig(
            benchmark, system, model, int(seed),
            model=_dysts_model(base),
            train=TrainConfig(
                steps=PAPER_STEPS["dysts"],
                batch_size=DYSTS_BATCH_SIZE,
                window=DYSTS_WINDOW,
                lr=DYSTS_LR,
                koopman_lr=DYSTS_KOOPMAN_LR,
                weight_decay=0.0,
                lr_schedule="cosine_tail",
                metrics_every=DYSTS_METRICS_EVERY,
                hard_init=False,
            ),
            eval=EvalConfig(
                horizons=(100, 500, 1000, 1500, 2000, 3000, 4000, 5000),
                periods=(10, 25, 50, 100, 150, 200),
            ),
        )
    if "model" in ablation:
        cfg.model = replace(cfg.model, **ablation["model"])
    if "train" in ablation:
        cfg.train = replace(cfg.train, **ablation["train"])
    return cfg


def run_dir(root: Path, cfg: RunConfig) -> Path:
    """``<root>/<benchmark>/<system>/<model>/seed_<seed>``, with ``_steps<N>`` for shortened runs."""
    suffix = "" if cfg.train.steps == PAPER_STEPS[cfg.benchmark] else f"_steps{cfg.train.steps}"
    return Path(root) / cfg.benchmark / cfg.system / cfg.model_name / f"seed_{cfg.seed}{suffix}"


def find_run(root: Path, benchmark: str, system: str, model: str, seed: int, include_short: bool = False) -> Path:
    """The directory of a paper run under ``root``.

    With ``include_short``, the longest shortened run (``seed_<seed>_steps<N>``, e.g. of a
    quick check) stands in when the full-length run has no ``final.pt``.
    """
    path = run_dir(root, paper_config(benchmark, system, model, seed))
    shortened = list(path.parent.glob(f"{path.name}_steps*/final.pt")) if include_short else []
    if (path / "final.pt").exists() or not shortened:
        return path
    return max(shortened, key=lambda final: int(final.parent.name.rsplit("steps", 1)[1])).parent

