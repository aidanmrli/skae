"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Architecture and loss weights of a :class:`skae.KoopmanAutoencoder`.

    The defaults are the paper's LISTA model on the multibasin benchmark. The
    other models of the paper change these fields:

    ===============  ==========================================================
    Sparse MLP       ``encoder="mlp", mlp_output_relu=True``
    Dense MLP        ``encoder="mlp", activation="tanh", sparsity_weight=0``
    -BD models       ``koopman="block_diagonal", num_blocks=<number of basins>``
    LISTA-SB         ``num_blocks=<number of basins>, soft_block_weight=1e-4``
    Dysts runs       ``input_normalization=True, reconstruction_weight=1.0,
                     loss_time_reduction="mean", num_blocks=2``, and
                     ``sparsity_weight=0.03`` except for Dense MLP; LISTA also
                     ``lista_threshold="nonnegative", lista_output="none",
                     lista_learn_thresholds=True``; -BD models also
                     ``koopman_rollout="assembled"``
    ===============  ==========================================================

    ``paper.config.paper_config`` builds the exact configuration of every run.
    """

    encoder: str = "lista"                # "lista" or "mlp"
    latent_dim: int = 256                 # width of the code z
    hidden: Tuple[int, ...] = (64, 64)    # hidden widths of the encoder MLP (LISTA: its pre-code MLP)
    activation: str = "relu"              # hidden activation: "relu" or "tanh"
    mlp_output_relu: bool = False         # MLP encoder: ReLU code (sparse) or linear code (dense)
    lista_loops: int = 2                  # unrolled LISTA iterations
    lista_alpha: float = 0.15             # the LISTA threshold is lista_alpha / L
    lista_output: str = "sign_split"      # "sign_split": z -> [relu(z), relu(-z)]; "none": z as is
    lista_threshold: str = "soft"         # "soft": sign(u) max(|u| - t, 0); "nonnegative": max(u - t, 0)
    lista_learn_thresholds: bool = False  # one learned threshold per code coordinate
    input_normalization: bool = False     # the model stores a per-coordinate mean and scale
    koopman: str = "dense"                # K: "dense" or "block_diagonal"
    koopman_rollout: str = "per_block"    # block-diagonal K: multiply block by block, or "assembled" K
    num_blocks: int = 0                   # diagonal blocks of K (block-diagonal K or soft-block penalty)
    soft_block_weight: float = 0.0        # L1 penalty on the entries of a dense K outside its blocks
    prediction_weight: float = 1.0        # weight of the forecast error of D(E(x_t) K^h)
    alignment_weight: float = 1.0         # weight of the gap between E(x_t) K^h and E(x_(t+h))
    reconstruction_weight: float = 0.03   # weight of the reconstruction error of D(E(x))
    sparsity_weight: float = 3e-3         # L1 penalty on the codes
    loss_time_reduction: str = "divide_horizon"  # "mean" over the L steps, or "divide_horizon": mean / L
