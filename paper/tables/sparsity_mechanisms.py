"""Sparsity-mechanisms table (appendix ``tab:sparsity_mechanisms``): hidden ReLU and L1 penalty on multibasin.

MSE rows: means over systems of per-system seed IQMs from
``forecasting_sparsity_ablation.csv``, with the two-level marks of the test against
Dense MLP; the Sparse MLP header carries ``\\S[K/N]``, the systems in which it
is significantly lower than Sparse MLP without the L1 penalty at every
horizon. Count rows: systems with significantly lower / higher MSE than the
ReLU-hidden Dense MLP. Sparsity rows: the summaries of
``sparsity_mechanisms.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from paper.config import SEEDS, systems_for

from .common import (DENSE, HORIZONS, MISSING, Coverage, cell, passing, read_forecasts, row, system_mean,
                     two_level_marks, write)

MODELS = (DENSE, "dense_mlp_relu", "sparse_mlp_l1_0", "sparse_mlp", "lista")
RELU = "dense_mlp_relu"


def build(evidence: Path, out: Path, coverage: Coverage) -> Path:
    """The table from ``forecasting_sparsity_ablation.csv`` and ``sparsity_mechanisms.json``."""
    evidence, systems, horizons = Path(evidence), systems_for("multibasin"), HORIZONS["multibasin"]
    values = read_forecasts(evidence / "forecasting_sparsity_ablation.csv")
    summary = json.loads((evidence / "sparsity_mechanisms.json").read_text())
    forecasts = {m: all([coverage.complete(values, m, systems, h) for h in horizons]) for m in MODELS}
    measured = {m: coverage.count(m, summary.get("runs", {}).get(m, 0), len(systems) * len(SEEDS)) for m in MODELS}
    l1 = set.intersection(*(passing(values, "sparse_mlp", "sparse_mlp_l1_0", h, systems) for h in horizons))
    lines = [r"\begin{tabular}{@{}l ccccc@{}}", r"\toprule",
             r"& Dense MLP & Dense MLP, & Sparse MLP, & & \\",
             r"& (tanh hidden) & ReLU hidden & \(\lambda_{\rm sp}=0\) & "
             rf"Sparse MLP\(^{{\S[{len(l1)}/{len(systems)}]}}\) & LISTA \\",
             r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Forecasting MSE}\(\downarrow\)} \\"]
    for h in horizons:
        lines.append(row(rf"\quad \(H={h}\)", [
            cell(system_mean(values, m, systems, h),
                 two_level_marks(passing(values, m, DENSE, h, systems), systems) if m != DENSE else "",
                 missing=not forecasts[m])
            for m in MODELS]))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{Systems with significantly lower / higher MSE than the ReLU-hidden "
                 r"Dense MLP} (\(H=100/500/1000\))} \\")

    def counts(candidate, reference):
        flag = "" if forecasts[candidate] and forecasts[reference] else MISSING
        return "/".join(str(len(passing(values, candidate, reference, h, systems))) for h in horizons) + flag

    lines.append(row(r"\quad lower", [counts(m, RELU) if m != RELU else "--" for m in MODELS]))
    lines.append(row(r"\quad higher", [counts(RELU, m) if m != RELU else "--" for m in MODELS]))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{Fraction of inactive code coordinates}} \\")
    inactive, hidden = summary["inactive_mean_of_system_means"], summary["hidden_zero_mean_of_system_means"]
    for target in ("encoded", "rollout"):
        lines.append(row(rf"\quad {target}", [cell(inactive[f"{m}/{target}"], missing=not measured[m])
                                                for m in MODELS]))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{Fraction of exactly zero hidden activations}} \\")
    for layer in (1, 2):
        lines.append(row(rf"\quad layer {layer}", [cell(hidden[f"{m}/layer{layer}"], missing=not measured[m])
                                                     for m in MODELS]))
    return write(Path(out), lines + [r"\bottomrule", r"\end{tabular}"])
