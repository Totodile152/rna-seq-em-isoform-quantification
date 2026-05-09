"""Evaluation metrics and ambiguity summaries."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .data_structures import Transcript


def get_true_abundances(transcripts: List[Transcript]) -> Dict[str, float]:
    """Extract the ground-truth abundances used in simulation."""
    return {t.name: t.abundance for t in transcripts}


def compute_metrics(true_mu: Dict[str, float], est_mu: Dict[str, float]) -> Dict[str, float]:
    """
    Print and return evaluation metrics:
        - mean squared error
        - Pearson correlation
    """
    metrics = compute_metrics_dict(true_mu, est_mu)

    print("\nEvaluation Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")

    print("\nTrue vs Estimated:")
    for name in true_mu.keys():
        print(f"  {name}: true={true_mu[name]:.4f}, est={est_mu[name]:.4f}")

    return metrics


def compute_metrics_dict(true_mu: Dict[str, float], est_mu: Dict[str, float]) -> Dict[str, float]:
    """Return metrics as a dictionary for experiments."""
    names = list(true_mu.keys())
    true_vals = np.array([true_mu[n] for n in names])
    est_vals = np.array([est_mu[n] for n in names])

    mse = np.mean((true_vals - est_vals) ** 2)

    # Correlation can be undefined if a vector is constant; eturn NaN in that case.
    if np.std(true_vals) == 0 or np.std(est_vals) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(true_vals, est_vals)[0, 1]

    return {
        "mse": float(mse),
        "correlation": float(corr),
    }


def get_ambiguity_stats(
    compatibility_df: pd.DataFrame,
    transcripts: List[Transcript],
) -> Dict[str, int]:
    """Count unique, ambiguous, and incompatible fragments."""
    compatible_cols = [f"{t.name}_compatible" for t in transcripts]
    num_compatible = compatibility_df[compatible_cols].sum(axis=1)

    unique = int((num_compatible == 1).sum())
    ambiguous = int((num_compatible > 1).sum())
    incompatible = int((num_compatible == 0).sum())

    return {
        "unique": unique,
        "ambiguous": ambiguous,
        "incompatible": incompatible,
    }
