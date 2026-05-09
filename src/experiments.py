"""Reusable experiment runners for toy isoform model."""

from __future__ import annotations

from typing import Dict

import random

from .compatibility import build_paired_end_compatibility_table
from .evaluation import compute_metrics_dict, get_ambiguity_stats, get_true_abundances
from .inference import compute_likelihood_matrix, run_em_algorithm, run_em_algorithm_length_adjusted
from .simulation import build_example_transcripts, simulate_paired_end_fragments


AMBIGUITY_SETTINGS = {
    "hard": {
        "E1": 120,
        "E2": 30,
        "E3": 70,
        "E4_short": 80,
        "E4_long_extra": 20,
    },
    "medium": {
        "E1": 120,
        "E2": 60,
        "E3": 70,
        "E4_short": 80,
        "E4_long_extra": 40,
    },
    "easy": {
        "E1": 120,
        "E2": 100,
        "E3": 70,
        "E4_short": 80,
        "E4_long_extra": 80,
    },
}


def run_one_experiment(
    exon_lengths: Dict[str, int],
    seed: int = 42,
    num_fragments: int = 200,
    read_length: int = 30,
    min_fragment_length: int = 80,
    max_fragment_length: int = 140,
    num_iterations: int = 10,
    verbose: bool = True,
):
    """Run full RNA-seq toy pipeline on one transcript design setting."""
    rng = random.Random(seed)

    transcripts = build_example_transcripts(rng, exon_lengths=exon_lengths)
    true_mu = get_true_abundances(transcripts)

    fragments = simulate_paired_end_fragments(
        transcripts=transcripts,
        num_fragments=num_fragments,
        read_length=read_length,
        min_fragment_length=min_fragment_length,
        max_fragment_length=max_fragment_length,
        rng=rng,
    )

    compatibility_df = build_paired_end_compatibility_table(
        fragments=fragments,
        transcripts=transcripts,
        read_length=read_length,
    )

    ambiguity_stats = get_ambiguity_stats(compatibility_df, transcripts)

    likelihood_df = compute_likelihood_matrix(
        compatibility_df=compatibility_df,
        transcripts=transcripts,
    )

    baseline_mu, baseline_history = run_em_algorithm(
        likelihood_df=likelihood_df,
        transcripts=transcripts,
        num_iterations=num_iterations,
        verbose=verbose,
    )

    length_mu, length_history, effective_lengths = run_em_algorithm_length_adjusted(
        likelihood_df=likelihood_df,
        transcripts=transcripts,
        num_iterations=num_iterations,
        verbose=verbose,
    )

    baseline_metrics = compute_metrics_dict(true_mu, baseline_mu)
    length_metrics = compute_metrics_dict(true_mu, length_mu)

    return {
        "true_mu": true_mu,
        "transcripts": transcripts,
        "fragments": fragments,
        "compatibility_df": compatibility_df,
        "likelihood_df": likelihood_df,
        "ambiguity": ambiguity_stats,
        "baseline_mu": baseline_mu,
        "baseline_history": baseline_history,
        "baseline_metrics": baseline_metrics,
        "length_mu": length_mu,
        "length_history": length_history,
        "length_metrics": length_metrics,
        "effective_lengths": effective_lengths,
    }


def run_ambiguity_experiments(verbose: bool = False):
    """Run hard, medium, and easy toy-model settings."""
    results = {}

    for setting_name, exon_lengths in AMBIGUITY_SETTINGS.items():
        print(f"\n\n===== Running setting: {setting_name} =====")
        results[setting_name] = run_one_experiment(
            exon_lengths=exon_lengths,
            verbose=verbose,
        )

    print("\n\n===== SUMMARY TABLE =====")
    for setting_name, result in results.items():
        print(f"\nSetting: {setting_name}")
        print(f"  Unique fragments:     {result['ambiguity']['unique']}")
        print(f"  Ambiguous fragments:  {result['ambiguity']['ambiguous']}")
        print(f"  Incompatible:         {result['ambiguity']['incompatible']}")

        print("  Baseline EM:")
        print(f"    MSE = {result['baseline_metrics']['mse']:.6f}")
        print(f"    Corr = {result['baseline_metrics']['correlation']:.4f}")

        print("  Length-adjusted EM:")
        print(f"    MSE = {result['length_metrics']['mse']:.6f}")
        print(f"    Corr = {result['length_metrics']['correlation']:.4f}")

    return results
