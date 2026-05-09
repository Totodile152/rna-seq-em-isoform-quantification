"""Likelihood, posterior, and EM abundance inference."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_structures import Transcript
from .simulation import Transcript


def compute_likelihood_matrix(
    compatibility_df: pd.DataFrame,
    transcripts: List[Transcript],
) -> pd.DataFrame:
    """
    Compute fragment likelihoods from placement counts.

    For transcript k and fragment i:

        likelihood = placements / (L_k - fragment_length + 1)

    This normalizes by number of possible positions in transcript.
    """
    transcript_lengths = {t.name: len(t.sequence) for t in transcripts}
    rows = []

    for _, row in compatibility_df.iterrows():
        frag_id = row["fragment_id"]
        true_t = row["true_transcript"]
        frag_len = row["fragment_length"]

        out_row = {
            "fragment_id": frag_id,
            "true_transcript": true_t,
            "fragment_length": frag_len,
        }

        for transcript in transcripts:
            t_name = transcript.name
            placements = row[f"{t_name}_placements"]
            total_possible_positions = transcript_lengths[t_name] - frag_len + 1

            if total_possible_positions <= 0:
                likelihood = 0.0
            else:
                likelihood = placements / total_possible_positions

            out_row[f"{t_name}_likelihood"] = likelihood

        rows.append(out_row)

    return pd.DataFrame(rows)


def summarize_likelihood_matrix(likelihood_df: pd.DataFrame, transcripts: List[Transcript]) -> None:
    """Print summary statistics for likelihoods."""
    print("\nLikelihood summary:")

    for transcript in transcripts:
        col = f"{transcript.name}_likelihood"
        nonzero = (likelihood_df[col] > 0).sum()
        mean_val = likelihood_df[col].mean()
        max_val = likelihood_df[col].max()

        print(f"  {transcript.name}: nonzero={nonzero}, mean={mean_val:.6f}, max={max_val:.6f}")

    display_cols = (
        ["fragment_id", "true_transcript", "fragment_length"]
        + [f"{t.name}_likelihood" for t in transcripts]
    )
    print("\nFirst 5 likelihood rows:")
    print(likelihood_df[display_cols].head())


def check_true_transcript_has_nonzero_likelihood(likelihood_df: pd.DataFrame) -> None:
    """Check that every simulated fragment has nonzero likelihood under its true transcript."""
    bad_rows = []

    for _, row in likelihood_df.iterrows():
        true_t = row["true_transcript"]
        true_likelihood = row[f"{true_t}_likelihood"]

        if true_likelihood <= 0:
            bad_rows.append(row["fragment_id"])

    print("\nTruth-likelihood sanity check:")
    if len(bad_rows) == 0:
        print("  PASS: every fragment has nonzero likelihood under its true transcript.")
    else:
        print(f"  FAIL: {len(bad_rows)} fragments had zero likelihood under true transcript.")
        print("  Example bad fragment IDs:", bad_rows[:10])


def compute_posterior_matrix(
    likelihood_df: pd.DataFrame,
    transcripts: List[Transcript],
) -> pd.DataFrame:
    """
    Compute posterior probabilities P(t_k | r_i) using true abundances.

    Why we did this: mainly a diagnostic step. EM algorithm later replaces true
    abundances with iteratively estimated abundances.
    """
    abundances = {t.name: t.abundance for t in transcripts}
    rows = []

    for _, row in likelihood_df.iterrows():
        numerators = {}

        for transcript in transcripts:
            t_name = transcript.name
            likelihood = row[f"{t_name}_likelihood"]
            numerators[t_name] = likelihood * abundances[t_name]

        denominator = sum(numerators.values())

        out_row = {
            "fragment_id": row["fragment_id"],
            "true_transcript": row["true_transcript"],
        }

        for t_name in numerators:
            out_row[f"{t_name}_posterior"] = numerators[t_name] / denominator if denominator > 0 else 0.0

        rows.append(out_row)

    return pd.DataFrame(rows)


def summarize_posterior_matrix(posterior_df: pd.DataFrame, transcripts: List[Transcript]) -> None:
    """Print summary of posterior probabilities."""
    print("\nPosterior summary:")

    for transcript in transcripts:
        col = f"{transcript.name}_posterior"
        mean_val = posterior_df[col].mean()
        max_val = posterior_df[col].max()
        print(f"  {transcript.name}: mean={mean_val:.4f}, max={max_val:.4f}")

    display_cols = (
        ["fragment_id", "true_transcript"]
        + [f"{t.name}_posterior" for t in transcripts]
    )
    print("\nFirst 5 posterior rows:")
    print(posterior_df[display_cols].head())


def check_posterior_sums(posterior_df: pd.DataFrame, transcripts: List[Transcript]) -> None:
    """Check that posterior probabilities sum to 1 for each fragment."""
    cols = [f"{t.name}_posterior" for t in transcripts]
    sums = posterior_df[cols].sum(axis=1)

    if np.allclose(sums, 1.0):
        print("\nPosterior normalization check: PASS (all rows sum to 1)")
    else:
        print("\nPosterior normalization check: FAIL")


def run_em_algorithm(
    likelihood_df: pd.DataFrame,
    transcripts: List[Transcript],
    num_iterations: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Baseline EM:
        - initialize abundances uniformly
        - E-step: compute posteriors using current abundances
        - M-step: average posteriors to update abundances
    """
    transcript_names = [t.name for t in transcripts]
    mu = {t.name: 1.0 / len(transcripts) for t in transcripts}
    history = []

    for iteration in range(num_iterations):
        posteriors = []

        for _, row in likelihood_df.iterrows():
            numerators = {}
            for t_name in transcript_names:
                likelihood = row[f"{t_name}_likelihood"]
                numerators[t_name] = likelihood * mu[t_name]

            denom = sum(numerators.values())
            posterior_row = {}
            for t_name in transcript_names:
                posterior_row[t_name] = numerators[t_name] / denom if denom > 0 else 0.0

            posteriors.append(posterior_row)

        new_mu = {t_name: 0.0 for t_name in transcript_names}

        for posterior_row in posteriors:
            for t_name in transcript_names:
                new_mu[t_name] += posterior_row[t_name]

        number_of_fragments = len(posteriors)
        for t_name in new_mu:
            new_mu[t_name] /= number_of_fragments

        mu = new_mu
        history.append(mu.copy())

        if verbose:
            print(f"\nIteration {iteration + 1}:")
            for t_name in mu:
                print(f"  {t_name}: {mu[t_name]:.4f}")

    return mu, history

# adding new method for EM:

def convert_em_abundance_to_length_corrected(
    em_mu: Dict[str, float],
    transcripts: List[Transcript],
    likelihood_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Convert final EM estimates into length-corrected abundance estimates.

    Our group found that:
      - Do NOT keep dividing by effective length inside every EM iteration.
      - First run standard EM until convergence.
      - Then apply one final length correction after EM has converged.

    So we found that:
      EM gives an estimate proportional to expected fragment counts.
      Longer transcripts end up generating more fragments.
      To estimate transcript abundance/concentration, divide by effective length once.
    """
    transcript_lengths = {t.name: len(t.sequence) for t in transcripts}

    mean_fragment_length = likelihood_df["fragment_length"].mean()
    # checking to see mean lengths used:
    print(f"\nMean fragment length used for post-EM length correction: {mean_fragment_length:.2f}")
    print(
        "Fragment length range:",
        likelihood_df["fragment_length"].min(),
        "to",
        likelihood_df["fragment_length"].max(),
    )

    effective_lengths = {}
    for t_name in em_mu:
        eff_len = transcript_lengths[t_name] - mean_fragment_length + 1
        effective_lengths[t_name] = max(eff_len, 1.0)

    length_corrected_raw = {}
    for t_name in em_mu:
        length_corrected_raw[t_name] = em_mu[t_name] / effective_lengths[t_name]

    total = sum(length_corrected_raw.values())

    if total <= 0:
        raise ValueError("Length-corrected abundance total is zero.")

    length_corrected_mu = {
        t_name: val / total
        for t_name, val in length_corrected_raw.items()
    }

    return length_corrected_mu

# end method EM new one


def run_em_algorithm_length_adjusted(
    likelihood_df: pd.DataFrame,
    transcripts: List[Transcript],
    num_iterations: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float]]:
    """
    Length-adjusted EM.

    The E-step is the same to baseline EM. In M-step, expected counts are
    divided by effective transcript length before being renormalized.

    In old methods used before, this usually over-corrected and gave worse
    performance, which is still useful as a finding ffor our group.
    """
    transcript_names = [t.name for t in transcripts]
    transcript_lengths = {t.name: len(t.sequence) for t in transcripts}

    mean_fragment_length = likelihood_df["fragment_length"].mean()

    effective_lengths = {}
    for t_name in transcript_names:
        eff_len = transcript_lengths[t_name] - mean_fragment_length + 1
        effective_lengths[t_name] = max(eff_len, 1.0)

    if verbose:
        print("\nEffective transcript lengths:")
        for t_name in transcript_names:
            print(f"  {t_name}: {effective_lengths[t_name]:.2f}")

    mu = {t.name: 1.0 / len(transcripts) for t in transcripts}
    history = []

    for iteration in range(num_iterations):
        posteriors = []

        for _, row in likelihood_df.iterrows():
            numerators = {}
            for t_name in transcript_names:
                likelihood = row[f"{t_name}_likelihood"]
                numerators[t_name] = likelihood * mu[t_name]

            denom = sum(numerators.values())

            posterior_row = {}
            for t_name in transcript_names:
                posterior_row[t_name] = numerators[t_name] / denom if denom > 0 else 0.0

            posteriors.append(posterior_row)

        expected_counts = {t_name: 0.0 for t_name in transcript_names}
        for posterior_row in posteriors:
            for t_name in transcript_names:
                expected_counts[t_name] += posterior_row[t_name]

        theta = {}
        for t_name in transcript_names:
            theta[t_name] = expected_counts[t_name] / effective_lengths[t_name]

        theta_sum = sum(theta.values())

        new_mu = {}
        for t_name in transcript_names:
            new_mu[t_name] = theta[t_name] / theta_sum if theta_sum > 0 else 0.0

        mu = new_mu
        history.append(mu.copy())

        if verbose:
            print(f"\nLength-adjusted EM iteration {iteration + 1}:")
            for t_name in transcript_names:
                print(f"  {t_name}: {mu[t_name]:.4f}")

    return mu, history, effective_lengths
