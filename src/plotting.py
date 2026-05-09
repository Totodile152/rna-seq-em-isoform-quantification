"""Plotting helpers for transcript abundance experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _save_or_show(save_path: str | Path | None) -> None:
    """Save a figure if a path is provided; otherwise display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_abundance_comparison(
    true_mu: Dict[str, float],
    est_mu: Dict[str, float],
    title: str = "True vs Estimated Transcript Abundance",
    save_path: str | Path | None = None,
) -> None:
    """Bar plot comparing true and estimated transcript abundances."""
    names = list(true_mu.keys())
    true_vals = [true_mu[n] for n in names]
    est_vals = [est_mu[n] for n in names]

    x = np.arange(len(names))

    plt.figure(figsize=(max(8, len(names) * 0.45), 5))
    plt.bar(x - 0.2, true_vals, width=0.4)
    plt.bar(x + 0.2, est_vals, width=0.4)

    plt.xticks(x, names, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Transcript")
    plt.ylabel("Abundance")
    plt.legend(["True", "Estimated"])
    _save_or_show(save_path)


def plot_em_convergence(
    history: List[Dict[str, float]],
    title: str = "EM Convergence",
    save_path: str | Path | None = None,
) -> None:
    """Line plot showing abundance estimates across EM iterations."""
    if not history:
        raise ValueError("Cannot plot EM convergence: history is empty.")

    names = list(history[0].keys())
    iterations = range(1, len(history) + 1)

    plt.figure(figsize=(max(8, len(names) * 0.45), 5))

    for t_name in names:
        vals = [h[t_name] for h in history]
        plt.plot(iterations, vals, marker="o", label=t_name)

    plt.xlabel("Iteration")
    plt.ylabel("Abundance")
    plt.title(title)
    plt.legend()
    _save_or_show(save_path)


def plot_encode_diagnostics(encode_df_expr, encode_df_small, output_dir: str | Path) -> None:
    """Create the two diagnostic plots from ENCODE."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(encode_df_small)), encode_df_small["abundance_prob"].values)
    plt.xlabel("Transcript rank in selected subset")
    plt.ylabel("Abundance probability")
    plt.title("ENCODE-Informed Abundance Distribution (Top Selected Transcripts)")
    _save_or_show(output_dir / "encode_abundance_distribution.png")

    plt.figure(figsize=(7, 4))
    # RNA-seq TPM values are pretty skewed where:
    # many transcripts have low TPM, and a few have very high TPM.
    # Log-scaling x-axis makes the low/medium TPM range easier to read.
    tpm_values = encode_df_expr["TPM"]
    tpm_values = tpm_values[tpm_values > 0]

    plt.hist(tpm_values, bins=40)
    plt.xscale("log")
    plt.xlabel("TPM (log scale)")
    plt.ylabel("Count")
    plt.title("Distribution of Expressed Transcript TPM Values")
    _save_or_show(output_dir / "encode_tpm_distribution_log.png")

def plot_top_abundance_comparison(
    true_mu,
    est_mu,
    top_n=20,
    title="Top Isoforms: True vs Estimated",
    save_path=None,
):

    top_names = sorted(true_mu.keys(), key=lambda name: true_mu[name], reverse=True)[:top_n]

    true_vals = [true_mu[name] for name in top_names]
    est_vals = [est_mu.get(name, 0.0) for name in top_names]

    x = np.arange(len(top_names))

    plt.figure(figsize=(14, 6))
    plt.bar(x - 0.2, true_vals, width=0.4, label="True")
    plt.bar(x + 0.2, est_vals, width=0.4, label="Estimated")

    short_labels = [name.split(".")[0].replace("ENST", "") for name in top_names]
    plt.xticks(x, short_labels, rotation=60, ha="right")

    plt.title(title)
    plt.xlabel("Top transcript IDs")
    plt.ylabel("Abundance")
    plt.legend()
    plt.tight_layout()

    _save_or_show(save_path)

# Stretch goals: new additions our group added

def _abundance_dicts_to_arrays(true_mu, est_mu):
    import numpy as np

    transcript_names = list(true_mu.keys())
    true_values = np.array([true_mu[name] for name in transcript_names], dtype=float)
    est_values = np.array([est_mu.get(name, 0.0) for name in transcript_names], dtype=float)

    return transcript_names, true_values, est_values

def plot_true_vs_estimated_scatter(true_mu, est_mu, title, save_path):
    import matplotlib.pyplot as plt

    _, true_values, est_values = _abundance_dicts_to_arrays(true_mu, est_mu)

    max_val = max(true_values.max(), est_values.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(true_values, est_values, alpha=0.7)
    plt.plot([0, max_val], [0, max_val], linestyle="--", label="Ideal: y = x")
    plt.xlabel("True abundance")
    plt.ylabel("Estimated abundance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_absolute_error_comparison(true_mu, baseline_mu, length_corrected_mu, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    names, true_values, baseline_values = _abundance_dicts_to_arrays(true_mu, baseline_mu)
    _, _, length_values = _abundance_dicts_to_arrays(true_mu, length_corrected_mu)

    baseline_abs_error = np.abs(baseline_values - true_values)
    length_abs_error = np.abs(length_values - true_values)

    max_val = max(baseline_abs_error.max(), length_abs_error.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(baseline_abs_error, length_abs_error, alpha=0.7)
    plt.plot([0, max_val], [0, max_val], linestyle="--", label="Equal error")
    plt.xlabel("Baseline EM absolute error")
    plt.ylabel("Length-corrected EM absolute error")
    plt.title("Absolute Error Comparison: Baseline vs Length-Corrected EM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_vs_length(transcript_lengths, true_mu, baseline_mu, length_corrected_mu, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    names, true_values, baseline_values = _abundance_dicts_to_arrays(true_mu, baseline_mu)
    _, _, length_values = _abundance_dicts_to_arrays(true_mu, length_corrected_mu)

    lengths = np.array(transcript_lengths, dtype=float)

    baseline_error = baseline_values - true_values
    length_error = length_values - true_values

    plt.figure(figsize=(8, 6))
    plt.scatter(lengths, baseline_error, alpha=0.7, label="Baseline EM")
    plt.scatter(lengths, length_error, alpha=0.7, label="Length-corrected EM")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Transcript length")
    plt.ylabel("Estimated abundance - true abundance")
    plt.title("Estimation Error vs Transcript Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_compatibility_heatmap(
    compatibility_df,
    transcripts,
    save_path,
    max_fragments=80,
    max_transcripts=30,
    annotate=True,
):

    transcript_names = [t.name for t in transcripts[:max_transcripts]]
    placement_cols = [f"{name}_placements" for name in transcript_names]

    small_df = compatibility_df.head(max_fragments)

    matrix = small_df[placement_cols].to_numpy()
    binary_matrix = (matrix > 0).astype(int)

    plt.figure(figsize=(12, 7))

    # Better colormap
    im = plt.imshow(binary_matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Compatibility (0 = no, 1 = yes)")

    # Labels  and title
    plt.xlabel("Transcript ID")
    plt.ylabel("Fragment Index")
    plt.title("Fragment–Transcript Compatibility Heatmap")

    # Show SOME transcript labels (there's a lot of them so it crowds)
    xtick_positions = np.arange(0, len(transcript_names), max(1, len(transcript_names)//10))
    plt.xticks(
        ticks=xtick_positions,
        labels=[transcript_names[i] for i in xtick_positions],
        rotation=90,
        fontsize=7,
    )

    # Keep y ticks min
    plt.yticks(fontsize=7)

    # Annotations (small dots so it doesn't clutter for heatmap)
    if annotate:
        for i in range(binary_matrix.shape[0]):
            for j in range(binary_matrix.shape[1]):
                if binary_matrix[i, j] == 1:
                    plt.text(j, i, "•", ha="center", va="center", color="black", fontsize=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_distribution(true_mu, baseline_mu, length_corrected_mu, save_path):

    def dict_to_array(d):
        return np.array(list(d.values()), dtype=float)

    true_vals = dict_to_array(true_mu)
    baseline_vals = dict_to_array(baseline_mu)
    length_vals = dict_to_array(length_corrected_mu)

    baseline_error = np.abs(baseline_vals - true_vals)
    length_error = np.abs(length_vals - true_vals)

    plt.figure(figsize=(8, 6))

    plt.hist(
        baseline_error,
        bins=30,
        alpha=0.6,
        label="Baseline EM",
    )

    plt.hist(
        length_error,
        bins=30,
        alpha=0.6,
        label="Length-corrected EM",
    )

    baseline_error = np.log10(baseline_error + 1e-8)
    length_error = np.log10(length_error + 1e-8)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Absolute Errors Across Transcripts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()