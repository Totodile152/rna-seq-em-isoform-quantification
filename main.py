"""Running commands for RNA-seq isoform project."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from src.compatibility import build_paired_end_compatibility_table, summarize_paired_end_compatibility
from src.encode_pipeline import run_encode_pipeline
from src.evaluation import compute_metrics, get_true_abundances
from src.experiments import run_ambiguity_experiments
from src.inference import (
    check_posterior_sums,
    check_true_transcript_has_nonzero_likelihood,
    compute_likelihood_matrix,
    compute_posterior_matrix,
    run_em_algorithm,
    run_em_algorithm_length_adjusted,
    summarize_likelihood_matrix,
    summarize_posterior_matrix,
)
from src.plotting import plot_abundance_comparison, plot_em_convergence
from src.simulation import (
    build_example_transcripts,
    simulate_paired_end_fragments,
    summarize_fragments,
    summarize_transcripts,
)


def run_toy_demo(output_dir: Path) -> None:
    """Run the main toy 4-isoform paired-end EM demo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    transcripts = build_example_transcripts(rng)
    summarize_transcripts(transcripts)

    fragments = simulate_paired_end_fragments(
        transcripts=transcripts,
        num_fragments=200,
        read_length=30,
        min_fragment_length=80,
        max_fragment_length=140,
        rng=rng,
    )
    summarize_fragments(fragments, transcripts)

    compatibility_df = build_paired_end_compatibility_table(
        fragments=fragments,
        transcripts=transcripts,
        read_length=30,
    )
    summarize_paired_end_compatibility(compatibility_df, transcripts)

    likelihood_df = compute_likelihood_matrix(
        compatibility_df=compatibility_df,
        transcripts=transcripts,
    )
    summarize_likelihood_matrix(likelihood_df, transcripts)
    check_true_transcript_has_nonzero_likelihood(likelihood_df)

    posterior_df = compute_posterior_matrix(
        likelihood_df=likelihood_df,
        transcripts=transcripts,
    )
    summarize_posterior_matrix(posterior_df, transcripts)
    check_posterior_sums(posterior_df, transcripts)

    print("\nRunning baseline EM...")
    final_mu, history = run_em_algorithm(
        likelihood_df=likelihood_df,
        transcripts=transcripts,
        num_iterations=10,
    )

    true_mu = get_true_abundances(transcripts)

    print("\nBaseline evaluation:")
    compute_metrics(true_mu, final_mu)

    plot_abundance_comparison(
        true_mu,
        final_mu,
        title="Toy Model: True vs Baseline EM",
        save_path=output_dir / "toy_baseline_abundance.png",
    )
    plot_em_convergence(
        history,
        title="Toy Model Baseline EM Convergence",
        save_path=output_dir / "toy_baseline_convergence.png",
    )

    print("\nRunning length-adjusted EM...")
    final_mu_len_adj, history_len_adj, effective_lengths = run_em_algorithm_length_adjusted(
        likelihood_df=likelihood_df,
        transcripts=transcripts,
        num_iterations=10,
    )

    print("\nLength-adjusted evaluation:")
    compute_metrics(true_mu, final_mu_len_adj)

    plot_abundance_comparison(
        true_mu,
        final_mu_len_adj,
        title="Toy Model: True vs Length-Adjusted EM",
        save_path=output_dir / "toy_length_adjusted_abundance.png",
    )
    plot_em_convergence(
        history_len_adj,
        title="Toy Model Length-Adjusted EM Convergence",
        save_path=output_dir / "toy_length_adjusted_convergence.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run toy and/or ENCODE-informed RNA-seq isoform abundance experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["toy", "ambiguity", "encode", "all"],
        default="toy",
        help="Which part of the project to run.",
    )
    parser.add_argument(
        "--encode-tsv",
        type=str,
        default=None,
        help="Path to ENCODE transcript quantification TSV. Required for --mode encode or all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory where figures should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.mode in {"toy", "all"}:
        run_toy_demo(output_dir)

    if args.mode in {"ambiguity", "all"}:
        run_ambiguity_experiments(verbose=False)

    if args.mode in {"encode", "all"}:
        if args.encode_tsv is None:
            raise SystemExit(
                "You selected ENCODE mode but did not provide --encode-tsv. "
                "Example: python main.py --mode encode --encode-tsv data/raw/ENCFF867KVU.tsv"
            )
        run_encode_pipeline(
            encode_tsv_path=args.encode_tsv,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
