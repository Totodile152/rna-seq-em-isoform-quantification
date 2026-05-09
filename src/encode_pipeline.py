"""ENCODE-informed preprocessing and simulation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List
import random

import numpy as np
import pandas as pd

from .compatibility import build_paired_end_compatibility_table, summarize_paired_end_compatibility
from .data_structures import Transcript
from .evaluation import compute_metrics, get_true_abundances, get_ambiguity_stats
from .inference import (
    check_posterior_sums,
    check_true_transcript_has_nonzero_likelihood,
    compute_likelihood_matrix,
    compute_posterior_matrix,
    convert_em_abundance_to_length_corrected,
    run_em_algorithm,
    run_em_algorithm_length_adjusted,
    summarize_likelihood_matrix,
    summarize_posterior_matrix,
)
from .plotting import (
    plot_abundance_comparison, 
    plot_em_convergence, 
    plot_encode_diagnostics, 
    plot_top_abundance_comparison,
    plot_true_vs_estimated_scatter,
    plot_error_vs_length,
    plot_absolute_error_comparison,
    plot_compatibility_heatmap,
    plot_error_distribution,
)
from .simulation import random_dna, simulate_paired_end_fragments, summarize_fragments, summarize_transcripts


REQUIRED_ENCODE_COLUMNS = [
    "transcript_id(s)",
    "length",
    "effective_length",
    "expected_count",
    "TPM",
    "FPKM",
]


def load_encode_quantification(path: str | Path) -> pd.DataFrame:
    """Load the ENCODE transcript quantification TSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find ENCODE TSV at {path}. "
            "Put the TSV in data/raw/ or pass --encode-tsv /path/to/file.tsv."
        )

    encode_df_raw = pd.read_csv(path, sep="\t")

    missing_columns = [col for col in REQUIRED_ENCODE_COLUMNS if col not in encode_df_raw.columns]
    if missing_columns:
        raise ValueError(f"ENCODE TSV is missing expected columns: {missing_columns}")

    print("Shape of raw ENCODE table:", encode_df_raw.shape)
    print("Columns:", list(encode_df_raw.columns))

    return encode_df_raw


def prepare_encode_subset(
    encode_df_raw: pd.DataFrame,
    tpm_threshold: float = 1.0,
    top_n: int = 20,
    max_fragment_length: int = 140,
    single_transcript_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean ENCODE quantification output and pick a small sim-ready subset.

    Returns:
        encode_df_expr: all expressed transcripts after TPM filtering.
        encode_df_ready: top-N, single-transcript, length-compatible subset.
    """
    encode_df = encode_df_raw[REQUIRED_ENCODE_COLUMNS].copy()

    encode_df = encode_df.rename(
        {
            "transcript_id(s)": "transcript_id",
            "length": "transcript_length",
            "effective_length": "effective_length",
            "expected_count": "expected_count",
            "TPM": "TPM",
            "FPKM": "FPKM",
        },
        axis=1
    )

    encode_df_expr = encode_df[encode_df["TPM"] > tpm_threshold].copy()
    encode_df_expr["abundance_prob"] = encode_df_expr["TPM"] / encode_df_expr["TPM"].sum()

    print(f"Number of expressed transcripts (TPM > {tpm_threshold}): {len(encode_df_expr)}")
    print("\nBasic TPM summary after filtering:")
    print(encode_df_expr["TPM"].describe())

    if single_transcript_only:
        encode_df_expr = encode_df_expr[
            ~encode_df_expr["transcript_id"].astype(str).str.contains(",", regex=False)
        ].copy()

    encode_df_expr["abundance_prob"] = encode_df_expr["TPM"] / encode_df_expr["TPM"].sum()

    encode_df_small = (
        encode_df_expr.sort_values("TPM", ascending=False)
        .head(top_n)
        .copy()
    )
    encode_df_small["abundance_prob"] = (
        encode_df_small["abundance_prob"] / encode_df_small["abundance_prob"].sum()
    )

    encode_df_ready = encode_df_small[
        encode_df_small["transcript_length"] >= max_fragment_length
    ].copy()
    encode_df_ready["abundance_prob"] = (
        encode_df_ready["abundance_prob"] / encode_df_ready["abundance_prob"].sum()
    )

    print("\nSelected clean subset size:", len(encode_df_ready))
    print("Subset abundance sum:", encode_df_ready["abundance_prob"].sum())
    print(
        encode_df_ready[
            ["transcript_id", "transcript_length", "TPM", "abundance_prob"]
        ]
    )

    return encode_df_expr, encode_df_ready

# New change to fix ambiguity issue from midterm presentation:

def _generate_isoform_exon_subsets(
    n_exons: int,
    k: int,
    rng: random.Random,
    middle_inclusion_prob: float = 0.6,
) -> List[List[int]]:
    """
    Make k distinct ordered exon-index subsets from a shared exon library.

    We always include both:
      - first exon
      - last exon

    Then we randomly include/exclude middle exons.

    This makes isoforms from the same synthetic gene that share some exons
    but differ in others, which ends up making realistic ambiguity.
    """
    middle_count = n_exons - 2

    if middle_count < 1:
        raise ValueError("Need at least 3 exons to create alternative isoforms.")

    distinct_patterns = (2 ** middle_count) - 1
    if k > distinct_patterns:
        raise ValueError(
            f"Cannot make {k} distinct isoforms from {middle_count} optional exons."
        )

    seen_masks = set()
    subsets: List[List[int]] = []

    max_attempts = max(k * 200, 1000)
    attempts = 0

    while len(subsets) < k:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError("Failed to generate enough distinct isoform patterns.")

        mask = tuple(
            rng.random() < middle_inclusion_prob
            for _ in range(middle_count)
        )

        # Skip isoforms that include no middle exons.
        # This prevents every transcript from just being first + last exon.
        if not any(mask):
            continue

        if mask in seen_masks:
            continue

        seen_masks.add(mask)

        indices = [0]
        indices += [i + 1 for i, included in enumerate(mask) if included]
        indices += [n_exons - 1]

        subsets.append(indices)

    return subsets

# end of new method

# putting under new helper method:

def build_encode_isoform_transcripts(
    encode_df_raw: pd.DataFrame,
    rng: random.Random,
    num_genes: int = 30,
    min_isoforms_per_gene: int = 2,
    max_isoforms_per_gene: int = 5,
    min_tpm: float = 1.0,
    exon_length_min: int = 120,
    exon_length_max: int = 400,
    extra_exons: int = 2,
    dirichlet_alpha: float = 0.5,
    min_transcript_length: int = 200,
    middle_inclusion_prob: float = 0.6,
) -> List[Transcript]:
    """
    Make ENCODE-informed synthetic isoforms with shared exon structure.

    Thought-process on why this replaces build_encode_informed_transcripts:
    - Old function treated each ENCODE row as an independent transcript.
    - Independent random transcript sequences made 0 ambiguous fragments. (no good)
    - Real isoform ambiguity comes from isoforms of same gene sharing exons.

    This function works since:
    1. samples expressed multi-isoform rows from ENCODE/RSEM output,
    2. treats each row as a gene-like thing with multiple listed isoforms,
    3. makes a shared synthetic exon library for that gene,
    4. builds each isoform as an ordered subset of those shared exons,
    5. splits the gene TPM across its isoforms using a Dirichlet distribution,
    6. normalizes all transcript abundances globally.
    """
    df = encode_df_raw.copy()

    # Keep expressed rows.
    df = df[df["TPM"] >= min_tpm].copy()

    # Drop spike-ins if there are an.
    df = df[~df["gene_id"].astype(str).str.startswith("gSpikein")].copy()

    # Keep rows with multiple listed transcript IDs.
    # These are the rows that can act like multi-isoform gene units.
    df = df[df["transcript_id(s)"].astype(str).str.contains(",", regex=False)].copy()

    # Count how many transcript IDs are listed in each row.
    df["_n_isoforms"] = df["transcript_id(s)"].astype(str).str.count(",") + 1
    df = df[df["_n_isoforms"] >= min_isoforms_per_gene].copy()

    if len(df) == 0:
        raise ValueError("No expressed multi-isoform rows found in the ENCODE table.")

    if len(df) < num_genes:
        raise ValueError(
            f"Requested num_genes={num_genes}, but only found {len(df)} usable rows. "
            "Reduce num_genes or min_tpm."
        )

    # NumPy RNG used for weighted gene sampling and Dirichlet splits.
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    gene_indices = df.index.to_numpy()
    gene_weights = df["TPM"].to_numpy()
    gene_weights = gene_weights / gene_weights.sum()

    chosen_idx = np_rng.choice(
        gene_indices,
        size=num_genes,
        replace=False,
        p=gene_weights,
    )

    chosen = df.loc[chosen_idx].reset_index(drop=True)

    candidates = []

    for _, row in chosen.iterrows():
        gene_tpm = float(row["TPM"])

        isoform_ids = [
            tid.strip()
            for tid in str(row["transcript_id(s)"]).split(",")
            if tid.strip()
        ]

        if len(isoform_ids) > max_isoforms_per_gene:
            isoform_ids = isoform_ids[:max_isoforms_per_gene]

        k = len(isoform_ids)

        # More isoforms need more possible exon inclusion patterns.
        n_exons = k + extra_exons

        exon_lengths = [
            rng.randint(exon_length_min, exon_length_max)
            for _ in range(n_exons)
        ]

        exon_sequences = [
            random_dna(length, rng)
            for length in exon_lengths
        ]

        subsets = _generate_isoform_exon_subsets(
            n_exons=n_exons,
            k=k,
            rng=rng,
            middle_inclusion_prob=middle_inclusion_prob,
        )

        # Split gene-level TPM across isoforms.
        isoform_weights = np_rng.dirichlet([dirichlet_alpha] * k)

        for transcript_id, subset, weight in zip(isoform_ids, subsets, isoform_weights):
            sequence = "".join(exon_sequences[i] for i in subset)

            if len(sequence) < min_transcript_length:
                continue

            raw_abundance = gene_tpm * float(weight)

            candidates.append(
                (transcript_id, sequence, raw_abundance)
            )

    if not candidates:
        raise RuntimeError(
            "No valid isoform transcripts were generated. "
            "Try lowering min_transcript_length or increasing exon lengths."
        )

    total_raw_abundance = sum(raw for _, _, raw in candidates)

    transcripts = [
        Transcript(
            name=transcript_id,
            sequence=sequence,
            abundance=raw_abundance / total_raw_abundance,
        )
        for transcript_id, sequence, raw_abundance in candidates
    ]

    if not np.isclose(sum(t.abundance for t in transcripts), 1.0):
        raise ValueError("Generated transcript abundances do not sum to 1.")

    return transcripts

# end of helpher method 2

def build_encode_informed_transcripts(
    encode_subset_df: pd.DataFrame,
    rng: random.Random,
) -> List[Transcript]:
    """
    Build Transcript objects using:
        - real transcript IDs from ENCODE
        - real transcript lengths from ENCODE
        - real abundance probabilities from ENCODE
        - synthetic DNA sequences generated to match those lengths
    """
    transcripts = []

    for _, row in encode_subset_df.iterrows():
        transcript_id = str(row["transcript_id"])
        transcript_length = int(round(row["transcript_length"]))
        abundance = float(row["abundance_prob"])

        sequence = random_dna(transcript_length, rng)

        transcripts.append(
            Transcript(
                name=transcript_id,
                sequence=sequence,
                abundance=abundance,
            )
        )

    total_abundance = sum(t.abundance for t in transcripts)
    if not np.isclose(total_abundance, 1.0):
        raise ValueError("ENCODE-informed transcript abundances do not sum to 1.")

    return transcripts


def run_encode_pipeline(
    encode_tsv_path: str | Path,
    output_dir: str | Path = "outputs/figures",
    seed: int = 123,
    tpm_threshold: float = 1.0,
    top_n: int = 20,
    num_fragments: int = 500,
    read_length: int = 30,
    min_fragment_length: int = 80,
    max_fragment_length: int = 140,
    num_iterations: int = 10,
):
    """Run the paired-end EM pipeline on ENCODE-informed transcripts."""
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    encode_df_raw = load_encode_quantification(encode_tsv_path)
    encode_df_expr, encode_df_ready = prepare_encode_subset(
        encode_df_raw=encode_df_raw,
        tpm_threshold=tpm_threshold,
        top_n=top_n,
        max_fragment_length=max_fragment_length,
    )

    plot_encode_diagnostics(encode_df_expr, encode_df_ready, output_dir)

    # commenting this out to replace it with new helper
    # keeping here in case we wanna compare old vs new method
    
    #encode_transcripts = build_encode_informed_transcripts(
    #    encode_subset_df=encode_df_ready,
    #    rng=rng,
    #)

    #print("\nNumber of ENCODE-informed transcript objects:", len(encode_transcripts))
    #summarize_transcripts(encode_transcripts)

    # new encode_transcripts below for helper methods:

    encode_transcripts = build_encode_isoform_transcripts(
        encode_df_raw=encode_df_raw,
        rng=rng,
        num_genes=30,
        min_isoforms_per_gene=2,
        max_isoforms_per_gene=5,
        min_tpm=tpm_threshold,
        exon_length_min=120,
        exon_length_max=400,
        extra_exons=2,
        dirichlet_alpha=0.5,
        min_transcript_length=max_fragment_length,
        middle_inclusion_prob=0.6,
    )

    print("\nNumber of ENCODE-informed isoform transcript objects:", len(encode_transcripts))
    summarize_transcripts(encode_transcripts)

    # end of new addition

    encode_fragments = simulate_paired_end_fragments(
        transcripts=encode_transcripts,
        num_fragments=num_fragments,
        read_length=read_length,
        min_fragment_length=min_fragment_length,
        max_fragment_length=max_fragment_length,
        rng=rng,
    )

    summarize_fragments(encode_fragments, encode_transcripts)

    encode_compatibility_df = build_paired_end_compatibility_table(
        fragments=encode_fragments,
        transcripts=encode_transcripts,
        read_length=read_length,
    )

    summarize_paired_end_compatibility(encode_compatibility_df, encode_transcripts)

    # new additions
    encode_ambiguity = get_ambiguity_stats(encode_compatibility_df, encode_transcripts)
    print("\nENCODE isoform ambiguity stats:", encode_ambiguity)

    encode_likelihood_df = compute_likelihood_matrix(
        compatibility_df=encode_compatibility_df,
        transcripts=encode_transcripts,
    )

    summarize_likelihood_matrix(encode_likelihood_df, encode_transcripts)
    check_true_transcript_has_nonzero_likelihood(encode_likelihood_df)

    encode_posterior_df = compute_posterior_matrix(
        likelihood_df=encode_likelihood_df,
        transcripts=encode_transcripts,
    )

    summarize_posterior_matrix(encode_posterior_df, encode_transcripts)
    check_posterior_sums(encode_posterior_df, encode_transcripts)

    print("\nRunning baseline EM on ENCODE-informed data...")
    encode_final_mu, encode_history = run_em_algorithm(
        likelihood_df=encode_likelihood_df,
        transcripts=encode_transcripts,
        num_iterations=num_iterations,
    )

    encode_true_mu = get_true_abundances(encode_transcripts)

    print("\nBaseline EM evaluation on ENCODE-informed data:")
    baseline_metrics = compute_metrics(encode_true_mu, encode_final_mu)

    print("\nRunning length-adjusted EM on ENCODE-informed data...")
    encode_final_mu_len_adj, encode_history_len_adj, encode_effective_lengths = run_em_algorithm_length_adjusted(
        likelihood_df=encode_likelihood_df,
        transcripts=encode_transcripts,
        num_iterations=num_iterations,
    )

    # adding new one for encode adjusted len

    encode_final_mu_length_corrected = convert_em_abundance_to_length_corrected(
        em_mu=encode_final_mu,
        transcripts=encode_transcripts,
        likelihood_df=encode_likelihood_df,
    )

    # old one we decided not to use
    #print("\nCorrected length-normalized evaluation:")

    #compute_metrics(
    #    true_mu=encode_true_mu,
    #    est_mu=encode_final_mu_length_corrected,
    #)

    print("\nCorrected length-normalized evaluation:")

    # new additon to see how ti works out in comparision to old method
    length_corrected_metrics = compute_metrics(
        true_mu=encode_true_mu,
        est_mu=encode_final_mu_length_corrected,
    )

    # end adjusted len addition

    print("\nLength-adjusted EM evaluation on ENCODE-informed data:")
    length_adjusted_metrics = compute_metrics(encode_true_mu, encode_final_mu_len_adj)

    print("\n=== ENCODE-INFORMED COMPARISON ===")
    print("Baseline EM:", baseline_metrics)
    print("Length-adjusted EM:", length_adjusted_metrics)

    plot_abundance_comparison(
        encode_true_mu,
        encode_final_mu,
        title="ENCODE-Informed: True vs Baseline EM",
        save_path=output_dir / "encode_baseline_abundance.png",
    )
    plot_em_convergence(
        encode_history,
        title="ENCODE-Informed Baseline EM Convergence",
        save_path=output_dir / "encode_baseline_convergence.png",
    )
    plot_abundance_comparison(
        encode_true_mu,
        encode_final_mu_len_adj,
        title="ENCODE-Informed: (Incorrect) True vs Length-Adjusted EM",
        save_path=output_dir / "encode_length_adjusted_abundance.png",
    )
    plot_em_convergence(
        encode_history_len_adj,
        title="ENCODE-Informed Length-Adjusted EM Convergence",
        save_path=output_dir / "encode_length_adjusted_convergence.png",
    )
    plot_top_abundance_comparison(
        true_mu=encode_true_mu,
        est_mu=encode_final_mu,
        top_n=20,
        title="ENCODE-Informed: Top 20 True vs Baseline EM",
        save_path=output_dir / "encode_top20_baseline_em.png",
    )
    plot_top_abundance_comparison(
        true_mu=encode_true_mu,
        est_mu=encode_final_mu_length_corrected,
        top_n=20,
        title="ENCODE-Informed: Top 20 True vs Final Length-Corrected Abundance",
        save_path=output_dir / "encode_top20_length_corrected.png",
    )
    
    # new stretch goals:

    transcript_lengths = [len(t.sequence) for t in encode_transcripts]

    plot_true_vs_estimated_scatter(
        encode_true_mu,
        encode_final_mu,
        "ENCODE-Informed: True vs Baseline EM Estimated Abundance",
        output_dir / "encode_true_vs_baseline_scatter.png",
    )

    plot_true_vs_estimated_scatter(
        encode_true_mu,
        encode_final_mu_length_corrected,
        "ENCODE-Informed: True vs Final Length-Corrected Abundance",
        output_dir / "encode_true_vs_length_corrected_scatter.png",
    )

    plot_error_vs_length(
        transcript_lengths,
        encode_true_mu,
        encode_final_mu,
        encode_final_mu_length_corrected,
        output_dir / "encode_error_vs_transcript_length.png",
    )

    plot_absolute_error_comparison(
        encode_true_mu,
        encode_final_mu,
        encode_final_mu_length_corrected,
        output_dir / "encode_absolute_error_comparison.png",
    )

    plot_compatibility_heatmap(
        compatibility_df=encode_compatibility_df,
        transcripts=encode_transcripts,
        save_path=output_dir / "encode_compatibility_heatmap.png",
    )

    plot_error_distribution(
        encode_true_mu,
        encode_final_mu,
        encode_final_mu_length_corrected,
        output_dir / "encode_error_distribution.png",
    )

    return {
        "encode_df_ready": encode_df_ready,
        "transcripts": encode_transcripts,
        "fragments": encode_fragments,
        "compatibility_df": encode_compatibility_df,
        "likelihood_df": encode_likelihood_df,
        "posterior_df": encode_posterior_df,

        "baseline_mu": encode_final_mu,
        "baseline_history": encode_history,
        "baseline_metrics": baseline_metrics,

        "length_mu": encode_final_mu_len_adj,

        # adding corrected ones
        "length_corrected_mu": encode_final_mu_length_corrected,
        "length_corrected_metrics": length_corrected_metrics,

        "length_history": encode_history_len_adj,
        "length_metrics": length_adjusted_metrics,
        "effective_lengths": encode_effective_lengths,
    }
