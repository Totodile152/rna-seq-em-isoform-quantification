"""Read/fragment compatibility and placement-count helpers."""

from __future__ import annotations

from typing import List

import pandas as pd

from .data_structures import SimulatedFragment, SimulatedRead, Transcript


def count_valid_placements(read_seq: str, transcript_seq: str) -> int:
    """Count how many exact matches of a read occur in a transcript."""
    read_len = len(read_seq)
    transcript_len = len(transcript_seq)

    if read_len > transcript_len:
        return 0

    count = 0
    for start in range(transcript_len - read_len + 1):
        if transcript_seq[start : start + read_len] == read_seq:
            count += 1

    return count


def build_single_end_compatibility_table(
    reads: List[SimulatedRead],
    transcripts: List[Transcript],
) -> pd.DataFrame:
    """Build a compatibility table for single-end reads."""
    rows = []

    for read in reads:
        row = {
            "read_id": read.read_id,
            "read_sequence": read.sequence,
            "true_transcript": read.transcript_name,
        }

        for transcript in transcripts:
            placements = count_valid_placements(read.sequence, transcript.sequence)
            row[f"{transcript.name}_placements"] = placements
            row[f"{transcript.name}_compatible"] = int(placements > 0)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_single_end_compatibility(
    compatibility_df: pd.DataFrame,
    transcripts: List[Transcript],
) -> None:
    """Summarize single-end read compatibility."""
    print("\nSingle-end compatibility summary:")

    for transcript in transcripts:
        compatible_count = compatibility_df[f"{transcript.name}_compatible"].sum()
        print(f"  Reads compatible with {transcript.name}: {compatible_count}")

    compatible_cols = [f"{t.name}_compatible" for t in transcripts]
    compatibility_df["num_compatible_transcripts"] = compatibility_df[compatible_cols].sum(axis=1)

    unique_reads = (compatibility_df["num_compatible_transcripts"] == 1).sum()
    ambiguous_reads = (compatibility_df["num_compatible_transcripts"] > 1).sum()
    incompatible_reads = (compatibility_df["num_compatible_transcripts"] == 0).sum()

    print("\nRead ambiguity summary:")
    print(f"  Uniquely compatible reads: {unique_reads}")
    print(f"  Ambiguous reads:           {ambiguous_reads}")
    print(f"  Incompatible reads:        {incompatible_reads}")


def count_valid_fragment_placements(
    left_read: str,
    right_read: str,
    transcript_seq: str,
    fragment_length: int,
    read_length: int,
) -> int:
    """
    Count how many valid paired-end fragment placements occur in a transcript.

    A valid placement has to match the left read at the beginning of the candidate
    fragment, match the right read at the end, and allso respect the exact fragment
    length.
    """
    transcript_len = len(transcript_seq)

    if fragment_length > transcript_len:
        return 0

    count = 0
    for start in range(transcript_len - fragment_length + 1):
        fragment_seq = transcript_seq[start : start + fragment_length]
        candidate_left = fragment_seq[:read_length]
        candidate_right = fragment_seq[-read_length:]

        if candidate_left == left_read and candidate_right == right_read:
            count += 1

    return count


def build_paired_end_compatibility_table(
    fragments: List[SimulatedFragment],
    transcripts: List[Transcript],
    read_length: int,
) -> pd.DataFrame:
    """Build a fragment-transcript compatibility table."""
    rows = []

    for frag in fragments:
        row = {
            "fragment_id": frag.fragment_id,
            "true_transcript": frag.transcript_name,
            "fragment_length": frag.fragment_length,
            "left_read": frag.left_read,
            "right_read": frag.right_read,
        }

        for transcript in transcripts:
            placements = count_valid_fragment_placements(
                left_read=frag.left_read,
                right_read=frag.right_read,
                transcript_seq=transcript.sequence,
                fragment_length=frag.fragment_length,
                read_length=read_length,
            )
            row[f"{transcript.name}_placements"] = placements
            row[f"{transcript.name}_compatible"] = int(placements > 0)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_paired_end_compatibility(
    compatibility_df: pd.DataFrame,
    transcripts: List[Transcript],
) -> None:
    """Summarizez paired-end fragment ambiguity."""
    print("\nPaired-end compatibility summary:")

    for transcript in transcripts:
        compatible_count = compatibility_df[f"{transcript.name}_compatible"].sum()
        print(f"  Fragments compatible with {transcript.name}: {compatible_count}")

    compatible_cols = [f"{t.name}_compatible" for t in transcripts]
    compatibility_df["num_compatible_transcripts"] = compatibility_df[compatible_cols].sum(axis=1)

    unique_frags = (compatibility_df["num_compatible_transcripts"] == 1).sum()
    ambiguous_frags = (compatibility_df["num_compatible_transcripts"] > 1).sum()
    incompatible_frags = (compatibility_df["num_compatible_transcripts"] == 0).sum()

    print("\nFragment ambiguity summary:")
    print(f"  Uniquely compatible fragments: {unique_frags}")
    print(f"  Ambiguous fragments:           {ambiguous_frags}")
    print(f"  Incompatible fragments:        {incompatible_frags}")

    display_cols = (
        ["fragment_id", "true_transcript", "fragment_length", "num_compatible_transcripts"]
        + [f"{t.name}_placements" for t in transcripts]
    )
    print("\nFirst 5 paired-end compatibility rows:")
    print(compatibility_df[display_cols].head())
