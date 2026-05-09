"""Simulation helpers for toy and ENCODE-informed RNA-seq experiments."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List
import random

from .data_structures import SimulatedFragment, SimulatedRead, Transcript


def random_dna(length: int, rng: random.Random) -> str:
    """Generate a random DNA sequence of a given length."""
    alphabet = ["A", "C", "G", "T"]
    return "".join(rng.choice(alphabet) for _ in range(length))


def build_example_transcripts(
    rng: random.Random,
    exon_lengths: Dict[str, int] | None = None,
) -> List[Transcript]:
    """
    Build a toy four-isoform gene model.

    Isoforms:
        t00: exon 2 absent, exon 4 short
        t01: exon 2 absent, exon 4 long
        t10: exon 2 present, exon 4 short
        t11: exon 2 present, exon 4 long

    The shared exons make ambiguity, while optional/extended exons make
    transcript-identifying signal.
    """
    if exon_lengths is None:
        exon_lengths = {
            "E1": 120,
            "E2": 60,
            "E3": 70,
            "E4_short": 80,
            "E4_long_extra": 40,
        }

    e1 = random_dna(exon_lengths["E1"], rng)
    e2 = random_dna(exon_lengths["E2"], rng)
    e3 = random_dna(exon_lengths["E3"], rng)
    e4_short = random_dna(exon_lengths["E4_short"], rng)
    e4_long_extra = random_dna(exon_lengths["E4_long_extra"], rng)

    e4_long = e4_long_extra + e4_short

    transcripts = [
        Transcript("t00", e1 + e3 + e4_short, 0.20),
        Transcript("t01", e1 + e3 + e4_long, 0.30),
        Transcript("t10", e1 + e2 + e3 + e4_short, 0.25),
        Transcript("t11", e1 + e2 + e3 + e4_long, 0.25),
    ]

    total_abundance = sum(t.abundance for t in transcripts)
    if abs(total_abundance - 1.0) > 1e-9:
        raise ValueError("Transcript abundances must sum to 1.")

    return transcripts


def sample_transcript(transcripts: List[Transcript], rng: random.Random) -> Transcript:
    """Sample one transcript according to its abundance."""
    names = [t.name for t in transcripts]
    weights = [t.abundance for t in transcripts]
    chosen_name = rng.choices(names, weights=weights, k=1)[0]
    return next(t for t in transcripts if t.name == chosen_name)


def summarize_transcripts(transcripts: List[Transcript]) -> None:
    """Print transcript lengths and true abundances."""
    print("Transcript summary:")
    for t in transcripts:
        print(f"  {t.name}: length={len(t.sequence)}, abundance={t.abundance:.4f}")


def simulate_single_end_reads(
    transcripts: List[Transcript],
    num_reads: int,
    read_length: int,
    rng: random.Random,
) -> List[SimulatedRead]:
    """Simulate single-end reads from transcripts."""
    reads: List[SimulatedRead] = []

    for read_id in range(num_reads):
        transcript = sample_transcript(transcripts, rng)

        if len(transcript.sequence) < read_length:
            raise ValueError(f"Transcript {transcript.name} shorter than read length.")

        max_start = len(transcript.sequence) - read_length
        start_pos = rng.randint(0, max_start)
        seq = transcript.sequence[start_pos : start_pos + read_length]

        reads.append(
            SimulatedRead(
                read_id=read_id,
                sequence=seq,
                transcript_name=transcript.name,
                start_pos=start_pos,
            )
        )

    return reads


def summarize_single_end_reads(reads: List[SimulatedRead], transcripts: List[Transcript]) -> None:
    """Print a quick summary of simulated single-end reads."""
    counts = Counter(r.transcript_name for r in reads)

    print("\nObserved read counts by source transcript:")
    for t in transcripts:
        print(f"  {t.name}: {counts[t.name]}")

    print("\nFirst 5 single-end reads:")
    for r in reads[:5]:
        print(
            f"  read_id={r.read_id}, source={r.transcript_name}, "
            f"start={r.start_pos}, seq={r.sequence[:20]}..."
        )


def simulate_paired_end_fragments(
    transcripts: List[Transcript],
    num_fragments: int,
    read_length: int,
    min_fragment_length: int,
    max_fragment_length: int,
    rng: random.Random,
) -> List[SimulatedFragment]:
    """
    Simulate paired-end fragments.

    Steps:
        1. Pick transcript by abundance.
        2. Pick fragment length uniformly.
        3. Pick valid fragment start position.
        4. Leave out left read from fragment start.
        5. Leave out right read from fragment end.
    """
    fragments: List[SimulatedFragment] = []

    for fragment_id in range(num_fragments):
        transcript = sample_transcript(transcripts, rng)
        frag_len = rng.randint(min_fragment_length, max_fragment_length)

        if len(transcript.sequence) < frag_len:
            raise ValueError(f"Transcript {transcript.name} shorter than fragment length.")

        if frag_len < 2 * read_length:
            raise ValueError("Fragment length must be at least 2 * read_length.")

        max_start = len(transcript.sequence) - frag_len
        start_pos = rng.randint(0, max_start)

        fragment_seq = transcript.sequence[start_pos : start_pos + frag_len]
        left_read = fragment_seq[:read_length]
        right_read = fragment_seq[-read_length:]

        fragments.append(
            SimulatedFragment(
                fragment_id=fragment_id,
                left_read=left_read,
                right_read=right_read,
                transcript_name=transcript.name,
                start_pos=start_pos,
                fragment_length=frag_len,
            )
        )

    return fragments


def summarize_fragments(fragments: List[SimulatedFragment], transcripts: List[Transcript]) -> None:
    """Print fragment source counts and preview a few examples."""
    counts = Counter(f.transcript_name for f in fragments)

    print("\nObserved fragment counts by source transcript:")
    for t in transcripts:
        print(f"  {t.name}: {counts[t.name]}")

    print("\nFirst 5 paired-end fragments:")
    for f in fragments[:5]:
        print(
            f"  fragment_id={f.fragment_id}, source={f.transcript_name}, "
            f"start={f.start_pos}, frag_len={f.fragment_length}, "
            f"left={f.left_read[:12]}..., right={f.right_read[:12]}..."
        )
