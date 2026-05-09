"""Core data containers for the RNA-seq isoform abundance project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Transcript:
    """
    Represents one transcript / isoform.

    Attributes:
        name: Transcript label, such as t00, t01, or an ENCODE transcript ID.
        sequence: Full nucleotide sequence of the transcript.
        abundance: True abundance used in sim. Should sum to 1 across transcripts.
    """

    name: str
    sequence: str
    abundance: float


@dataclass
class SimulatedRead:
    """
    Represents a simulated single-end read.

    Attributes:
        read_id: Integer read ID.
        sequence: Read nucleotide sequence.
        transcript_name: True transcript source.
        start_pos: True start position within source transcript.
    """

    read_id: int
    sequence: str
    transcript_name: str
    start_pos: int


@dataclass
class SimulatedFragment:
    """
    Represents a simulated paired-end fragment.

    Attributes:
        fragment_id: Integer fragment ID.
        left_read: Left-end read sequence.
        right_read: Right-end read sequence.
        transcript_name: True transcript source.
        start_pos: True fragment start position.
        fragment_length: Full fragment length.
    """

    fragment_id: int
    left_read: str
    right_read: str
    transcript_name: str
    start_pos: int
    fragment_length: int
