# RNA-seq Isoform Abundance EM Project

This is the code for our group project where we're looking at
probabilistic assignment of RNA-seq fragments for isoform
quantification.

Our goal is to try and estimate transcript/isoform abundances
when paired-end fragments are ambiguous and can match with
multiple transcripts.

We use an EM algorithm to assign fragments probabilitically
rather than hard-counting them.
- Lets us model ambiguity more realistically when frags can match multiple isoforms

## Project goal

RNA-seq fragments are short, and we know that isoforms from
the same gene usually share exons. Thus, one fragment might
still match with multiple transcript isoforms. That makes
counting more unreliable by hand.

The project has two main parts:

1. **Toy 4-isoform simulation**
   - Makes four synthetic transcript isoforms using two binary splicing choices.
   - Simulates paired-end fragments.
   - Makes a fragment-transcript compatibility table.
   - Finds likelihoods, posteriors, and EM estimates.
   - Compares estimated abundances to known ground truth.

2. **ENCODE-informed simulation**
   - Loads transcript-level quantification data from an ENCODE TSV.
   - Uses real transcript IDs, transcript lengths, and TPM-derived abundance probabilities.
   - Makes synthetic transcript sequences with those real-data-informed properties.
   - Runs paired-end EM pipeline.
   - Later on we ended up improving ENCODE sim by making
     shared exon libraries across isoforms of same gene
     sharing exon seq. (explains why ambig was 0 in prev one)

## Main idea of it is:

We find this for each fragment:
- which trascripts it's comptabile with
- likelihood of fragment under each transcript
- posterior prob that each transcript made the fragment

Then, EM alternates between:
- E-step: soft-assigns fragments to transcripts
- M-step: updates transcript abundance estimates

## Folder structure

```text
bio-project/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
├── outputs/
│   └── figures/
└── src/
    ├── data_structures.py
    ├── simulation.py
    ├── compatibility.py
    ├── inference.py
    ├── evaluation.py
    ├── plotting.py
    ├── experiments.py
    └── encode_pipeline.py
```
## What each source file does:

- `simulation.py`
  Makes synthetic transcripts and paired-end fragments.

- `compatibility.py`
  Makes fragment-transcript compatibility tables and checks ambiguity.

- `inference.py`
  Finds likelihoods, posterior probabilities, and EM abundance estimates.

- `encode_pipeline.py`
  Runs ENCODE-informed simulation pipeline and evaluation.

- `plotting.py`
  Makes plots/figures used in report and slides.

- `evaluation.py`
  Finds metrics like MSE and correlation against ground truth.

## Data Used:

The ENCODE TSV used for this project was ENCFF867KVU.tsv

Our group gor this transcript quantification table
from ENCODE and used it to make ENCODE-informed transcript
abundance distributions.

Link: https://www.encodeproject.org/files/ENCFF867KVU/

## System Requirements

- Tested on Python 3.11+
- MacOS + VS Code (IDE)
- Didn't use GPU/cloud compute
- Ran locally on mac on VS code

Libraries Used:
- numpy, pands, matplotLib

## Setup for Running

From the project folder path:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows if not Mac, do this:

```bash
.venv\Scripts\activate
```

## Runs toy demo 4-isoform sim

```bash
python main.py --mode toy
```

This runs the main toy 4-isoform paired-end simulation and saves figures into:

```text
outputs/figures/
```

## Run ambiguity experiments (easy/med/hard)

```bash
python main.py --mode ambiguity
```

This runs the hard, medium, and easy transcript-structure settings.

## Run ENCODE-informed pipeline sim

Place your ENCODE TSV file inside `data/raw/`, for example:

```text
data/raw/ENCFF867KVU.tsv
```

Then run:

```bash
python main.py --mode encode --encode-tsv data/raw/ENCFF867KVU.tsv
```

## Run everything (easiest, but output will be long)

```bash
python main.py --mode all --encode-tsv data/raw/ENCFF867KVU.tsv
```

## Notes

- Since outputs are long since we're dealing with a lot of
data, what our group did to assess if the numbers/math
made sense was going on our IDE (VS Code), and then going to
Settings -> Search for Scrollback –> and increase line count (we did like 50K in case)
- The ENCODE TSV is not committed by default because raw data files can be like supeer large. The `.gitignore` leaves out `data/raw/*.tsv` and `data/raw/*.csv`.
- The figures are generated outputs and are also ignored by Git by default.