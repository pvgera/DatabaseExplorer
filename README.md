# DatabaseExplorer

**A variable-dictionary-driven pipeline for scoring public-health questionnaire
data — and reproducing the associations from the study _"Surrogates of the
Central Autonomic Network as Predictive Markers for Arrhythmia."_**

You describe your questionnaire items once, in a spreadsheet
(`VariableDict.xlsx`). One script (`ScoringFunctions.py`) reads that spreadsheet,
loads the raw data it points to, and writes clean 0–100 composite scores. From
there, a notebook reproduces the study's statistical findings. No data ships
with this repo — you bring your own NHANES / HRS files.

This single README is meant to get you launched. Each component also has its own
short guide (linked below).

---

## What's here

```
DatabaseExplorer/
├── README.md                 ← you are here
├── requirements.txt          ← Python dependencies (pip install -r)
├── LICENSE                   ← MIT (code) + data-use disclaimer
│
├── VariableDict.xlsx         ← THE control panel: describe your variables here
├── VD_README.md              ← how to fill the dictionary (worked examples)
│
├── ScoringFunctions.py       ← reads the dictionary, scores composites
├── SF_README.md              ← how scoring works (the Sum-Score Model)
│
├── StatsAnal.ipynb           ← reproduces the study's associations, Fig 1–7
│
├── SupplementaryScripts/
│   ├── MedicationClassifier.py   drug-name -> class mapping
│   ├── StatsHelpers.py           shared paths + stats utilities  (SH_README.md)
│   ├── prep/                     longitudinal data-prep pipeline
│   ├── legacy_figures/           original published-figure scripts (reference)
│   └── SS_README.md              guide to this folder
│
├── RawData/                  ← put your downloaded data here (git-ignored)
│   └── RD_README.md              what to download and where it goes
│
└── SF_OUTPUT/                ← scored CSVs land here (git-ignored)
    └── SFO_README.md
```

---

## What it does / what's been done

- **Scoring engine** — `ScoringFunctions.py` turns raw NHANES (`.xpt`), HRS RAND
  (`.dta`), and HRS Core (`.DA`) files into 0–100 burden scores using an
  un-weighted **Sum-Score Model**, driven entirely by `VariableDict.xlsx`.
- **Validated on real HRS data** — the current pipeline reproduces the study's
  core associations: the sleep↔depression correlation (Spearman ρ ≈ 0.48 on a
  single wave vs the memoir's pooled 0.55), and the headline **sex-specific
  sleep→arrhythmia** signal (2016-landmark Cox: women HR ≈ 1.22/SD, p ≈ 0.002;
  men null — matching memoir Fig 4h, women up to 1.24/SD significant, men not).
- **Reproducible analyses** — `StatsAnal.ipynb` walks the seven memoir figures in
  order, recomputing each association from the current pipeline, with method
  notes and limitations.
- **Provenance** — the exact scripts that produced the published figures are kept
  read-only in `SupplementaryScripts/legacy_figures/`.

---

## Quickstart

```bash
# 1. Install dependencies (Python 3.13; a fresh virtualenv is recommended)
pip install -r requirements.txt

# 2. Get the data (see RawData/RD_README.md) and place it under RawData/

# 3. Describe your variables in VariableDict.xlsx (see VD_README.md)

# 4. Score
python ScoringFunctions.py                 # -> SF_OUTPUT/*.csv

# 5. (For the longitudinal analyses) build the analytic frames
python SupplementaryScripts/prep/build_fullcohort.py
python SupplementaryScripts/prep/build_analytic_dataset_expanded.py
python SupplementaryScripts/prep/analysis_timelag.py
python SupplementaryScripts/prep/analysis_timelag_multiwave.py
python SupplementaryScripts/prep/event_anchored_trajectory.py

# 6. Reproduce the associations
jupyter notebook StatsAnal.ipynb
```

---

## The scoring model, in one line

```
score = ( Σ (r − min) / (max − min) ) / y  × 100
```

per participant per composite, where `r` is a response, `(min, max)` are the
item's effect bounds from the dictionary, and `y` is the number of items
answered. Reverse-coded items just set `min > max`. Full detail in
[SF_README.md](SF_README.md).

---

## Reproducing the study

The published findings came from the original scripts (now in
`SupplementaryScripts/legacy_figures/`), which used a slightly different scoring
method. `StatsAnal.ipynb` instead recomputes the same **associations** with the
current `VariableDict.xlsx` + `ScoringFunctions.py` approach — the aim is to land
on the same conclusions, not the same pixels. See the notebook's intro for the
full method note.

---

## Data & ethics

No participant-level data is included. NHANES and HRS are obtained directly from
their programs under their data-use terms:

- NHANES / CDC NCHS — <https://www.cdc.gov/nchs/nhanes/>
- Health and Retirement Study, University of Michigan (NIA U01AG009740) —
  <https://hrsdata.isr.umich.edu/>

See [LICENSE](LICENSE) for the code license (MIT) and the full data disclaimer.

---

## Where to go next

| I want to… | Read |
|------------|------|
| Fill in the variable dictionary | [VD_README.md](VD_README.md) |
| Understand scoring | [SF_README.md](SF_README.md) |
| Place my raw data | [RawData/RD_README.md](RawData/RD_README.md) |
| Understand the helper scripts | [SupplementaryScripts/SS_README.md](SupplementaryScripts/SS_README.md) |
| Reproduce the analyses | `StatsAnal.ipynb` |
