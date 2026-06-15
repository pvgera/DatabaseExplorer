# ScoringFunctions.py — guide

`ScoringFunctions.py` turns raw questionnaire files into clean, scored CSVs. It
is the **only** script you run to produce scores. It is driven entirely by
`VariableDict.xlsx`, so most of the time you change *the dictionary*, not the
code.

---

## What it does, in three steps

1. **Reads** `VariableDict.xlsx` (in the repo root — do not rename it; the script
   looks for that exact name).
2. **Loads** the raw files the dictionary points to, from `RawData/`. It picks a
   reader automatically from the file extension:

   | Extension | Source | Reader |
   |-----------|--------|--------|
   | `.xpt` | NHANES | `pandas.read_sas` |
   | `.dta` | HRS RAND longitudinal file | `pandas.read_stata` |
   | `.DA`  | HRS Core per-wave files | `pandas.read_fwf` + codebook `.txt` |

   > **HRS `.DA` files have no column headers.** The script reads the column
   > positions from the matching codebook `.txt` (e.g. `H16C_R.txt` next to
   > `H16C_R.da`). Keep that `.txt` beside the `.DA` file. See
   > [RawData/RD_README.md](RawData/RD_README.md).

3. **Scores** each composite with the Sum-Score Model and **writes** one CSV per
   dataset to `SF_OUTPUT/`.

---

## The Sum-Score Model

For one participant and one composite (e.g. sleep-quality burden), the score is:

```
            Σ  (r − min) / (max − min)
score  =  ─────────────────────────────  ×  100
                      y
```

- `r` — the participant's response to an item
- `min`, `max` — the item's **minimum effect** / **maximum effect** bounds from
  the dictionary
- `y` — the number of items the participant actually answered

Every item is weighted equally — this is the un-weighted **Sum-Score Model**,
the same approach validated for longitudinal questionnaire scoring by Schlechter
et al. (2022) and used in the associated study. Higher score = more burden,
on a 0–100 scale.

**Reverse-coded items** need no special code: set `min_effect` greater than
`max_effect` for that row and the formula inverts automatically. Worked example
in [VD_README.md](VD_README.md).

**Quality threshold.** A composite is only returned if the participant answered
at least `min_items_fraction` of its items; otherwise the score is `NaN`. Set
this per composite in the dictionary.

---

## How to run

```bash
pip install -r requirements.txt          # once
python ScoringFunctions.py
```

Before running:

1. Fill in `VariableDict.xlsx` — see [VD_README.md](VD_README.md).
2. Put raw files in `RawData/` — see [RawData/RD_README.md](RawData/RD_README.md).
3. Open `ScoringFunctions.py` and edit the **`DATASETS`** block near the top to
   choose which datasets to score and to **name your output files**:

   ```python
   DATASETS = {
       "NHANES_pilot": {"output_name": "nhanes_scored.csv", "id": "SEQN"},
       "HRS_RAND":     {"output_name": "hrs_rand_scored.csv", "id": "HHIDPN"},
       "HRS_2016":     {"output_name": "hrs_2016_scored.csv", "id": "HHID_PN"},
   }
   ```

   - The **key** must match the `dataset` column in your dictionary.
   - `output_name` is whatever you want the file in `SF_OUTPUT/` to be called.
   - `id` is the participant-id column: `SEQN` for NHANES, `HHIDPN` for the HRS
     RAND file, `HHID_PN` for HRS Core `.DA` files (built automatically from
     `HHID` + `PN`).
   - Comment out a dataset to skip it.

---

## Score vs. pass-through

Each dictionary row is one of two kinds, set by the `composite_type` column:

- **`score`** — the row is an item in a multi-item composite (e.g. each CESD-8
  question). Rows sharing a `composite_name` are summed into one 0–100 score.
- **`passthrough`** — the variable is emitted as a cleaned raw value, not scored
  (e.g. age, sex, BMI, a disease flag). The study did **not** score age, sex,
  blood pressure, weight, height, or medication use — those are pass-through.

The output CSV has one row per participant: the id column, then every
pass-through variable, then every scored composite.

---

## When you *do* edit the code

You should rarely need to, but:

- **Add a new dataset** → add an entry to `DATASETS` and describe its variables
  in the dictionary. If it is a new file format, add a reader in `load_file()`.
- **Change missing-value handling** → NHANES sentinels (7, 9, 77, …) are handled
  automatically; for other datasets, list missing codes in the dictionary's
  `missing_values` column. The base rules live in `clean_val()` /
  `parse_missing_set()`.
- **Change the score formula** → it lives in `normalize_item()` and
  `score_pct()`. Changing it changes every composite, so prefer editing the
  dictionary instead.

---

## Output

CSVs land in `SF_OUTPUT/`. Their contents are git-ignored (they may be derived
from data you are not allowed to redistribute), but the folder and its README
stay in the repo. Downstream analysis reads from here — see
[StatsAnal.ipynb](StatsAnal.ipynb).
