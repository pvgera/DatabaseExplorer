# StatsHelpers.py — guide

A small shared toolbox imported by the prep scripts and by `StatsAnal.ipynb`.
It keeps repo paths and common statistical operations in one place so the
analysis code stays short and consistent.

```python
import sys; sys.path.insert(0, "SupplementaryScripts")
from StatsHelpers import (REPO_ROOT, RAW, SCORED, ANALYTIC, FIGS,
                          significance_label, ci95, mwu_effect_r,
                          build_exposure_groups, run_pairwise_mwu, load_hrs_da)
```

## What's inside

**Repo paths** — one source of truth for the analysis layer:

| Name | Points to |
|------|-----------|
| `REPO_ROOT` | the repository root |
| `RAW` | `RawData/` (your downloaded NHANES / HRS files) |
| `SCORED` | `SF_OUTPUT/` (outputs of `ScoringFunctions.py`) |
| `ANALYTIC` | `SF_OUTPUT/analytic/` (intermediate CSVs from `prep/`) |
| `FIGS` | `SF_OUTPUT/figures/` (rendered figures) |

`ANALYTIC` and `FIGS` are created automatically on import.

**Statistics**

- `significance_label(p)` — `"p < 0.001 ***"`, `"p < 0.01 **"`, … for annotations.
- `ci95(arr)` — returns `(mean, lo, hi, n)` for a 95% normal-approx CI.
- `mwu_effect_r(U, n1, n2)` — effect size `r = |Z| / sqrt(N)` for a Mann-Whitney U.

**Medication analysis** (used by the drug-class figure)

- `build_exposure_groups(df, score_col)` — non-exclusive per-class score pools
  plus a no-medication reference. Needs a `MedicationClasses` column from
  `MedicationClassifier.participant_drug_classes`.
- `run_pairwise_mwu(groups, reference_key)` — each class vs. the reference,
  Holm-corrected, with effect sizes and significance labels.

**HRS loader**

- `load_hrs_da(da_path, variables)` — thin wrapper over `ScoringFunctions._load_da`
  for reading HRS Core fixed-width `.DA` files (needs the codebook `.txt`).

## Adjusting paths

If you keep raw data or outputs somewhere else, edit the path block at the top
of `StatsHelpers.py`. Everything downstream follows from those constants.
