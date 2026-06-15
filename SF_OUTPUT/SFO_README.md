# SF_OUTPUT — scored outputs

This folder collects everything the pipeline produces. **Its contents are
git-ignored** (they are derived from data you may not redistribute); only this
README and a `.gitkeep` are tracked, so the folder always exists.

## What lands here

| Path | Produced by | Contents |
|------|-------------|----------|
| `SF_OUTPUT/*.csv` | `ScoringFunctions.py` | one wide scored file per dataset — id column, pass-through variables, and 0–100 composite scores. You choose the file names in the `DATASETS` block of `ScoringFunctions.py`. |
| `SF_OUTPUT/analytic/*.csv` | `SupplementaryScripts/prep/` | longitudinal analytic frames (wide cohort, cox-long, multiwave scores, event-anchored scores) used by the notebook. |
| `SF_OUTPUT/figures/*.png` | the analyses / prep scripts | rendered figures. |

## Naming your scored files

In `ScoringFunctions.py`:

```python
DATASETS = {
    "NHANES_pilot": {"output_name": "nhanes_scored.csv", "id": "SEQN"},
    "HRS_RAND":     {"output_name": "hrs_rand_scored.csv", "id": "HHIDPN"},
}
```

`output_name` is yours to choose. The downstream analyses
(`StatsAnal.ipynb`) read from here — if you rename a file, update the path
constants in the notebook's setup cell to match.

## From here

Open `../StatsAnal.ipynb` to reproduce the study's associations from these
outputs.
