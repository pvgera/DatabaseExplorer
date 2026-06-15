# SupplementaryScripts — guide

Everything in here supports the analysis layer. Scoring itself lives in the
repo root (`ScoringFunctions.py`); this folder holds the medication classifier,
shared statistical helpers, the longitudinal **data-prep** pipeline, and a
read-only archive of the **legacy figure scripts** that produced the published
figures.

```
SupplementaryScripts/
├── MedicationClassifier.py   drug-name -> class mapping (feeds the drug-class figure)
├── StatsHelpers.py           repo paths + shared stats utilities  (see SH_README.md)
├── prep/                     longitudinal data-prep pipeline (builds analytic CSVs)
└── legacy_figures/           the original, as-published figure scripts (reference only)
```

---

## `MedicationClassifier.py`

Maps free-text prescription names (e.g. NHANES `RXDDRUG`) to broad drug classes
and rolls them up to one row per participant. Medication is **not scored** — this
just labels it for the drug-class analysis. See the docstring for usage.

---

## `StatsHelpers.py`

Repo-path constants (`RAW`, `SCORED`, `ANALYTIC`, `FIGS`), significance/CI
helpers, and the medication exposure-group + pairwise Mann-Whitney machinery.
Imported by the prep scripts and by `StatsAnal.ipynb`. Full notes in
[SH_README.md](SH_README.md).

---

## `prep/` — the longitudinal data-prep pipeline

The memoir's longitudinal analyses (event-anchored trajectories, landmark Cox,
competing risks) operate on **wave-by-wave** analytic frames, not on a single
scored file. These scripts build those frames from the HRS RAND `.dta` file and
the per-wave HRS Core `.DA`/codebook files. Run them **in this order**:

| # | Script | Produces (in `SF_OUTPUT/analytic/`) |
|---|--------|--------------------------------------|
| 1 | `build_fullcohort.py` | `hrs_analytic_wide_fullcohort.csv` |
| 2 | `build_analytic_dataset_expanded.py` | `hrs_cox_long_expanded.csv` |
| 3 | `analysis_timelag.py` | `wide_with_intermediate_waves.csv` |
| 4 | `analysis_timelag_multiwave.py` | `multiwave_scores.csv` |
| 5 | `event_anchored_trajectory.py` | `event_anchored_scores.csv` |

```bash
python SupplementaryScripts/prep/build_fullcohort.py
python SupplementaryScripts/prep/build_analytic_dataset_expanded.py
python SupplementaryScripts/prep/analysis_timelag.py
python SupplementaryScripts/prep/analysis_timelag_multiwave.py
python SupplementaryScripts/prep/event_anchored_trajectory.py
```

**Inputs** (place under `RawData/` — see [../RawData/RD_README.md](../RawData/RD_README.md)):
the RAND HRS file (`randhrs1992_2022v1.dta`) and the per-wave HRS Core
`.DA` + codebook `.txt` files (2010–2022). Paths have been adapted to this
repo's layout (`RawData/`, `SF_OUTPUT/analytic/`).

> **Method note.** These prep scripts re-derive wave-level sleep / depression
> burden internally using the study's original logic. They predate the unified
> `ScoringFunctions.py` engine, so their scoring can differ slightly in edge
> cases (item imputation, missing-code handling). `StatsAnal.ipynb` documents
> where this matters and confirms the **associations** are unchanged.

---

## `legacy_figures/` — reference archive (read-only)

These are the **exact scripts that produced the published memoir figures**. They
are kept verbatim for provenance and for anyone who wants the original panels.

> ⚠️ They are **not** wired into this repo's clean layout. They reference the
> original project's module structure (`NHANESPooled`, `engine.viz_style`, the
> original `results/` tree) and are not guaranteed to run standalone here. Treat
> them as a record of *how the published figures were made*. The reproducible,
> association-focused analysis lives in `StatsAnal.ipynb`.

| Memoir figure | Legacy script |
|---------------|---------------|
| Fig 1 — attrition | `attrition_charts.py` |
| Fig 2 — sleep vs depression scatter | `sleep_dep_scatter.py` |
| Fig 3 — drug class vs burden | `nhanes_combined_drug_ci.py`, `nhanes_drug_category_comparison.py` |
| Fig 4 — sex-stratified trajectories + hazard | `hrs_combined_trajectory_hr_bysex.py`, `sex_stratified_landmark_cox.py` |
| Fig 5 — CHARGE-AF C-statistic + slopes | `update_cstat_model4.py`, `slope_chargeaf_eventanchored.py`, `slope_chargeaf_interaction.py` |
| Fig 6 — CVD by trajectory | `figure6_combined.py`, `hrs_combined_cvd_figure.py` |
| Fig 7 — competing-risks CVD incidence | `figure7_cvd_survival.py` |
