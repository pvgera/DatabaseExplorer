"""
nhanes_combined_drug_ci.py
Side-by-side CI dot chart: drug class vs Sleep Quality (left) and
drug class vs Depression (right), sharing the same y-axis.

Output:
  results/CherryPicked2/NHANES_combined_drug_ci.png
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engine.viz_style as V
V.apply()

# Import helpers and constants from NHANESPooled (safe — main block is guarded)
sys.path.insert(0, str(ROOT / "NHANES"))
from NHANESPooled import (build_exposure_groups, run_pairwise_mwu,

_EDIT = "--edit" in sys.argv
                           load_cycle, CYCLE_DIRS, CYCLE_YEARS,
                           DEP_MIN_ITEMS, AGE_DISPLAY, AGE_LOW)

OUT     = ROOT / "results" / "CherryPicked2"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DPI = 150

_BRAIN  = {"ssri", "snri", "anxiolytic", "antidepressant (other)", "antipsychotic"}
BRAIN_C = "#1565C0"
HEART_C = "#D32F2F"
REF_C   = "#888888"

# ──────────────────────────────────────────────────────────────────────────────
# Load data (rebuild full df50 including MedicationClasses)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading NHANES cycles ...")
dfs = []
for suffix in CYCLE_DIRS:
    dfs.append(load_cycle(suffix, CYCLE_DIRS[suffix], CYCLE_YEARS[suffix]))
df_all = pd.concat(dfs, ignore_index=True)
df50   = df_all[df_all["RIDAGEYR"] >= AGE_LOW].copy()
df50   = df50[df50["DepressionScore"].notna() & df50["SleepQualityScore"].notna()].copy()
print(f"  Shared analytic cohort: {len(df50):,} participants")

# ──────────────────────────────────────────────────────────────────────────────
# Compute CI stats for both outcomes
# ──────────────────────────────────────────────────────────────────────────────
REF_KEY = "none (reference)"

def _ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    if n < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean()), n
    se = arr.std(ddof=1) / np.sqrt(n)
    m  = arr.mean()
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se), int(n)

groups_slp = build_exposure_groups(df50, score_col="SleepQualityScore")
groups_dep = build_exposure_groups(df50, score_col="DepressionScore")

mwu_slp = run_pairwise_mwu(groups_slp, reference_key=REF_KEY)
mwu_dep = run_pairwise_mwu(groups_dep, reference_key=REF_KEY)

sig_slp = set(mwu_slp[mwu_slp["significant"]]["DrugClass"]) if not mwu_slp.empty else set()
sig_dep = set(mwu_dep[mwu_dep["significant"]]["DrugClass"]) if not mwu_dep.empty else set()

# Union of significant classes from either outcome
all_sig = sig_slp | sig_dep

# Fixed order: heart bottom → brain above → reference at top
heart_sigs = sorted([k for k in all_sig if k not in _BRAIN], reverse=True)
brain_sigs = sorted([k for k in all_sig if k in _BRAIN],     reverse=True)
ordered    = heart_sigs + brain_sigs   # index 0 = bottom of plot

def _row_data(groups, key):
    if key in groups:
        return _ci95(groups[key])
    # class not significant in this outcome — still compute CI for visual reference
    return _ci95(groups[key]) if key in groups else (np.nan, np.nan, np.nan, 0)

rows_slp, rows_dep = [], []
for k in ordered:
    for rows, groups in [(rows_slp, groups_slp), (rows_dep, groups_dep)]:
        m, lo, hi, n = _row_data(groups, k)
        rows.append({"key": k, "mean": m, "lo": lo, "hi": hi, "n": n,
                     "color": BRAIN_C if k in _BRAIN else HEART_C})

# Reference row
ref_slp = _ci95(groups_slp[REF_KEY])
ref_dep = _ci95(groups_dep[REF_KEY])
for rows, vals in [(rows_slp, ref_slp), (rows_dep, ref_dep)]:
    m, lo, hi, n = vals
    rows.append({"key": REF_KEY, "mean": m, "lo": lo, "hi": hi, "n": n,
                 "color": REF_C, "is_ref": True})

n_rows = len(rows_slp)

# ──────────────────────────────────────────────────────────────────────────────
# Build figure
# ──────────────────────────────────────────────────────────────────────────────
fig_h = max(5, n_rows * 0.62)
fig, (ax_slp, ax_dep) = plt.subplots(
    1, 2, figsize=(12, fig_h),
    sharey=True,
    gridspec_kw={"wspace": 0.06}
)

y_positions = list(range(n_rows))
# n= counts embedded as second line under each drug class label
ytick_labels = [f"{r['key']}\nn={r['n']:,}" for r in rows_slp]

panel_configs = [
    (ax_slp, rows_slp, (20, 55),  range(20, 56, 5),
     "Drug class vs Sleep Quality Burden Score",
     "Sleep Quality Score (%) — Mean ± 95% CI", "a"),
    (ax_dep, rows_dep, (0, 40),  range(0, 41, 5),
     "Drug class vs Depression Burden Score",
     "Depression Score (%) — Mean ± 95% CI", "b"),
]

for ax, rows, xlim, xticks, panel_title, xlabel, letter in panel_configs:
    ref_mean = next(r["mean"] for r in rows if r.get("is_ref"))
    ax.axvline(ref_mean, color=REF_C, lw=1.0, linestyle="--", alpha=0.6)

    if heart_sigs and brain_sigs:
        ax.axhline(len(heart_sigs) - 0.5, color="#DDDDDD", lw=0.8, linestyle=":", zorder=1)

    for i, row in enumerate(rows):
        ax.plot([row["lo"], row["hi"]], [i, i], color=row["color"], lw=2.0, zorder=2)
        marker = "D" if row.get("is_ref") else "o"
        ax.scatter(row["mean"], i, color=row["color"], s=60, marker=marker, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(panel_title, fontsize=11, pad=8, color="black", loc="center")
    # Letter pinned to same vertical position as title (1.0 + pad_pts/axes_pts ≈ 1.02)
    ax.text(-0.05, 1.02, letter,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend in bottom-right of panel b
brain_patch = mpatches.Patch(color=BRAIN_C, label="Brain drugs (psychiatric)")
heart_patch = mpatches.Patch(color=HEART_C, label="Heart drugs (cardiovascular)")
ref_patch   = mpatches.Patch(color=REF_C,   label="No medication (reference)")
ax_dep.legend(handles=[brain_patch, heart_patch, ref_patch],
              loc="lower right", fontsize=10, framealpha=0.85)

# Y-axis labels with embedded n= on left panel only (sharey handles ticks)
ax_slp.set_yticks(y_positions)
ax_slp.set_yticklabels(ytick_labels, fontsize=10)

fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])

out_path = OUT / "NHANES_combined_drug_ci.png"
fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close(fig)
print(f"  Saved -> {out_path}")
print("Done.")
