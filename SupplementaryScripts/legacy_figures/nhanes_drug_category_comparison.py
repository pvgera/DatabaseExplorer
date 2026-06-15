"""
nhanes_drug_category_comparison.py
Compare sleep quality and depression scores across 4 mutually-assigned
drug category groups:
  No medication  |  Heart drugs only  |  Brain drugs only  |  Both

Pairwise Mann-Whitney U with Holm correction.

Output:
  results/CherryPicked2/NHANES_drug_category_comparison.png
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from itertools import combinations

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engine.viz_style as V
V.apply()

sys.path.insert(0, str(ROOT / "NHANES"))
from NHANESPooled import load_cycle, CYCLE_DIRS, CYCLE_YEARS, AGE_LOW

_EDIT = "--edit" in sys.argv

OUT     = ROOT / "results" / "CherryPicked2"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DPI = 150

_BRAIN  = {"ssri", "snri", "anxiolytic", "antidepressant (other)", "antipsychotic"}

COLORS = {
    "No medication":   "#888888",
    "Heart drugs only": "#D32F2F",
    "Brain drugs only": "#1565C0",
    "Both":             "#7B1FA2",
}
GROUP_ORDER = ["No medication", "Heart drugs only", "Brain drugs only", "Both"]

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
print("Loading NHANES cycles ...")
dfs = []
for suffix in CYCLE_DIRS:
    dfs.append(load_cycle(suffix, CYCLE_DIRS[suffix], CYCLE_YEARS[suffix]))
df_all = pd.concat(dfs, ignore_index=True)
df50   = df_all[df_all["RIDAGEYR"] >= AGE_LOW].copy()
df50   = df50[df50["DepressionScore"].notna() & df50["SleepQualityScore"].notna()].copy()
print(f"  Analytic cohort: {len(df50):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Classify into 4 groups
# ──────────────────────────────────────────────────────────────────────────────
def _parse_classes(val):
    if isinstance(val, list):
        return {str(c).lower().strip() for c in val}
    if isinstance(val, str):
        # stored as Python list string e.g. "['ssri', 'beta_blocker']"
        val = val.strip("[]").replace("'", "").replace('"', "")
        return {c.lower().strip() for c in val.split(",") if c.strip()}
    return set()

mc = df50["MedicationClasses"].apply(_parse_classes)
has_brain = mc.apply(lambda s: bool(s & _BRAIN))
has_heart = mc.apply(lambda s: bool(s - _BRAIN - {"none", ""}))

def _assign_group(b, h):
    if b and h:   return "Both"
    if b:         return "Brain drugs only"
    if h:         return "Heart drugs only"
    return "No medication"

df50["drug_group"] = [_assign_group(b, h) for b, h in zip(has_brain, has_heart)]

dist = df50["drug_group"].value_counts()
print("\nGroup sizes:")
for g in GROUP_ORDER:
    print(f"  {g:20s}: n={dist.get(g, 0):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Compute mean ± 95% CI per group
# ──────────────────────────────────────────────────────────────────────────────
def _ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    if n < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean()), n
    se = arr.std(ddof=1) / np.sqrt(n)
    m  = arr.mean()
    return float(m), float(m - 1.96*se), float(m + 1.96*se), int(n)

rows_slp, rows_dep = [], []
for g in GROUP_ORDER:
    sub = df50[df50["drug_group"] == g]
    for rows, col in [(rows_slp, "SleepQualityScore"), (rows_dep, "DepressionScore")]:
        m, lo, hi, n = _ci95(sub[col].dropna().values)
        rows.append({"group": g, "mean": m, "lo": lo, "hi": hi, "n": n,
                     "color": COLORS[g]})

# ──────────────────────────────────────────────────────────────────────────────
# Pairwise Mann-Whitney U with Holm correction
# ──────────────────────────────────────────────────────────────────────────────
def _pairwise_mwu_holm(df, score_col, groups):
    pairs = list(combinations(groups, 2))
    results = []
    for g1, g2 in pairs:
        a = df.loc[df["drug_group"] == g1, score_col].dropna().values
        b = df.loc[df["drug_group"] == g2, score_col].dropna().values
        if len(a) < 2 or len(b) < 2:
            results.append((g1, g2, 1.0))
            continue
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        results.append((g1, g2, p))
    # Holm correction
    results.sort(key=lambda x: x[2])
    n_tests = len(results)
    corrected = []
    for rank, (g1, g2, p) in enumerate(results):
        p_holm = min(p * (n_tests - rank), 1.0)
        corrected.append((g1, g2, p, p_holm))
    return corrected

print("\nPairwise Mann-Whitney U (Holm-corrected):")
mwu_slp = _pairwise_mwu_holm(df50, "SleepQualityScore", GROUP_ORDER)
mwu_dep = _pairwise_mwu_holm(df50, "DepressionScore",   GROUP_ORDER)

for label, results in [("Sleep Quality", mwu_slp), ("Depression", mwu_dep)]:
    print(f"\n  {label}:")
    for g1, g2, p_raw, p_holm in results:
        sig = "***" if p_holm < 0.001 else ("**" if p_holm < 0.01 else ("*" if p_holm < 0.05 else "ns"))
        print(f"    {g1:20s} vs {g2:20s}  p_raw={p_raw:.4f}  p_holm={p_holm:.4f}  {sig}")

# Build lookup for significance stars
def _sig_lookup(mwu_results):
    lut = {}
    for g1, g2, p_raw, p_holm in mwu_results:
        stars = ("***" if p_holm < 0.001 else
                 ("**"  if p_holm < 0.01  else
                  ("*"   if p_holm < 0.05  else "ns")))
        lut[(g1, g2)] = stars
        lut[(g2, g1)] = stars
    return lut

sig_slp = _sig_lookup(mwu_slp)
sig_dep = _sig_lookup(mwu_dep)

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
fig_h = max(4, len(GROUP_ORDER) * 0.9)
fig, (ax_slp, ax_dep) = plt.subplots(
    1, 2, figsize=(13, fig_h),
    sharey=True,
    gridspec_kw={"wspace": 0.06}
)

y_pos = list(range(len(GROUP_ORDER)))
ytick_labels = [f"{r['group']}\nn={r['n']:,}" for r in rows_slp]

panel_configs = [
    (ax_slp, rows_slp, sig_slp, (20, 55), range(20, 56, 5),
     "Drug Category vs Sleep Quality Burden Score",
     "Sleep Quality Score (%) — Mean ± 95% CI", "a"),
    (ax_dep, rows_dep, sig_dep, (0,  40), range(0,  41, 5),
     "Drug Category vs Depression Burden Score",
     "Depression Score (%) — Mean ± 95% CI", "b"),
]

REF_GROUP = "No medication"

for ax, rows, sig_lut, xlim, xticks, panel_title, xlabel, letter in panel_configs:
    ref_mean = next(r["mean"] for r in rows if r["group"] == REF_GROUP)
    ax.axvline(ref_mean, color=COLORS[REF_GROUP], lw=1.0, linestyle="--", alpha=0.6)

    # Separator between no-med and drug groups
    ax.axhline(0.5, color="#DDDDDD", lw=0.8, linestyle=":", zorder=1)

    for i, row in enumerate(rows):
        ax.plot([row["lo"], row["hi"]], [i, i], color=row["color"], lw=2.5, zorder=2)
        marker = "D" if row["group"] == REF_GROUP else "o"
        ax.scatter(row["mean"], i, color=row["color"], s=70, marker=marker, zorder=3)

    # Annotate pairwise significance for the key comparison: Brain vs Heart
    brain_idx = next(i for i, r in enumerate(rows) if r["group"] == "Brain drugs only")
    heart_idx = next(i for i, r in enumerate(rows) if r["group"] == "Heart drugs only")
    stars = sig_lut.get(("Brain drugs only", "Heart drugs only"), "ns")
    if stars != "ns":
        xmid  = (rows[brain_idx]["mean"] + rows[heart_idx]["mean"]) / 2
        ymid  = (brain_idx + heart_idx) / 2
        ax.annotate("", xy=(xmid, brain_idx - 0.08), xytext=(xmid, heart_idx + 0.08),
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=1.0))
        ax.text(xmid + 0.3, ymid, f"Brain vs Heart\n{stars}",
                ha="left", va="center", fontsize=8, color="#444444")

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(panel_title, fontsize=11, pad=8, color="black")
    ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
            ha="left", va="center", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend
patches = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUP_ORDER]
ax_dep.legend(handles=patches, loc="lower right", fontsize=10, framealpha=0.85)

# Y-axis labels (left panel only)
ax_slp.set_yticks(y_pos)
ax_slp.set_yticklabels(ytick_labels, fontsize=10)

fig.tight_layout()

# ── Significance matrix inset (upper-right of panel b) ────────────────────────
ABBREV = {"No medication": "No med", "Heart drugs only": "Heart",
          "Brain drugs only": "Brain", "Both": "Both"}
SIG_COLORS = {"***": "#1A5276", "**": "#2E86C1", "*": "#85C1E9", "ns": "#EEEEEE"}
SIG_TEXT_C = {"***": "white",   "**": "white",   "*": "#333333", "ns": "#999999"}

n_grp  = len(GROUP_ORDER)
ax_ins = ax_dep.inset_axes([0.52, 0.48, 0.46, 0.50])

for row_i, g_row in enumerate(GROUP_ORDER):
    for col_j, g_col in enumerate(GROUP_ORDER):
        if col_j >= row_i:          # upper triangle + diagonal: blank
            ax_ins.add_patch(plt.Rectangle((col_j, n_grp - 1 - row_i), 1, 1,
                                           color="white", zorder=1))
            continue
        stars = sig_slp.get((g_row, g_col), "ns")   # sleep significance
        stars_dep = sig_dep.get((g_row, g_col), "ns")
        # Show sleep / dep stacked as two half-rows
        for half, s, yoff in [(0, stars, 0.5), (1, stars_dep, 0.0)]:
            fc = SIG_COLORS[s]
            tc = SIG_TEXT_C[s]
            y  = n_grp - 1 - row_i + yoff
            ax_ins.add_patch(plt.Rectangle((col_j, y), 1, 0.5, color=fc, zorder=2))
            ax_ins.text(col_j + 0.5, y + 0.25, s, ha="center", va="center",
                        fontsize=6.5, color=tc, fontweight="bold", zorder=3)

ax_ins.set_xlim(0, n_grp)
ax_ins.set_ylim(0, n_grp)
ax_ins.set_xticks([i + 0.5 for i in range(n_grp)])
ax_ins.set_xticklabels([ABBREV[g] for g in GROUP_ORDER], fontsize=7, rotation=30, ha="right")
ax_ins.set_yticks([n_grp - 1 - i + 0.5 for i in range(n_grp)])
ax_ins.set_yticklabels([ABBREV[g] for g in GROUP_ORDER], fontsize=7)
ax_ins.tick_params(length=0)
ax_ins.set_title("Pairwise significance\n(top=sleep, bottom=dep)", fontsize=7, pad=4)
for spine in ax_ins.spines.values():
    spine.set_visible(False)

out_path = OUT / "NHANES_drug_category_comparison.png"
fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close(fig)
print(f"\nSaved -> {out_path}")
print("Done.")
