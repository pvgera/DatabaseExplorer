"""
slope_chargeaf_interaction.py
==============================
Tests whether the SQB slope interacts with CHARGE-AF risk to predict arrhythmia.

Core question: does a rising SQB slope add discriminative value specifically
among people who already have CHARGE-AF risk factors?

Analyses:
  1. Cox model: slope x CHARGE-AF linear predictor interaction
  2. Stratified slopes: cases vs high-risk controls vs low-risk controls
  3. C-statistic comparison: CHARGE-AF alone vs CHARGE-AF + slope (in high-risk only)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import linregress, mannwhitneyu

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from engine.variable_loader import VariableDictionary

_EDIT = "--edit" in sys.argv
_vd = VariableDictionary()
DATA_DIR   = ROOT / "results" / "data"
SCORES_CSV = ROOT / "results" / "05_timelag" / "data" / "multiwave_scores.csv"
COX_CSV    = DATA_DIR / "hrs_cox_long_expanded.csv"
OUT_DIR    = ROOT / "results" / "CherryPicked2"

WAVE_YEAR = {2:1994,3:1996,4:1998,5:2000,6:2002,7:2004,8:2006,
             10:2010,12:2014,13:2016,14:2018,15:2020,16:2022}
MIN_POINTS = 3

print("=" * 68)
print("  SQB SLOPE x CHARGE-AF INTERACTION ANALYSIS")
print("=" * 68)

# ── Load data and compute slopes ──────────────────────────────────────────────
cox    = pd.read_csv(COX_CSV)
scores = pd.read_csv(SCORES_CSV)

SLEEP_WAVE_COLS = {n: f"sleep_w{n}" for n in [6,7,8,10,12,13,14,15,16]
                   if f"sleep_w{n}" in scores.columns}

slope_rows = []
for _, row in scores.iterrows():
    slp_pts = [(WAVE_YEAR[n], row[col])
               for n, col in SLEEP_WAVE_COLS.items() if not pd.isna(row[col])]
    slp_slope = np.nan
    if len(slp_pts) >= MIN_POINTS:
        yrs, vals = zip(*slp_pts)
        slp_slope, *_ = linregress(yrs, vals)
    slope_rows.append({"HHID_PN": row["HHID_PN"], "sleep_slope": slp_slope})

cox = cox.merge(pd.DataFrame(slope_rows), on="HHID_PN", how="left")

# ── Standardise ───────────────────────────────────────────────────────────────
for col in ["age_2016", "height_cm", "weight_kg", "dep_2016", "sleep_2016", "sleep_slope"]:
    if col in cox.columns:
        mu, sd = cox[col].mean(), cox[col].std()
        if sd > 0:
            cox[col + "_z"] = (cox[col] - mu) / sd

_cont    = _vd.continuous_composites("HRS_2016", "charge_covariate")
_bin     = _vd.binary_composites("HRS_2016", "charge_covariate")
CHARGE_Z = [c + "_z" for c in _cont] + _bin
cc = cox[CHARGE_Z + ["sleep_slope_z", "dep_2016_z", "sleep_2016_z",
                      "tstart", "tstop", "event"]].dropna()
cox = cox.loc[cc.index].copy()
print(f"  Complete cases: n={len(cox):,}  events={cox['event'].sum()}")

# ── Step 1: Fit CHARGE-AF model and extract linear predictor ──────────────────
print("\n[1] Fitting CHARGE-AF base model ...")
cph_base = CoxPHFitter(penalizer=0.01)
cph_base.fit(cox[CHARGE_Z + ["tstart","tstop","event"]].dropna(),
             duration_col="tstop", event_col="event", entry_col="tstart",
             formula=" + ".join(CHARGE_Z))
cox["charge_lp"] = cph_base.predict_log_partial_hazard(cox[CHARGE_Z].fillna(0))

# Z-score the linear predictor
mu, sd = cox["charge_lp"].mean(), cox["charge_lp"].std()
cox["charge_lp_z"] = (cox["charge_lp"] - mu) / sd

c_base = concordance_index(cox["tstop"], -cph_base.predict_partial_hazard(
    cox[CHARGE_Z].fillna(0)), cox["event"])
print(f"  CHARGE-AF C = {c_base:.4f}")

# ── Step 2: Cox with slope x CHARGE-AF interaction ────────────────────────────
print("\n[2] Cox model: sleep_slope x CHARGE-AF interaction ...")
cox["slope_x_charge"] = cox["sleep_slope_z"] * cox["charge_lp_z"]

int_cols = CHARGE_Z + ["sleep_slope_z", "slope_x_charge", "tstart", "tstop", "event"]
cc_int = cox[int_cols].dropna()

cph_int = CoxPHFitter(penalizer=0.01)
cph_int.fit(cc_int, duration_col="tstop", event_col="event", entry_col="tstart",
            formula=" + ".join(CHARGE_Z) + " + sleep_slope_z + slope_x_charge")

r_int = cph_int.summary.loc["slope_x_charge"]
hr_int = np.exp(r_int["coef"])
p_int  = r_int["p"]
ci_lo  = np.exp(r_int["coef lower 95%"])
ci_hi  = np.exp(r_int["coef upper 95%"])

r_slope = cph_int.summary.loc["sleep_slope_z"]
hr_slope = np.exp(r_slope["coef"])
p_slope  = r_slope["p"]

print(f"  sleep_slope main effect: HR={hr_slope:.3f}  p={p_slope:.4f}")
print(f"  slope x CHARGE-AF interaction: HR={hr_int:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]  p={p_int:.4f}")
sig = "SIGNIFICANT" if p_int < 0.05 else ("marginal" if p_int < 0.10 else "ns")
print(f"  → Interaction is {sig}")

c_int = concordance_index(cc_int["tstop"],
                           -cph_int.predict_partial_hazard(cc_int), cc_int["event"])
print(f"  C-stat with interaction: {c_int:.4f}  (vs CHARGE-AF alone: {c_base:.4f})")

# ── Step 3: Stratify controls by CHARGE-AF risk level ────────────────────────
print("\n[3] Stratified slope comparison ...")

# CHARGE-AF risk tertiles among controls
controls = cox[cox["event"] == 0].copy()
cases    = cox[cox["event"] == 1].copy()

tertile_cuts = controls["charge_lp"].quantile([1/3, 2/3]).values
controls["risk_group"] = pd.cut(controls["charge_lp"],
                                 bins=[-np.inf, tertile_cuts[0], tertile_cuts[1], np.inf],
                                 labels=["Low risk", "Medium risk", "High risk"])
cases["risk_group"] = "Cases"

all_groups = pd.concat([cases, controls])

print(f"\n  {'Group':<20} {'n':>6}  {'mean slope':>12}  {'median slope':>13}  {'SD':>8}")
print(f"  {'─'*65}")

group_data = {}
for grp in ["Cases", "Low risk", "Medium risk", "High risk"]:
    sub = all_groups[all_groups["risk_group"] == grp]["sleep_slope"].dropna()
    group_data[grp] = sub.values
    print(f"  {grp:<20} {len(sub):>6}  {sub.mean():>12.4f}  {np.median(sub):>13.4f}  {sub.std():>8.4f}")

# MWU: Cases vs each control tier
print(f"\n  Mann-Whitney U comparisons (cases vs control tiers):")
for grp in ["Low risk", "Medium risk", "High risk"]:
    U, p = mannwhitneyu(group_data["Cases"], group_data[grp], alternative="two-sided")
    r = (2*U)/(len(group_data["Cases"])*len(group_data[grp])) - 1
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  Cases vs {grp:<15}: U={U:.0f}  p={p:.4f}  r={r:.3f}  {sig}")

# ── Step 4: C-stat in high-risk subgroup ─────────────────────────────────────
print("\n[4] C-stat in high-CHARGE-AF-risk participants only ...")
high_risk_mask = (cox["charge_lp"] >= tertile_cuts[1]) | (cox["event"] == 1)
cox_hi = cox[high_risk_mask].copy()
print(f"  High-risk subgroup: n={len(cox_hi):,}  events={cox_hi['event'].sum()}")

needed = CHARGE_Z + ["sleep_slope_z", "tstart", "tstop", "event"]
cc_hi = cox_hi[needed].dropna()

cph_hi_base = CoxPHFitter(penalizer=0.01)
cph_hi_base.fit(cc_hi[CHARGE_Z + ["tstart","tstop","event"]],
                duration_col="tstop", event_col="event", entry_col="tstart",
                formula=" + ".join(CHARGE_Z))
c_hi_base = concordance_index(cc_hi["tstop"],
    -cph_hi_base.predict_partial_hazard(cc_hi[CHARGE_Z]), cc_hi["event"])

cph_hi_slope = CoxPHFitter(penalizer=0.01)
cph_hi_slope.fit(cc_hi, duration_col="tstop", event_col="event", entry_col="tstart",
                  formula=" + ".join(CHARGE_Z) + " + sleep_slope_z")
c_hi_slope = concordance_index(cc_hi["tstop"],
    -cph_hi_slope.predict_partial_hazard(cc_hi[needed[:-3]]), cc_hi["event"])

print(f"  CHARGE-AF alone:       C = {c_hi_base:.4f}")
print(f"  CHARGE-AF + SQB slope: C = {c_hi_slope:.4f}  (ΔC = {c_hi_slope - c_hi_base:+.4f})")

# ── Plot: slope distributions by risk group ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

colors_grp = {"Cases": "#E07828", "Low risk": "#AEC6CF",
              "Medium risk": "#6A9FB5", "High risk": "#2C5F8A"}
order = ["Low risk", "Medium risk", "High risk", "Cases"]

for i, grp in enumerate(order):
    data = all_groups[all_groups["risk_group"] == grp]["sleep_slope"].dropna()
    bp = ax.boxplot(data, positions=[i], widths=0.5,
                    patch_artist=True, notch=False,
                    boxprops=dict(facecolor=colors_grp[grp], alpha=0.7),
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", alpha=0.2, markersize=3),
                    whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))

ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels([f"{g}\n(n={len(all_groups[all_groups['risk_group']==g]['sleep_slope'].dropna()):,})"
                    for g in order], fontsize=11)
ax.set_ylabel("SQB Slope (score units / year)", fontsize=12)
ax.set_title("SQB Trajectory Slope by CHARGE-AF Risk Group and Arrhythmia Case Status\n"
             "Controls stratified into CHARGE-AF risk tertiles (2016 baseline, HRS)",
             fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

patches = [mpatches.Patch(color=colors_grp[g], alpha=0.7, label=g) for g in order]
ax.legend(handles=patches, fontsize=9, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "HRS_slope_by_chargeaf_risk.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close()
print(f"\n  Plot saved -> {out.name}")
print("=" * 68)
