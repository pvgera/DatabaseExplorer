"""
slope_chargeaf_eventanchored.py
================================
Event-anchored SQB slope comparison stratified by CHARGE-AF risk.

For each participant, the SQB slope is computed from data at lags -10 to -5
(i.e., 5-10 years before their arrhythmia event for cases, or 5-10 years
before their anchor year for controls). This avoids post-event contamination
and aligns with the methodology of Figures 4 and 5.

Controls are stratified into CHARGE-AF risk tertiles.
Cases (n=944) are compared against each control tier.
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
from scipy.stats import linregress, mannwhitneyu

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from engine.variable_loader import VariableDictionary

_EDIT = "--edit" in sys.argv
_vd = VariableDictionary()
DATA_DIR    = ROOT / "results" / "data"
ANCHORED    = ROOT / "results" / "06_event_anchored" / "data" / "event_anchored_scores.csv"
COX_CSV     = DATA_DIR / "hrs_cox_long_expanded.csv"
WIDE_CSV    = DATA_DIR / "hrs_analytic_wide_fullcohort.csv"
OUT_DIR     = ROOT / "results" / "CherryPicked2"

MIN_POINTS  = 3
LAG_MIN, LAG_MAX = -10, -2   # up to 2 years before event (reverse causation buffer)

print("=" * 68)
print("  EVENT-ANCHORED SQB SLOPE x CHARGE-AF RISK STRATIFICATION")
print(f"  Slope window: lags {LAG_MIN} to {LAG_MAX} years before event")
print("=" * 68)

# ── Load event-anchored scores ────────────────────────────────────────────────
df_anc = pd.read_csv(ANCHORED)
df_anc = df_anc[df_anc["lag"].between(LAG_MIN, LAG_MAX)].copy()
print(f"\n  Event-anchored data (lags {LAG_MIN} to {LAG_MAX}): {len(df_anc):,} rows")

CASE_LABEL = "Cases (incident arrhythmia)"
CTRL_LABEL = "Controls (never arrhythmia)"

# ── Compute per-participant SQB slope in the window ───────────────────────────
slope_rows = []
for pid, grp in df_anc.groupby("HHID_PN"):
    grp = grp.sort_values("lag")
    valid = grp["sleep"].dropna()
    lags  = grp.loc[valid.index, "lag"].values
    vals  = valid.values
    slope = np.nan
    if len(vals) >= MIN_POINTS:
        slope, *_ = linregress(lags, vals)
    slope_rows.append({
        "HHID_PN":   pid,
        "group":     grp["group"].iloc[0],
        "sleep_slope": slope,
    })

slopes = pd.DataFrame(slope_rows)
cases_slopes = slopes[slopes["group"] == CASE_LABEL].dropna(subset=["sleep_slope"])
ctrl_slopes  = slopes[slopes["group"] == CTRL_LABEL].dropna(subset=["sleep_slope"])

print(f"  Cases with valid slope (≥{MIN_POINTS} obs in window): n={len(cases_slopes):,}")
print(f"  Controls with valid slope: n={len(ctrl_slopes):,}")

# ── Load CHARGE-AF covariates to stratify controls ────────────────────────────
wide = pd.read_csv(WIDE_CSV)
cox  = pd.read_csv(COX_CSV)

# Get CHARGE-AF covariates at 2016 baseline for controls
covs = cox[["HHID_PN", "age_2016", "height_cm", "weight_kg",
             "hypertension", "diabetes"]].copy()
ctrl_slopes = ctrl_slopes.merge(covs, on="HHID_PN", how="left")

# Fit CHARGE-AF model on full Cox dataset; use its scale for z-scoring
_cont       = _vd.continuous_composites("HRS_2016", "charge_covariate")
_bin        = _vd.binary_composites("HRS_2016", "charge_covariate")
CHARGE_COVS = _cont + _bin
CHARGE_Z    = [c + "_z" for c in _cont] + _bin

cox_full = pd.read_csv(COX_CSV)
scale = {}
for col in ["age_2016", "height_cm", "weight_kg"]:
    mu, sd = cox_full[col].mean(), cox_full[col].std()
    scale[col] = (mu, sd)
    if sd > 0:
        cox_full[col + "_z"] = (cox_full[col] - mu) / sd

cox_cc = cox_full[CHARGE_Z + ["tstart", "tstop", "event"]].dropna()
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_cc, duration_col="tstop", event_col="event", entry_col="tstart",
        formula=" + ".join(CHARGE_Z))

# Apply same scale to controls
ctrl_cc = ctrl_slopes[CHARGE_COVS + ["HHID_PN", "sleep_slope"]].dropna().copy()
for col in ["age_2016", "height_cm", "weight_kg"]:
    mu, sd = scale[col]
    if sd > 0:
        ctrl_cc[col + "_z"] = (ctrl_cc[col] - mu) / sd

ctrl_cc["charge_lp"] = cph.predict_log_partial_hazard(ctrl_cc[CHARGE_Z])

# Stratify controls into tertiles
cuts = ctrl_cc["charge_lp"].quantile([1/3, 2/3]).values
ctrl_cc["risk_group"] = pd.cut(
    ctrl_cc["charge_lp"],
    bins=[-np.inf, cuts[0], cuts[1], np.inf],
    labels=["Low risk", "Medium risk", "High risk"]
)

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n  {'Group':<20} {'n':>6}  {'mean slope':>12}  {'median':>8}  {'SD':>8}")
print(f"  {'─'*60}")

group_data = {"Cases": cases_slopes["sleep_slope"].values}
for grp in ["Low risk", "Medium risk", "High risk"]:
    group_data[grp] = ctrl_cc[ctrl_cc["risk_group"] == grp]["sleep_slope"].values

for grp in ["Cases", "Low risk", "Medium risk", "High risk"]:
    d = group_data[grp]
    print(f"  {grp:<20} {len(d):>6}  {np.mean(d):>12.4f}  "
          f"{np.median(d):>8.4f}  {np.std(d):>8.4f}")

# ── MWU comparisons ───────────────────────────────────────────────────────────
print(f"\n  Mann-Whitney U: Cases vs each control tier")
for grp in ["Low risk", "Medium risk", "High risk"]:
    a, b = group_data["Cases"], group_data[grp]
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    r = (2*U)/(len(a)*len(b)) - 1
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  Cases vs {grp:<15}: U={U:.0f}  p={p:.4f}  r={r:.3f}  {sig}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

colors_grp = {"Cases": "#E07828", "Low risk": "#AEC6CF",
              "Medium risk": "#6A9FB5", "High risk": "#2C5F8A"}
order = ["Low risk", "Medium risk", "High risk", "Cases"]

for i, grp in enumerate(order):
    data = group_data[grp]
    bp = ax.boxplot(data, positions=[i], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors_grp[grp], alpha=0.7),
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", alpha=0.15, markersize=3),
                    whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))

ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(
    [f"{g}\n(n={len(group_data[g]):,})" for g in order], fontsize=11)
ax.set_ylabel("SQB Slope (score units / year)", fontsize=12)
ax.set_title(
    f"Event-Anchored SQB Slope by CHARGE-AF Risk Group\n"
    f"Slope computed from lags {LAG_MIN} to {LAG_MAX} years before arrhythmia onset "
    f"(HRS 2010–2022)",
    fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
patches = [mpatches.Patch(color=colors_grp[g], alpha=0.7, label=g) for g in order]
ax.legend(handles=patches, fontsize=9, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "HRS_slope_chargeaf_eventanchored.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close()
print(f"\n  Plot saved -> {out.name}")
print("=" * 68)
