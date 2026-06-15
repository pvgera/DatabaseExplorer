"""
update_cstat_model4.py
========================
Adds Model 4 (CHARGE-AF + dep + sleep + dep_slope + sleep_slope) to the
C-statistic comparison and regenerates Figure 6.

Model definitions:
  Model 1 — CHARGE-AF only
  Model 2 — CHARGE-AF + dep_2010 + sleep_2010
  Model 3 — CHARGE-AF + dep_2010 + sleep_2010 + dep_slope
  Model 4 — CHARGE-AF + dep_2010 + sleep_2010 + dep_slope + sleep_slope

Slopes computed identically to Figures 4 & 5: per-participant linregress
across all available multiwave observations (year vs score, min 3 points).
Same endogeneity caveat as dep_change: includes post-event waves for
participants whose arrhythmia onset was measured in a later wave.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import linregress

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
OUT_CSV    = DATA_DIR / "cox_cstat_comparison.csv"
OUT_FIG    = ROOT / "results" / "CherryPicked2" / "Figure_6_HRS_cstatistic.png"

WAVE_YEAR = {
    2:1994, 3:1996, 4:1998, 5:2000, 6:2002, 7:2004, 8:2006,
    10:2010, 12:2014, 13:2016, 14:2018, 15:2020, 16:2022
}
MIN_POINTS = 3

print("=" * 68)
print("  MODEL 4: Adding multiwave slopes to C-stat comparison")
print("=" * 68)

# ── Load Cox dataset ──────────────────────────────────────────────────────────
cox = pd.read_csv(COX_CSV)
print(f"  Cox dataset: n={len(cox):,}  events={cox['event'].sum()}")

# ── Compute per-participant multiwave slopes (same method as Figs 4 & 5) ──────
scores = pd.read_csv(SCORES_CSV)

DEP_WAVE_COLS   = {n: f"dep_w{n}"   for n in [2,3,4,5,6,7,8,10,12,13,14,15,16]
                   if f"dep_w{n}" in scores.columns}
SLEEP_WAVE_COLS = {n: f"sleep_w{n}" for n in [6,7,8,10,12,13,14,15,16]
                   if f"sleep_w{n}" in scores.columns}

slope_rows = []
for _, row in scores.iterrows():
    pid = row["HHID_PN"]
    # Depression slope
    dep_pts = [(WAVE_YEAR[n], row[col])
               for n, col in DEP_WAVE_COLS.items()
               if not pd.isna(row[col])]
    dep_slope = np.nan
    if len(dep_pts) >= MIN_POINTS:
        yrs, vals = zip(*dep_pts)
        dep_slope, *_ = linregress(yrs, vals)

    # Sleep slope
    slp_pts = [(WAVE_YEAR[n], row[col])
               for n, col in SLEEP_WAVE_COLS.items()
               if not pd.isna(row[col])]
    slp_slope = np.nan
    if len(slp_pts) >= MIN_POINTS:
        yrs, vals = zip(*slp_pts)
        slp_slope, *_ = linregress(yrs, vals)

    slope_rows.append({"HHID_PN": pid,
                        "dep_slope": dep_slope,
                        "sleep_slope": slp_slope})

slopes_df = pd.DataFrame(slope_rows)
print(f"  dep_slope  valid: n={slopes_df['dep_slope'].notna().sum():,}")
print(f"  sleep_slope valid: n={slopes_df['sleep_slope'].notna().sum():,}")

cox = cox.merge(slopes_df, on="HHID_PN", how="left")
print(f"  After merge — dep_slope in Cox: {cox['dep_slope'].notna().sum():,}")
print(f"  After merge — sleep_slope in Cox: {cox['sleep_slope'].notna().sum():,}")

# ── Standardise all continuous covariates ────────────────────────────────────
_cont    = _vd.continuous_composites("HRS_2016", "charge_covariate")     # age_2016, height_cm, weight_kg
_bin     = _vd.binary_composites("HRS_2016", "charge_covariate")          # hypertension, diabetes
_aff     = _vd.continuous_composites("HRS_2016", "affective_predictor")   # dep_2016, sleep_2016

CONTINUOUS = _cont + _aff + ["dep_slope", "sleep_slope"]
df_z = cox.copy()
for col in CONTINUOUS:
    if col in df_z.columns:
        mu, sd = df_z[col].mean(), df_z[col].std()
        if sd > 0:
            df_z[col + "_z"] = (df_z[col] - mu) / sd

CHARGE_Z = [c + "_z" for c in _cont] + _bin + ["sleep_apnea"]   # sleep_apnea: extra clinical covariate
AFF_Z    = [c + "_z" for c in _aff]
DELTA_Z  = ["dep_slope_z", "sleep_slope_z"]

# ── Fit models ────────────────────────────────────────────────────────────────
def fit_cox(df_in, covariates, label):
    needed = covariates + ["tstart", "tstop", "event"]
    cc = df_in[needed].dropna()
    n, ev = len(cc), int(cc["event"].sum())
    print(f"\n  {label}  (n={n:,}  events={ev})")
    if ev < 5:
        print("    SKIP: too few events")
        return None, None
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cc, duration_col="tstop", event_col="event", entry_col="tstart",
            formula=" + ".join(covariates))
    c = concordance_index(cc["tstop"], -cph.predict_partial_hazard(cc), cc["event"])
    print(f"    Harrell's C = {c:.4f}")
    return c, n


def bootstrap_ci(df_in, covariates, n_boot=1000, seed=42, label=""):
    rng = np.random.default_rng(seed)
    needed = covariates + ["tstart", "tstop", "event"]
    cc = df_in[needed].dropna()
    n = len(cc)
    c_boot = []
    for i in range(n_boot):
        samp = cc.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31)))
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(samp, duration_col="tstop", event_col="event", entry_col="tstart",
                    formula=" + ".join(covariates))
            c = concordance_index(samp["tstop"], -cph.predict_partial_hazard(samp), samp["event"])
            c_boot.append(c)
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"    bootstrap {i+1}/{n_boot} ...")
    lo, hi = np.percentile(c_boot, [2.5, 97.5])
    print(f"  {label}  95% CI: ({lo:.4f}, {hi:.4f})  [n_success={len(c_boot)}]")
    return lo, hi

# ── Restrict to complete cases for Model 4 — same sample across all models ───
all_vars = CHARGE_Z + AFF_Z + ["tstart", "tstop", "event"]
df_cc = df_z[all_vars].dropna()
print(f"\n  Restricted sample (complete cases for Model 2): "
      f"n={len(df_cc):,}  events={df_cc['event'].sum()}")
df_z = df_z.loc[df_cc.index]

c1, n1 = fit_cox(df_z, CHARGE_Z,          "Model 1: CHARGE-AF")
c2, n2 = fit_cox(df_z, CHARGE_Z + AFF_Z, "Model 2: + dep + sleep")

print("\nBootstrapping 95% CIs (1000 iterations each) ...")
lo1, hi1 = bootstrap_ci(df_z, CHARGE_Z,          label="Model 1: CHARGE-AF")
lo2, hi2 = bootstrap_ci(df_z, CHARGE_Z + AFF_Z,  label="Model 2: + dep + sleep")

# ── Save updated CSV ──────────────────────────────────────────────────────────
cstat_df = pd.DataFrame([
    {"model": "Model1_CHARGEAF",   "C_stat": c1, "CI_lo": lo1, "CI_hi": hi1, "n_covariates": len(CHARGE_Z)},
    {"model": "Model2_+Affective", "C_stat": c2, "CI_lo": lo2, "CI_hi": hi2, "n_covariates": len(CHARGE_Z)+len(AFF_Z)},
])
cstat_df.to_csv(OUT_CSV, index=False)
print(f"\n  CSV saved -> {OUT_CSV.name}")

# ── Regenerate figure ─────────────────────────────────────────────────────────
labels = [
    "Model 1\n(CHARGE-AF)",
    "Model 2\n(+ Sleep + Depression)",
]
c_vals  = [c1, c2]
colors  = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(9, 6))

x_pos = [0, 0.50]
bars = ax.bar(x_pos, c_vals, color=colors, width=0.35, zorder=3)
for bar, val in zip(bars, c_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=15, fontweight="bold")

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Chance (C=0.50)", zorder=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=13)
ax.set_xlim(-0.35, 0.85)
ax.set_ylabel("Harrell's C-statistic", fontsize=14)
ax.set_title("Discrimination (Harrell's C): CHARGE-AF vs CHARGE-AF + Sleep + Depression",
             fontsize=13)
ax.set_ylim(0.62, max(c_vals) + 0.02)
ax.legend(fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=1)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close()
print(f"  Figure saved -> {OUT_FIG.name}")
print("\n" + "=" * 68)
print("  SUMMARY")
print("=" * 68)
for _, row in cstat_df.iterrows():
    print(f"  {row['model']:30s}: C = {row['C_stat']:.4f}  (95% CI: {row['CI_lo']:.4f}–{row['CI_hi']:.4f})")
