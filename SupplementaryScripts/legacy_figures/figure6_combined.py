"""
figure6_combined.py
====================
Combined Figure 6:
  a  Harrell's C-statistic: CHARGE-AF vs CHARGE-AF + SQB + DB (2016 baseline)
  b  Event-anchored SQB slope by CHARGE-AF risk group (lags -10 to -2)
  c  Event-anchored DB slope by CHARGE-AF risk group (lags -10 to -2)
     Both b and c show mean ± 95% CI as horizontal error bars.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import linregress, mannwhitneyu, sem

_EDIT = "--edit" in sys.argv

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "results" / "data"
ANCHORED    = ROOT / "results" / "06_event_anchored" / "data" / "event_anchored_scores.csv"
SCORES_CSV  = ROOT / "results" / "05_timelag" / "data" / "multiwave_scores.csv"
COX_CSV     = DATA_DIR / "hrs_cox_long_expanded.csv"
OUT_DIR     = ROOT / "results" / "CherryPicked2"

WAVE_YEAR   = {2:1994,3:1996,4:1998,5:2000,6:2002,7:2004,8:2006,
               10:2010,12:2014,13:2016,14:2018,15:2020,16:2022}
MIN_POINTS  = 3
LAG_MIN, LAG_MAX = -10, -2

C_CHARGEAF  = "#4C72B0"
C_AUGMENTED = "#5BA4C4"
C_CASES_SQB = "#E07828"
C_CASES_DB  = "#7B5EA7"
COLORS_GRP  = {"Low risk": "#AAAAAA", "Medium risk": "#777777",
               "High risk": "#333333", "Arrhythmia cases": None}  # Cases colour set per panel
ORDER_PLOT  = ["Low risk", "Medium risk", "High risk", "Arrhythmia cases"]  # Cases ends up at top after invert

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A DATA — C-stat comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Computing C-statistics ...")
cox    = pd.read_csv(COX_CSV)
scores = pd.read_csv(SCORES_CSV)

SLEEP_WAVE_COLS = {n: f"sleep_w{n}" for n in [6,7,8,10,12,13,14,15,16]
                   if f"sleep_w{n}" in scores.columns}
slope_rows = []
for _, row in scores.iterrows():
    pts = [(WAVE_YEAR[n], row[col]) for n,col in SLEEP_WAVE_COLS.items()
           if not pd.isna(row[col])]
    sl = np.nan
    if len(pts) >= MIN_POINTS:
        yrs, vals = zip(*pts)
        sl, *_ = linregress(yrs, vals)
    slope_rows.append({"HHID_PN": row["HHID_PN"], "sleep_slope": sl})

cox = cox.merge(pd.DataFrame(slope_rows), on="HHID_PN", how="left")

scale = {}
for col in ["age_2016", "height_cm", "weight_kg"]:
    mu, sd = cox[col].mean(), cox[col].std()
    scale[col] = (mu, sd)
    if sd > 0:
        cox[col + "_z"] = (cox[col] - mu) / sd
for col in ["dep_2016", "sleep_2016", "sleep_slope"]:
    mu, sd = cox[col].mean(), cox[col].std()
    if sd > 0:
        cox[col + "_z"] = (cox[col] - mu) / sd

CHARGE_Z = ["age_2016_z", "height_cm_z", "weight_kg_z", "hypertension", "diabetes", "sleep_apnea"]
AFF_Z    = ["dep_2016_z", "sleep_2016_z"]

all_vars = CHARGE_Z + AFF_Z + ["tstart", "tstop", "event"]
df_cc    = cox[all_vars].dropna()
cox_cc   = cox.loc[df_cc.index].copy()

def cstat(df, covs):
    cc = df[covs + ["tstart", "tstop", "event"]].dropna()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cc, duration_col="tstop", event_col="event", entry_col="tstart",
            formula=" + ".join(covs))
    return concordance_index(cc["tstop"], -cph.predict_partial_hazard(cc), cc["event"])

c1 = cstat(cox_cc, CHARGE_Z)
c2 = cstat(cox_cc, CHARGE_Z + AFF_Z)
print(f"  CHARGE-AF: C={c1:.4f}   + SQB + DB: C={c2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PANELS B & C DATA — event-anchored slopes by CHARGE-AF risk group
# ══════════════════════════════════════════════════════════════════════════════
print("Computing event-anchored slopes ...")
df_anc = pd.read_csv(ANCHORED)
df_anc = df_anc[df_anc["lag"].between(LAG_MIN, LAG_MAX)].copy()

CASE_LABEL = "Cases (incident arrhythmia)"
CTRL_LABEL = "Controls (never arrhythmia)"

# Per-participant slopes for sleep (SQB) and depression (DB)
ea_rows = []
for pid, grp in df_anc.groupby("HHID_PN"):
    grp = grp.sort_values("lag")
    for measure, col in [("sleep_slope", "sleep"), ("dep_slope", "dep")]:
        valid = grp[col].dropna()
        lags  = grp.loc[valid.index, "lag"].values
        vals  = valid.values
        sl    = np.nan
        if len(vals) >= MIN_POINTS:
            sl, *_ = linregress(lags, vals)
        ea_rows.append({"HHID_PN": pid, "group": grp["group"].iloc[0],
                        "measure": measure, "slope": sl})

ea_df = pd.DataFrame(ea_rows)

def get_slopes(measure):
    sub = ea_df[ea_df["measure"] == measure]
    cases = sub[sub["group"] == CASE_LABEL].dropna(subset=["slope"])
    ctrl  = sub[sub["group"] == CTRL_LABEL].dropna(subset=["slope"])
    return cases, ctrl

cases_sqb, ctrl_sqb = get_slopes("sleep_slope")
cases_db,  ctrl_db  = get_slopes("dep_slope")

# Stratify controls by CHARGE-AF linear predictor (built once, reused for both)
covs_df = pd.read_csv(COX_CSV)[["HHID_PN","age_2016","height_cm",
                                  "weight_kg","hypertension","diabetes","sleep_apnea"]].copy()

def stratify_controls(ctrl_slopes):
    ctrl = ctrl_slopes.merge(covs_df, on="HHID_PN", how="left")
    ctrl_cc = ctrl[["HHID_PN","slope","age_2016","height_cm",
                    "weight_kg","hypertension","diabetes","sleep_apnea"]].dropna().copy()
    for col in ["age_2016","height_cm","weight_kg"]:
        mu, sd = scale[col]
        if sd > 0:
            ctrl_cc[col+"_z"] = (ctrl_cc[col]-mu)/sd

    cox_base = pd.read_csv(COX_CSV).copy()
    for col in ["age_2016","height_cm","weight_kg"]:
        mu, sd = scale[col]
        if sd > 0:
            cox_base[col+"_z"] = (cox_base[col]-mu)/sd
    cox_base_cc = cox_base[CHARGE_Z+["tstart","tstop","event"]].dropna()
    cph_base = CoxPHFitter(penalizer=0.01)
    cph_base.fit(cox_base_cc, duration_col="tstop", event_col="event",
                 entry_col="tstart", formula=" + ".join(CHARGE_Z))

    ctrl_cc["charge_lp"] = cph_base.predict_log_partial_hazard(ctrl_cc[CHARGE_Z])
    cuts = ctrl_cc["charge_lp"].quantile([1/3, 2/3]).values
    ctrl_cc["risk_group"] = pd.cut(ctrl_cc["charge_lp"],
                                    bins=[-np.inf, cuts[0], cuts[1], np.inf],
                                    labels=["Low risk","Medium risk","High risk"])
    return ctrl_cc

ctrl_sqb_strat = stratify_controls(ctrl_sqb)
ctrl_db_strat  = stratify_controls(ctrl_db)

def build_group_stats(cases_slopes, ctrl_strat, slope_col="slope"):
    """Return dict of group → (mean, ci_lo, ci_hi, n) and cases array."""
    result = {}
    case_vals = cases_slopes[slope_col].values
    n = len(case_vals)
    m = np.mean(case_vals)
    ci = 1.96 * sem(case_vals)
    result["Arrhythmia cases"] = (m, m - ci, m + ci, n)
    for grp in ["Low risk", "Medium risk", "High risk"]:
        vals = ctrl_strat[ctrl_strat["risk_group"] == grp]["slope"].values
        n = len(vals)
        m = np.mean(vals)
        ci = 1.96 * sem(vals)
        result[grp] = (m, m - ci, m + ci, n)
    return result, case_vals

stats_sqb, cases_sqb_arr = build_group_stats(cases_sqb, ctrl_sqb_strat)
stats_db,  cases_db_arr  = build_group_stats(cases_db,  ctrl_db_strat)

# MWU: cases vs high-risk
def mwu_sig(a, b):
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    label = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    return p, label

p_sqb, sig_sqb = mwu_sig(cases_sqb_arr, ctrl_sqb_strat[ctrl_sqb_strat["risk_group"]=="High risk"]["slope"].values)
p_db,  sig_db  = mwu_sig(cases_db_arr,  ctrl_db_strat[ctrl_db_strat["risk_group"]=="Low risk"]["slope"].values)
print(f"  SQB Cases vs High risk: p={p_sqb:.4f}  {sig_sqb}")
print(f"  DB  Cases vs Low risk:  p={p_db:.4f}  {sig_db}")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                          gridspec_kw={"width_ratios": [1, 1.4, 1.4]})

# ── Panel a: C-stat bars ──────────────────────────────────────────────────────
ax = axes[0]
x_pos  = [0, 0.50]
c_vals = [c1, c2]
colors = [C_CHARGEAF, C_AUGMENTED]
bars   = ax.bar(x_pos, c_vals, color=colors, width=0.35, zorder=3)
for bar, val in zip(bars, c_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.axhline(0.5, color="gray", linestyle=":", lw=1.2, label="Chance (C=0.50)", zorder=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(["Model 1\n(CHARGE-AF)",
                     "Model 2\n(+ Sleep + Depression)"], fontsize=11)
ax.set_xlim(-0.35, 0.85)
ax.set_ylim(0.62, max(c_vals) + 0.025)
ax.set_ylabel("Harrell's C-statistic", fontsize=12)
ax.set_title("Discrimination (Harrell's C)", fontsize=12)
ax.legend(fontsize=9)
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=1)
ax.text(-0.12, 1.02, "a", transform=ax.transAxes,
        fontsize=15, fontweight="bold", va="center")

# ── Helper: draw horizontal CI panel ─────────────────────────────────────────
def draw_ci_panel(ax, stats, cases_color, title, xlabel, p_val, sig_label, panel_letter,
                  bracket_group="High risk"):
    y_positions = {grp: i for i, grp in enumerate(ORDER_PLOT)}

    for grp in ORDER_PLOT:
        mean, lo, hi, n = stats[grp]
        y = y_positions[grp]
        col = cases_color if grp == "Arrhythmia cases" else COLORS_GRP[grp]
        mk  = "D" if grp == "Arrhythmia cases" else "o"
        ms  = 9  if grp == "Arrhythmia cases" else 7
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                    fmt=mk, color=col, markersize=ms,
                    capsize=4, capthick=1.5, elinewidth=1.5, zorder=3)
        ax.text(hi + 0.015, y, f"{mean:.2f}  n={n:,}",
                va="center", ha="left", fontsize=8.5, color=col)

    # Significance bracket: Cases vs bracket_group
    x_max = max(stats[g][2] for g in ORDER_PLOT) + 0.18
    y0, y1 = y_positions["Arrhythmia cases"], y_positions[bracket_group]
    ax.plot([x_max, x_max + 0.04, x_max + 0.04, x_max],
            [y0, y0, y1, y1], color="black", lw=1.2, clip_on=False)
    ax.text(x_max + 0.06, (y0 + y1) / 2,
            f"{sig_label}\np={p_val:.3f}", va="center", ha="left", fontsize=9,
            clip_on=False)

    ax.axvline(0, color="gray", linestyle="--", lw=1, alpha=0.6)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(ORDER_PLOT, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.14, 1.02, panel_letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="center")

draw_ci_panel(axes[1], stats_sqb, C_CASES_SQB,
              "Event-Anchored SQB Slope\nby CHARGE-AF Risk Group",
              "SQB Slope — Mean ± 95% CI\n(score units / year)",
              p_sqb, sig_sqb, "b")

draw_ci_panel(axes[2], stats_db, C_CASES_DB,
              "Event-Anchored DB Slope\nby CHARGE-AF Risk Group",
              "DB Slope — Mean ± 95% CI\n(score units / year)",
              p_db, sig_db, "c", bracket_group="Low risk")

fig.suptitle("CHARGE-AF Risk Stratification with Sleep & Depression Burden Surrogates\n"
             "HRS 2010–2022, ages 50+", fontsize=13, fontweight="bold")
plt.tight_layout()
out = OUT_DIR / "Figure_6_HRS_cstatistic.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close()
print(f"\n  Figure saved -> {out.name}")
