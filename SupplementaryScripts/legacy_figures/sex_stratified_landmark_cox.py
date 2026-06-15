"""
sex_stratified_landmark_cox.py
================================
Sex-stratified landmark Cox regression for SQB predicting arrhythmia
at the 4-year and 6-year baselines (where the overall effect was significant).

Tests:
  1. Stratified models: Cox run separately for males and females
  2. Interaction model: sleep × sex interaction term at each baseline
  3. Forest plot output comparing male vs female HRs

Baselines:
  4yr: landmark_year = 2018, wave 14
  6yr: landmark_year = 2016, wave 13
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

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from engine.variable_loader import VariableDictionary

_EDIT = "--edit" in sys.argv
_vd = VariableDictionary()
DATA_OUT = ROOT / "results" / "CherryPicked2"
DATA_OUT.mkdir(parents=True, exist_ok=True)

WIDE_CSV   = ROOT / "results" / "data"  / "hrs_analytic_wide_fullcohort.csv"
SCORES_CSV = ROOT / "results" / "05_timelag" / "data" / "multiwave_scores.csv"

WAVE_YEAR = {14: 2018}
LAGS      = {4: 14}   # lag_years: wave_n

C_SLEEP  = "#E07828"
C_MALE   = "#2E7DD1"
C_FEMALE = "#E84B8A"

print("=" * 68)
print("  SEX-STRATIFIED LANDMARK COX  —  4yr & 6yr baselines")
print("=" * 68)

# ── Load cohort ───────────────────────────────────────────────────────────────
df_wide   = pd.read_csv(WIDE_CSV).rename(columns={"HHID_PN": "hhidpn"})
df_scores = pd.read_csv(SCORES_CSV).rename(columns={"HHID_PN": "hhidpn"})

cohort = df_wide.merge(df_scores, on="hhidpn", how="left")
print(f"  Cohort: n={len(cohort):,}")
print(f"  Female: {(cohort['sex_female']==1).sum():,}  Male: {(cohort['sex_female']==0).sum():,}")

def _cat_num(s):
    return pd.to_numeric(s.astype(str).str.split(".").str[0], errors="coerce")

def get_cov(df, wave_n, suffix):
    col = f"r{wave_n}{suffix}"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    if suffix in ("hibpe", "diabe"):
        s = _cat_num(df[col])
        return s.where(s.isin([0, 1]))
    return pd.to_numeric(df[col], errors="coerce")

# Load RAND HRS covariates for waves 13 and 14 from the Stata file if available,
# otherwise fall back to wide_with_intermediate_waves for age/bmi/covariates
WIDE_INT = ROOT / "results" / "05_timelag" / "data" / "wide_with_intermediate_waves.csv"
df_int = pd.read_csv(WIDE_INT).rename(columns={"HHID_PN": "hhidpn"})

# Pull age and covariates from existing wide dataset (2016 baseline)
_cont_cov = _vd.continuous_composites("HRS_2016", "charge_covariate")   # age_2016, height_cm, weight_kg
_bin_cov  = _vd.binary_composites("HRS_2016", "charge_covariate")        # hypertension, diabetes
cov_cols  = ["hhidpn"] + _cont_cov + ["bmi_2016"] + _bin_cov + ["sex_female", "sleep_apnea"]
available = [c for c in cov_cols if c in df_int.columns]
df_covs = df_int[available].copy()

cohort = cohort.merge(df_covs, on="hhidpn", how="left", suffixes=("", "_int"))

# Resolve sex_female — prefer the one from df_wide
if "sex_female_int" in cohort.columns:
    cohort["sex_female"] = cohort["sex_female"].combine_first(cohort["sex_female_int"])
    cohort.drop(columns=["sex_female_int"], inplace=True)

print(f"  Covariates merged. BMI valid: {cohort['bmi_2016'].notna().sum():,}")

all_results = []

for lag_yr, wave_n in LAGS.items():
    lm_year = 2022 - lag_yr
    sleep_col = f"sleep_w{wave_n}"

    print(f"\n{'─'*68}")
    print(f"  Lag {lag_yr}yr  (baseline {lm_year}, wave {wave_n})")
    print(f"{'─'*68}")

    if sleep_col not in cohort.columns:
        print(f"  SKIP: {sleep_col} not in cohort")
        continue

    # Eligible: AFib-free at baseline
    elig = cohort[
        cohort["afib_onset_wave"].isna() | (cohort["afib_onset_wave"] > lm_year)
    ].copy()

    has_onset     = elig["afib_onset_wave"].notna()
    elig["tstop"] = (elig["afib_onset_wave"].where(has_onset, 2022.0) - lm_year).clip(lower=0.01)
    elig["event"] = (has_onset & (elig["afib_onset_wave"] > lm_year)).astype(int)
    elig["sleep_lm"] = elig[sleep_col]

    # Use 2016 covariates as proxy (closest available pre-computed)
    elig["age_lm"]   = elig["age_2016"]
    elig["bmi_lm"]   = elig["bmi_2016"]
    elig["hibpe_lm"] = elig["hypertension"]
    elig["diabe_lm"] = elig["diabetes"]

    # Z-score within eligible sample
    for col in ["sleep_lm", "age_lm", "bmi_lm"]:
        mu, sd = elig[col].mean(), elig[col].std()
        if sd > 0:
            elig[col + "_z"] = (elig[col] - mu) / sd

    needed = ["sleep_lm_z", "age_lm_z", "bmi_lm_z",
              "hibpe_lm", "diabe_lm", "sleep_apnea", "tstop", "event", "sex_female"]
    cc_all = elig[needed].dropna()

    print(f"  Complete cases: n={len(cc_all):,}  events={cc_all['event'].sum()}")

    # ── 1. Sex-stratified models ──────────────────────────────────────────────
    for sex_label, sex_val, col in [("Female", 1.0, C_FEMALE), ("Male", 0.0, C_MALE)]:
        sub = cc_all[cc_all["sex_female"] == sex_val].copy()
        n, ev = len(sub), int(sub["event"].sum())

        if ev < 5:
            print(f"  SKIP {sex_label}: events={ev}")
            continue

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(sub[["sleep_lm_z","age_lm_z","bmi_lm_z","hibpe_lm","diabe_lm","sleep_apnea","tstop","event"]],
                duration_col="tstop", event_col="event",
                formula="sleep_lm_z + age_lm_z + bmi_lm_z + hibpe_lm + diabe_lm + sleep_apnea")
        r  = cph.summary.loc["sleep_lm_z"]
        hr = np.exp(r["coef"]); lo = np.exp(r["coef lower 95%"]); hi = np.exp(r["coef upper 95%"])
        p  = r["p"]
        sig = "***" if p < 0.05 else ("~" if p < 0.10 else "ns")
        print(f"  {sex_label:8s}: HR={hr:.3f} [{lo:.3f}–{hi:.3f}]  p={p:.4f}  {sig}  n={n:,}  ev={ev}")
        all_results.append({"lag": lag_yr, "lm_year": lm_year, "sex": sex_label,
                            "HR": round(hr,3), "CI_lo": round(lo,3), "CI_hi": round(hi,3),
                            "p": round(p,4), "n": n, "events": ev})

    # ── 2. Interaction model: sleep × sex ────────────────────────────────────
    cc_int = cc_all.copy()
    cc_int["sleep_x_sex"] = cc_int["sleep_lm_z"] * cc_int["sex_female"]
    int_cols = ["sleep_lm_z","sex_female","sleep_x_sex",
                "age_lm_z","bmi_lm_z","hibpe_lm","diabe_lm","sleep_apnea","tstop","event"]
    sub_int = cc_int[int_cols].dropna()

    cph_int = CoxPHFitter(penalizer=0.01)
    cph_int.fit(sub_int, duration_col="tstop", event_col="event",
                formula="sleep_lm_z + sex_female + sleep_x_sex + age_lm_z + bmi_lm_z + hibpe_lm + diabe_lm + sleep_apnea")
    r_int  = cph_int.summary.loc["sleep_x_sex"]
    hr_int = np.exp(r_int["coef"])
    lo_int = np.exp(r_int["coef lower 95%"])
    hi_int = np.exp(r_int["coef upper 95%"])
    p_int  = r_int["p"]
    sig_int = "SIGNIFICANT" if p_int < 0.05 else ("marginal" if p_int < 0.10 else "ns")
    print(f"  Sleep×Sex interaction: HR={hr_int:.3f} [{lo_int:.3f}–{hi_int:.3f}]  "
          f"p={p_int:.4f}  → {sig_int}")

# ── Forest plot ───────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_results)
print(f"\n{'─'*68}")
print("  RESULTS SUMMARY")
print(f"{'─'*68}")
print(df_res.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 2.2))
y_pos = {"Female_4": 1.0, "Male_4": 0.5}

for _, row in df_res.iterrows():
    key = f"{row['sex']}_{row['lag']}"
    y   = y_pos.get(key, 0)
    col = C_FEMALE if row["sex"] == "Female" else C_MALE
    ax.plot([row["CI_lo"], row["CI_hi"]], [y, y], color=col, lw=2.5, alpha=0.85)
    ax.scatter([row["HR"]], [y], color=col, s=90, zorder=3, marker="D")
    sig = "  ***" if row["p"] < 0.05 else ("  ~" if row["p"] < 0.10 else "")
    ax.text(max(row["CI_hi"], 1.3) + 0.04, y,
            f"HR={row['HR']:.3f} [{row['CI_lo']:.3f}–{row['CI_hi']:.3f}]  "
            f"p={row['p']:.3f}{sig}  (n={row['n']:,})",
            va="center", ha="left", fontsize=9,
            color=col)

ax.axvline(1.0, color="black", lw=1.2, linestyle="--")
ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(["Female — 4yr baseline", "Male — 4yr baseline"], fontsize=10)
ax.set_xlabel("Hazard Ratio for Arrhythmia per SD Sleep Quality Score", fontsize=10)
ax.set_title("Sex-Stratified Landmark Cox: SQB → Arrhythmia\n"
             "4-year baseline, adjusted for age, BMI, hypertension, diabetes, sleep apnea",
             fontsize=10)
ax.set_xlim(0.5, 3.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
f_patch = mpatches.Patch(color=C_FEMALE, label="Female")
m_patch = mpatches.Patch(color=C_MALE,   label="Male")
ax.legend(handles=[f_patch, m_patch], fontsize=9, loc="lower right")
plt.tight_layout()
out_path = DATA_OUT / "Figure_5_HRS_sex_landmark_cox.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close()
print(f"\n  Figure saved -> {out_path}")
print("=" * 68)
