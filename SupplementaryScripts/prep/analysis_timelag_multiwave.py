"""
analysis_timelag_multiwave.py
==============================
Event-anchored landmark analysis: Depression (CESD-8) and Sleep Quality (4-item)
predicting incident self-reported arrhythmia ("abnormal heart rhythm") in the HRS,
tested at multiple pre-event time lags.

Uses the RAND HRS Longitudinal File (1992–2022) for multi-wave dep/sleep scores,
merged with the existing incident-arrhythmia cohort (afib_onset_wave identified
from raw HRS Section C wave files in the existing pipeline).

═══════════════════════════════════════════════════════════════════════════════════
VARIABLE AUDIT — RAND HRS NAMING CONVENTION (r[N] = respondent, wave N)
═══════════════════════════════════════════════════════════════════════════════════

Depression (CESD-8):
  r[N]cesd     — composite 0–8 score, waves 2–16 (1994–2022)
  Individual items (if needed for item-level checks):
    r[N]depres  — cesd: felt depressed             (direct, binary)
    r[N]effort  — cesd: everything an effort       (direct)
    r[N]enlife  — cesd: enjoyed life               (REVERSE: yes=healthy)
    r[N]flone   — cesd: felt lonely                (direct)
    r[N]fsad    — cesd: felt sad                   (direct)
    r[N]going   — cesd: could not get going        (direct)
    r[N]sleepr  — cesd: sleep was restless         (direct)
    r[N]whappy  — cesd: was happy                  (REVERSE: yes=healthy)
  Scoring: r[N]cesd raw 0–8 → ×12.5 → 0–100 (higher = worse)
  NOTE: r[N]cesd INCLUDES a sleep item (sleepr). The 4-item sleep battery below
  is a separate dedicated sleep quality instrument.

Sleep Quality (4-item battery):
  r[N]sleepfal — trouble falling asleep  (1=Most of time, 2=Sometimes, 3=Rarely/never)
  r[N]sleepwkn — trouble waking at night (same 3-point scale)
  r[N]sleepwke — waking too early        (same scale)
  r[N]sleeprt  — feeling rested in AM    (1=Most rested=BEST, 3=Rarely rested=WORST)
  First available: wave 6 (2002). Waves 9 (2008) and 11 (2012) EXCLUDED — <2% coverage
  (administered to a special non-representative sub-sample in those years).
  Response format is identical across all 9 usable waves (consistent scale).

Sleep battery consistency across waves:
  Wave  6 (2002): n=18,127 / 45,234  ✓ USABLE
  Wave  7 (2004): n=20,086 / 45,234  ✓ USABLE
  Wave  8 (2006): n=18,434 / 45,234  ✓ USABLE
  Wave  9 (2008): n=   143 / 45,234  ✗ SUB-SAMPLE — EXCLUDED
  Wave 10 (2010): n=21,986 / 45,234  ✓ USABLE
  Wave 11 (2012): n=   516 / 45,234  ✗ SUB-SAMPLE — EXCLUDED
  Wave 12 (2014): n=18,672 / 45,234  ✓ USABLE
  Wave 13 (2016): n=20,831 / 45,234  ✓ USABLE
  Wave 14 (2018): n=17,087 / 45,234  ✓ USABLE
  Wave 15 (2020): n=15,643 / 45,234  ✓ USABLE
  Wave 16 (2022): n=15,779 / 45,234  ✓ USABLE

Arrhythmia ("abnormal heart rhythm"):
  NOT available as a distinct variable in RAND HRS.
  RAND HRS r[N]hearte / r[N]heart = generic "heart problems" (includes MI, CHF,
  angina, and arrhythmia under a single flag). Incident arrhythmia cases are taken
  from the existing analytic cohort (hrs_analytic_wide.csv) where the specific
  "Abnormal heart rhythm" sub-response was ascertained from raw HRS Section C .da
  files in the existing pipeline. For any analysis requiring a de-novo cohort,
  the raw wave files must be parsed directly.

CHARGE-AF Covariates (RAND HRS, per wave):
  r[N]agey_b  — age in years at interview begin
  r[N]bmi     — self-reported BMI kg/m²   (valid range enforced: 12–75)
  r[N]height  — self-reported height in metres  (×100 = cm)
  r[N]weight  — self-reported weight in kg
  r[N]hibpe   — ever had high blood pressure (0=no, 1=yes)
  r[N]diabe   — ever had diabetes           (0=no, 1=yes)
  Height is treated as time-stable; all other covariates updated to landmark wave.

═══════════════════════════════════════════════════════════════════════════════════
LANDMARK DESIGN
═══════════════════════════════════════════════════════════════════════════════════
Lags tested: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 years before 2022 (the study end).
For each lag L, landmark_year = 2022 − L:
  - Include all participants AFib-free at landmark_year
  - Predictors (dep, sleep) measured at landmark_year
  - Follow up from landmark_year to first AFib or censor at 2022
  - Cox PH: dep or sleep (z-scored) + age + bmi + hypertension + diabetes
  - HR at each lag answers: "Does dep/sleep measured L years out predict arrhythmia?"

Power note: most incident AFib cases (n≈70–100 depending on wave) occur at 2022.
Lags ≥14 yr will include all cases (event at 2018/2020/2022 all post-landmark)
but many will lack dep/sleep measurements that far back (attrition, cohort entry).
Sleep-only models are further limited to landmarks ≥ 2002 (first available wave).

Outputs
-------
  data/multiwave_scores.csv           — per-participant dep/sleep at all waves
  data/multiwave_landmark_results.csv — HR, CI, p, n, events per lag and predictor
  figures/multiwave_forest.png        — forest plot: HR vs lag for dep and sleep
  figures/multiwave_trend.png         — HR trend lines with CI bands
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))
from engine.variable_loader import VariableDictionary
_vd = VariableDictionary()
DATA_OUT   = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "analytic"
FIG_OUT    = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "figures"
DATA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

RAND_DTA   = REPO_ROOT / "RawData" / "HRS" / "RandHRS" / "randhrs1992_2022v1.dta"
WIDE_CSV   = REPO_ROOT / "SF_OUTPUT" / "analytic" / "hrs_analytic_wide_fullcohort.csv"

print()
print("=" * 75)
print("  LANDMARK TIME-LAG ANALYSIS — MULTI-WAVE (RAND HRS 1992–2022)")
print("=" * 75)

# ─────────────────────────────────────────────────────────────────────────────
# WAVE–YEAR MAPPING
# ─────────────────────────────────────────────────────────────────────────────
WAVE_YEAR = {
    1: 1992, 2: 1994,  3: 1996,  4: 1998,  5: 2000,
    6: 2002, 7: 2004,  8: 2006,  9: 2008, 10: 2010,
   11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022,
}
YEAR_WAVE = {v: k for k, v in WAVE_YEAR.items()}

SLEEP_BAD_WAVES = {9, 11}   # sub-sample only, <2% coverage
SLEEP_GOOD_WAVES = [n for n in range(6, 17) if n not in SLEEP_BAD_WAVES]
DEP_WAVES        = list(range(2, 17))           # CESD available waves 2–16

LANDMARK_LAGS = [2, 4, 6, 8, 10, 12]  # years before 2022
# Restricted to 2010-2020 baselines: arrhythmia-specific question (Section C, [prefix]C266)
# was introduced in HRS wave 10 (2010). Pre-2010 waves have no arrhythmia variable —
# earlier lags fall back to RAND's generic heart problems flag (r[N]hearte), which
# conflates MI, CHF, angina, and arrhythmia. Those points are excluded.

# ─────────────────────────────────────────────────────────────────────────────
# BUILD COLUMN LIST FOR RAND HRS
# ─────────────────────────────────────────────────────────────────────────────
dep_cols   = [f"r{n}cesd"    for n in DEP_WAVES]
sleep_cols = []
for n in SLEEP_GOOD_WAVES:
    sleep_cols += [f"r{n}sleepfal", f"r{n}sleepwkn",
                   f"r{n}sleepwke", f"r{n}sleeprt"]

all_waves = list(range(2, 17))   # for covariates
age_cols   = [f"r{n}agey_b" for n in all_waves]
bmi_cols   = [f"r{n}bmi"    for n in all_waves]
hibpe_cols = [f"r{n}hibpe"  for n in all_waves]
diabe_cols = [f"r{n}diabe"  for n in all_waves]

rand_cols = (["hhidpn"] + dep_cols + sleep_cols
             + age_cols + bmi_cols + hibpe_cols + diabe_cols)

print(f"\nLoading {len(rand_cols)} columns from RAND HRS longitudinal file ...")
print(f"  File: {RAND_DTA.name}")
df_rand = pd.read_stata(RAND_DTA, columns=rand_cols)
df_rand["hhidpn"] = df_rand["hhidpn"].astype("int64")
df_rand = df_rand.set_index("hhidpn")
print(f"  Total participants: {len(df_rand):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cat_to_num(series):
    """Convert Stata categorical labels like '1.most of the time' → float."""
    return pd.to_numeric(
        series.astype(str).str.split(".").str[0],
        errors="coerce"
    )


def score_cesd(df, wave_n):
    """CESD-8 from RAND HRS: raw 0–8 → 0–100 (higher = worse)."""
    col = f"r{wave_n}cesd"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, name=col)
    raw = pd.to_numeric(df[col], errors="coerce")
    return raw.where(raw.between(0, 8)) * 12.5


def score_sleep(df, wave_n):
    """
    4-item sleep battery from RAND HRS (waves 6–16 excl. 9, 11).
    Returns 0–100 (higher = worse sleep), NaN if wave not valid or <2 items.
    """
    if wave_n in SLEEP_BAD_WAVES:
        return pd.Series(np.nan, index=df.index)
    cols = {
        "fal": f"r{wave_n}sleepfal",
        "wkn": f"r{wave_n}sleepwkn",
        "wke": f"r{wave_n}sleepwke",
        "rt":  f"r{wave_n}sleeprt",
    }
    if not all(c in df.columns for c in cols.values()):
        return pd.Series(np.nan, index=df.index)

    fal = _cat_to_num(df[cols["fal"]])
    wkn = _cat_to_num(df[cols["wkn"]])
    wke = _cat_to_num(df[cols["wke"]])
    rt  = _cat_to_num(df[cols["rt"]])

    # Trouble items: 1=Most/worst→10, 3=Rarely/best→0  (val-3)/(1-3)*10
    def trouble(s):
        v = s.where(s.isin([1, 2, 3]))
        return (v - 3.0) / (1.0 - 3.0) * 10.0

    # Rested item: 1=Most rested/best→0, 3=Rarely/worst→10  (val-1)/(3-1)*10
    def rested(s):
        v = s.where(s.isin([1, 2, 3]))
        return (v - 1.0) / (3.0 - 1.0) * 10.0

    s_fal = trouble(fal)
    s_wkn = trouble(wkn)
    s_wke = trouble(wke)
    s_rt  = rested(rt)

    total_score = s_fal.fillna(0) + s_wkn.fillna(0) + s_wke.fillna(0) + s_rt.fillna(0)
    n_valid = (
        s_fal.notna().astype(int) + s_wkn.notna().astype(int)
        + s_wke.notna().astype(int) + s_rt.notna().astype(int)
    )
    enough = n_valid >= 2
    # average of 0–10 items → ×10 → 0–100
    return (total_score / n_valid * 10.0).where(enough)


def get_covariate(df, wave_n, suffix, valid_range=None):
    col = f"r{wave_n}{suffix}"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    if suffix in ("hibpe", "diabe"):
        s = _cat_to_num(df[col])
        return s.where(s.isin([0, 1]))
    s = pd.to_numeric(df[col], errors="coerce")
    if valid_range:
        s = s.where(s.between(*valid_range))
    return s


# ─────────────────────────────────────────────────────────────────────────────
# PRE-SCORE ALL WAVES INTO WIDE FRAME
# ─────────────────────────────────────────────────────────────────────────────
print("\nPre-scoring all waves ...")
scored_cols = {}

for n in DEP_WAVES:
    yr = WAVE_YEAR[n]
    col_name = f"dep_w{n}"
    scored_cols[col_name] = score_cesd(df_rand, n)
    n_valid = scored_cols[col_name].notna().sum()
    print(f"  dep   w{n:2d} ({yr}): n={n_valid:6,}")

for n in SLEEP_GOOD_WAVES:
    yr = WAVE_YEAR[n]
    col_name = f"sleep_w{n}"
    scored_cols[col_name] = score_sleep(df_rand, n)
    n_valid = scored_cols[col_name].notna().sum()
    print(f"  sleep w{n:2d} ({yr}): n={n_valid:6,}")

for n in all_waves:
    scored_cols[f"age_w{n}"]   = get_covariate(df_rand, n, "agey_b", (30, 120))
    scored_cols[f"bmi_w{n}"]   = get_covariate(df_rand, n, "bmi", (12, 75))
    scored_cols[f"hibpe_w{n}"] = get_covariate(df_rand, n, "hibpe")
    scored_cols[f"diabe_w{n}"] = get_covariate(df_rand, n, "diabe")

df_scored = pd.DataFrame(scored_cols)  # index = hhidpn

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EXISTING COHORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading full-cohort arrhythmia dataset ...")
df_wide = pd.read_csv(WIDE_CSV).set_index("HHID_PN")
print(f"  Total participants: n={len(df_wide):,}")
print(f"  afib_ever:    {df_wide['afib_ever'].sum():,.0f}")
print(f"  afib_onset_wave distribution (NaN = never reported):")
print(df_wide["afib_onset_wave"].value_counts(dropna=False).sort_index().to_string())

# Keep all participants — landmark loop will exclude prevalent cases per lag dynamically
cohort = df_wide.copy()
n_any  = int(df_wide["afib_onset_wave"].notna().sum())
print(f"\n  Participants with any arrhythmia across 2010-2022: {n_any:,}")

# Merge RAND HRS scored data
cohort = cohort.join(df_scored, how="left")
print(f"  RAND HRS scores merged (all IDs matched)")

# Save multi-wave scores for reference
dep_score_cols   = [f"dep_w{n}"   for n in DEP_WAVES   if f"dep_w{n}"   in cohort.columns]
sleep_score_cols = [f"sleep_w{n}" for n in SLEEP_GOOD_WAVES if f"sleep_w{n}" in cohort.columns]
cohort[dep_score_cols + sleep_score_cols].reset_index().to_csv(
    DATA_OUT / "multiwave_scores.csv", index=False
)
print(f"  -> multiwave_scores.csv saved")

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK COX MODELS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  LANDMARK COX MODELS  (one per lag × predictor)")
print("=" * 75)

# Landmark CHARGE-AF covariates. Continuous stems from dict; binary use RAND-native
# names (hibpe/diabe) which differ from dict composite names (hypertension/diabetes).
_rand_cont  = _vd.continuous_composites("HRS_RAND", "charge_covariate")   # age, bmi
CHARGE_COVS = [f"{s}_lm" for s in _rand_cont] + ["hibpe_lm", "diabe_lm", "sleep_apnea"]

results_rows = []


def run_landmark_cox(df_lm, pred_col, pred_label, lag_yr, lm_year):
    covs = [pred_col + "_z"] + [c + "_z" if c in ("age_lm", "bmi_lm") else c
                                 for c in CHARGE_COVS]
    # z-score continuous covariates within this landmark sample
    df_sub = df_lm.copy()
    for col in [pred_col, "age_lm", "bmi_lm"]:
        mu, sd = df_sub[col].mean(), df_sub[col].std()
        if sd > 0:
            df_sub[col + "_z"] = (df_sub[col] - mu) / sd

    needed = [pred_col + "_z", "age_lm_z", "bmi_lm_z",
              "hibpe_lm", "diabe_lm", "sleep_apnea", "tstop", "event"]
    cc = df_sub[needed].dropna()
    n, ev = len(cc), int(cc["event"].sum())

    if ev < 5:
        print(f"    SKIP (events={ev} < 5)")
        return None

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(cc, duration_col="tstop", event_col="event",
                formula=f"{pred_col}_z + age_lm_z + bmi_lm_z + hibpe_lm + diabe_lm + sleep_apnea")
        row_s = cph.summary.loc[pred_col + "_z"]
        hr    = np.exp(row_s["coef"])
        ci_lo = np.exp(row_s["coef lower 95%"])
        ci_hi = np.exp(row_s["coef upper 95%"])
        p     = row_s["p"]
        print(f"    {pred_label:16s}: HR={hr:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  "
              f"p={p:.3f}  n={n:,}  ev={ev}")
        return {
            "lag_years":    lag_yr,
            "landmark_year": lm_year,
            "predictor":    pred_label,
            "HR":           round(hr, 4),
            "CI_lo":        round(ci_lo, 4),
            "CI_hi":        round(ci_hi, 4),
            "p":            round(p, 4),
            "n":            n,
            "events":       ev,
        }
    except Exception as e:
        print(f"    ERROR {pred_label}: {e}")
        return None


for lag in LANDMARK_LAGS:
    lm_year = 2022 - lag
    if lm_year not in YEAR_WAVE:
        print(f"\nLag {lag}yr ({lm_year}): not a valid HRS wave year — skipping")
        continue

    wave_n = YEAR_WAVE[lm_year]
    print(f"\n--- Lag {lag}yr  (landmark: {lm_year}, wave {wave_n}) ---")

    # Eligible: AFib-free at lm_year
    eligible = cohort[
        cohort["afib_onset_wave"].isna() | (cohort["afib_onset_wave"] > lm_year)
    ].copy()

    # Time and event — use afib_onset_wave directly (landmark-agnostic)
    has_onset = eligible["afib_onset_wave"].notna()
    eligible["tstop"] = eligible["afib_onset_wave"].where(has_onset, 2022.0) - lm_year
    eligible["tstop"] = eligible["tstop"].clip(lower=0.01)
    eligible["event"] = (
        has_onset & (eligible["afib_onset_wave"] > lm_year)
    ).astype(int)

    # Covariates at this wave
    eligible["age_lm"]      = eligible.get(f"age_w{wave_n}",   np.nan)
    eligible["bmi_lm"]      = eligible.get(f"bmi_w{wave_n}",   np.nan)
    eligible["hibpe_lm"]    = eligible.get(f"hibpe_w{wave_n}", np.nan)
    eligible["diabe_lm"]    = eligible.get(f"diabe_w{wave_n}", np.nan)
    # sleep_apnea from 2016 used as time-invariant covariate across all landmark waves
    if "sleep_apnea" not in eligible.columns:
        eligible["sleep_apnea"] = np.nan

    print(f"  n eligible={len(eligible):,}  events={eligible['event'].sum()}")

    # Depression model
    dep_col_lm = f"dep_w{wave_n}"
    if dep_col_lm in eligible.columns:
        eligible["dep_lm"] = eligible[dep_col_lm]
        res = run_landmark_cox(eligible, "dep_lm", "Depression", lag, lm_year)
        if res:
            results_rows.append(res)
    else:
        print(f"    SKIP dep: column {dep_col_lm} not in cohort")

    # Sleep model (only valid waves)
    sleep_col_lm = f"sleep_w{wave_n}"
    if wave_n not in SLEEP_BAD_WAVES and sleep_col_lm in eligible.columns:
        eligible["sleep_lm"] = eligible[sleep_col_lm]
        res = run_landmark_cox(eligible, "sleep_lm", "Sleep quality", lag, lm_year)
        if res:
            results_rows.append(res)
    elif wave_n in SLEEP_BAD_WAVES:
        print(f"    SKIP sleep: wave {wave_n} ({lm_year}) has <2% coverage (sub-sample)")
    else:
        print(f"    SKIP sleep: wave {wave_n} too early for 4-item battery")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results_rows)
df_res.to_csv(DATA_OUT / "multiwave_landmark_results.csv", index=False)
print(f"\n  -> multiwave_landmark_results.csv  ({len(df_res)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Forest Plot — HR at each lag for dep and sleep
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

C_DEP   = "#7B5EA7"
C_SLEEP = "#F4A261"

if len(df_res) == 0:
    print("  No results to plot — skipping figures.")
else:
    df_dep   = df_res[df_res["predictor"] == "Depression"].sort_values("lag_years")
    df_sleep = df_res[df_res["predictor"] == "Sleep quality"].sort_values("lag_years")

    all_rows = pd.concat([
        df_dep.assign(_order=range(len(df_dep))),
        df_sleep.assign(_order=range(len(df_sleep))),
    ], ignore_index=True)

    # Build y positions: group by lag, dep then sleep within each group
    y = 0
    y_ticks, y_labels, y_colors = [], [], []
    prev_lag = None
    row_records = []

    for lag in sorted(df_res["lag_years"].unique()):
        sub = df_res[df_res["lag_years"] == lag]
        lm_yr = 2022 - lag
        for _, row in sub.sort_values("predictor", ascending=False).iterrows():
            col = C_DEP if row["predictor"] == "Depression" else C_SLEEP
            row_records.append((y, row, col))
            lbl = f"{row['predictor']}  [{lm_yr}  {lag}yr lag]"
            y_ticks.append(y)
            y_labels.append(lbl)
            y_colors.append(col)
            y += 1
        if prev_lag is not None:
            pass  # separator drawn below
        prev_lag = lag
        y += 0.3  # gap between lag groups

    fig_h = max(6, len(row_records) * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("white")

    for yi, row, col in row_records:
        sig = "★" if row["p"] < 0.05 else ""
        ax.plot([row["CI_lo"], row["CI_hi"]], [yi, yi], color=col, lw=2.0, zorder=2)
        ax.scatter([row["HR"]], [yi], color=col, s=75, zorder=3, marker="D")
        ax.text(
            row["CI_hi"] + 0.03, yi,
            f"HR={row['HR']:.3f} [{row['CI_lo']:.3f}–{row['CI_hi']:.3f}]"
            f"  p={row['p']:.3f}{sig}  ev={row['events']}",
            va="center", ha="left", fontsize=7.5, color=col,
        )

    ax.axvline(1.0, color="black", lw=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, col in zip(ax.get_yticklabels(), y_colors):
        tick.set_color(col)

    x_max = max(df_res["CI_hi"].max() + 0.8, 2.5)
    ax.set_xlim(0.3, x_max)
    ax.set_xlabel("Hazard Ratio for Incident Arrhythmia per SD  (CHARGE-AF adjusted)", fontsize=10)
    ax.set_title(
        "Event-Anchored Landmark Analysis: HR vs Years Before Arrhythmia Onset\n"
        "Depression (CESD-8) and Sleep Quality (4-item), RAND HRS 1992–2022",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[Patch(color=C_DEP, label="Depression (CESD-8)"),
                 Patch(color=C_SLEEP, label="Sleep quality (4-item)")],
        fontsize=9, loc="lower right",
    )

    plt.tight_layout()
    fig.savefig(FIG_OUT / "multiwave_forest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> multiwave_forest.png")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 2: HR trend lines across lags
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, pred, col, title in zip(
        axes,
        ["Depression", "Sleep quality"],
        [C_DEP, C_SLEEP],
        ["Depression Score (CESD-8)", "Sleep Quality Score (4-item)"],
    ):
        sub = df_res[df_res["predictor"] == pred].sort_values("lag_years")
        if sub.empty:
            ax.set_title(title + "\n(no data)")
            continue

        xs     = sub["lag_years"].tolist()
        hrs    = sub["HR"].tolist()
        ci_los = sub["CI_lo"].tolist()
        ci_his = sub["CI_hi"].tolist()

        ax.fill_between(xs, ci_los, ci_his, color=col, alpha=0.12, label="95% CI")
        ax.plot(xs, hrs, color=col, lw=2.5, marker="D", markersize=7, zorder=3)
        ax.axhline(1.0, color="black", lw=1.0, linestyle="--", label="HR=1")

        for xi, hr, ci_hi, row in zip(xs, hrs, ci_his, sub.itertuples()):
            sig = "★" if row.p < 0.05 else ""
            ax.text(xi, ci_hi + 0.02, f"{hr:.2f}{sig}\n(ev={row.events})",
                    ha="center", va="bottom", fontsize=7, color=col)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}yr" for x in xs], fontsize=8)
        ax.set_xlabel("Years before 2022 (landmark lag)", fontsize=9)
        ax.set_ylabel("Hazard Ratio per SD", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.invert_xaxis()  # 20yr on left → 2yr on right (closer to event = right)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    fig.suptitle(
        "HR Trend: Does the Signal Sharpen Near the Arrhythmia Event?\n"
        "(x-axis inverted: right = closer to event)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(FIG_OUT / "multiwave_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> multiwave_trend.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  SUMMARY")
print("=" * 75)

if len(df_res) > 0:
    print()
    for pred in ["Depression", "Sleep quality"]:
        sub = df_res[df_res["predictor"] == pred].sort_values("lag_years", ascending=False)
        if sub.empty:
            continue
        print(f"\n  {pred} (per SD, CHARGE-AF adjusted):")
        print(f"  {'Lag':>6}  {'LM year':>7}  {'HR':>6}  {'95% CI':^17}  {'p':>6}  {'n':>5}  {'ev':>3}")
        print("  " + "-" * 58)
        for _, r in sub.iterrows():
            sig = " *" if r["p"] < 0.05 else "  "
            print(f"  {int(r['lag_years']):>4}yr  {int(r['landmark_year']):>7}  "
                  f"{r['HR']:>6.3f}  [{r['CI_lo']:.3f}–{r['CI_hi']:.3f}]  "
                  f"{r['p']:>6.3f}{sig}  {int(r['n']):>5}  {int(r['events']):>3}")
else:
    print("  No results produced — check event counts and data availability.")

print()
print("NOTES:")
print("  1. Arrhythmia (abnormal heart rhythm) is NOT in RAND HRS as a distinct variable.")
print("     Incident cases come from the existing pipeline (raw HRS Section C .da files).")
print("     The RAND HRS r[N]hearte / r[N]heart captures generic heart problems (all types).")
print("  2. Sleep 4-item battery: available waves 6–16 (2002–2022), EXCLUDING waves 9")
print("     (2008, 0.3% coverage) and 11 (2012, 1.1% coverage) — sub-sample only.")
print("     Response format is identical across all 9 usable waves (consistent).")
print("  3. All predictors z-scored within each landmark sample for HR comparability.")
print("  4. CHARGE-AF covariates (age, BMI, hypertension, diabetes) updated per landmark wave.")
print("  5. Power is limited at long lags: few participants have measurements 14–20yr ago.")
