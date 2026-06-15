"""
hrs_combined_trajectory_hr_bysex.py
Sex-stratified 2×2 combined figures (Male and Female):
  a  Event-anchored depression trajectory (cases vs controls)
  b  Event-anchored sleep quality trajectory (cases vs controls)
  c  HR trend across landmark lags — Depression (CESD-8)
  d  HR trend across landmark lags — Sleep Quality (4-item)

Data sources (same as hrs_combined_trajectory_hr.py + analysis_timelag_multiwave.py):
  - results/06_event_anchored/data/event_anchored_scores.csv   (trajectory panels)
  - results/data/hrs_analytic_wide_fullcohort.csv              (sex, arrhythmia, sleep_apnea)
  - raw_data/HRS/RandHRS/randhrs1992_2022v1.dta                (wave-level covariates for Cox)

Outputs:
  results/CherryPicked2/Figure_4_Male.png
  results/CherryPicked2/Figure_4_Female.png
"""

import warnings
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engine.viz_style as V
from engine.variable_loader import VariableDictionary

V.apply()

OUT     = ROOT / "results" / "CherryPicked2"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DPI = 150

C_DEP   = "#7B5EA7"
C_SLEEP = "#E07828"
C_CTRL  = "#8E8E93"

# ──────────────────────────────────────────────────────────────────────────────
# WAVE / LAG CONSTANTS (same as analysis_timelag_multiwave.py)
# ──────────────────────────────────────────────────────────────────────────────
WAVE_YEAR = {
    1: 1992, 2: 1994,  3: 1996,  4: 1998,  5: 2000,
    6: 2002, 7: 2004,  8: 2006,  9: 2008, 10: 2010,
   11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022,
}
YEAR_WAVE        = {v: k for k, v in WAVE_YEAR.items()}
SLEEP_BAD_WAVES  = {9, 11}
SLEEP_GOOD_WAVES = [n for n in range(6, 17) if n not in SLEEP_BAD_WAVES]
DEP_WAVES        = list(range(2, 17))
LANDMARK_LAGS    = [2, 4, 6, 8, 10, 12]
ALL_WAVES        = list(range(2, 17))
LAGS_TO_PLOT     = [-10, -8, -6, -4, -2, 0]

# ──────────────────────────────────────────────────────────────────────────────
# LOAD EVENT-ANCHORED TRAJECTORY DATA
# ──────────────────────────────────────────────────────────────────────────────
agg_path  = ROOT / "results" / "06_event_anchored" / "data" / "event_anchored_scores.csv"
wide_path = ROOT / "results" / "data" / "hrs_analytic_wide_fullcohort.csv"
rand_path = ROOT / "raw_data" / "HRS" / "RandHRS" / "randhrs1992_2022v1.dta"

print("Loading event-anchored trajectory data ...")
long_df = pd.read_csv(agg_path)

print("Loading full-cohort wide dataset ...")
df_wide = pd.read_csv(wide_path).set_index("HHID_PN")

# Attach sex to trajectory rows
long_df = long_df.join(df_wide[["sex_female"]], on="HHID_PN")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD RAND HRS (for wave-level Cox covariates)
# ──────────────────────────────────────────────────────────────────────────────
dep_cols   = [f"r{n}cesd"    for n in DEP_WAVES]
sleep_cols = []
for n in SLEEP_GOOD_WAVES:
    sleep_cols += [f"r{n}sleepfal", f"r{n}sleepwkn", f"r{n}sleepwke", f"r{n}sleeprt"]
age_cols   = [f"r{n}agey_b" for n in ALL_WAVES]
bmi_cols   = [f"r{n}bmi"    for n in ALL_WAVES]
hibpe_cols = [f"r{n}hibpe"  for n in ALL_WAVES]
diabe_cols = [f"r{n}diabe"  for n in ALL_WAVES]

rand_cols = (["hhidpn"] + dep_cols + sleep_cols
             + age_cols + bmi_cols + hibpe_cols + diabe_cols)

print(f"\nLoading {len(rand_cols)} columns from RAND HRS ...")
df_rand = pd.read_stata(rand_path, columns=rand_cols)
df_rand["hhidpn"] = df_rand["hhidpn"].astype("int64")
df_rand = df_rand.set_index("hhidpn")
print(f"  {len(df_rand):,} participants")

# ──────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS (identical to analysis_timelag_multiwave.py)
# ──────────────────────────────────────────────────────────────────────────────

def _cat_to_num(series):
    return pd.to_numeric(
        series.astype(str).str.split(".").str[0], errors="coerce"
    )


def score_cesd(df, wave_n):
    col = f"r{wave_n}cesd"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, name=col)
    raw = pd.to_numeric(df[col], errors="coerce")
    return raw.where(raw.between(0, 8)) * 12.5


def score_sleep(df, wave_n):
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

    def trouble(s):
        v = s.where(s.isin([1, 2, 3]))
        return (v - 3.0) / (1.0 - 3.0) * 10.0

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
    return (total_score / n_valid * 10.0).where(n_valid >= 2)


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


# ──────────────────────────────────────────────────────────────────────────────
# PRE-SCORE ALL WAVES
# ──────────────────────────────────────────────────────────────────────────────
print("\nPre-scoring all RAND HRS waves ...")
scored_cols = {}
for n in DEP_WAVES:
    scored_cols[f"dep_w{n}"]   = score_cesd(df_rand, n)
for n in SLEEP_GOOD_WAVES:
    scored_cols[f"sleep_w{n}"] = score_sleep(df_rand, n)
for n in ALL_WAVES:
    scored_cols[f"age_w{n}"]   = get_covariate(df_rand, n, "agey_b", (30, 120))
    scored_cols[f"bmi_w{n}"]   = get_covariate(df_rand, n, "bmi", (12, 75))
    scored_cols[f"hibpe_w{n}"] = get_covariate(df_rand, n, "hibpe")
    scored_cols[f"diabe_w{n}"] = get_covariate(df_rand, n, "diabe")

df_scored = pd.DataFrame(scored_cols)

# Build merged cohort (wide + scored RAND)
cohort = df_wide.copy()
cohort = cohort.join(df_scored, how="left")

# ──────────────────────────────────────────────────────────────────────────────
# COX MODEL HELPER (same formula as analysis_timelag_multiwave.py)
# ──────────────────────────────────────────────────────────────────────────────
_vd = VariableDictionary()
_rand_cont  = _vd.continuous_composites("HRS_RAND", "charge_covariate")
CHARGE_COVS = [f"{s}_lm" for s in _rand_cont] + ["hibpe_lm", "diabe_lm", "sleep_apnea"]


def run_landmark_cox(df_lm, pred_col, pred_label, lag_yr, lm_year):
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
            "lag_years":      lag_yr,
            "landmark_year":  lm_year,
            "predictor":      pred_label,
            "HR":             round(hr, 4),
            "CI_lo":          round(ci_lo, 4),
            "CI_hi":          round(ci_hi, 4),
            "p":              round(p, 4),
            "n":              n,
            "events":         ev,
        }
    except Exception as e:
        print(f"    ERROR {pred_label}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CI HELPER
# ──────────────────────────────────────────────────────────────────────────────
def ci95(s):
    s = s.dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan
    se = s.std(ddof=1) / np.sqrt(n)
    return s.mean() - 1.96 * se, s.mean() + 1.96 * se


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP: generate one figure per sex
# ──────────────────────────────────────────────────────────────────────────────
for sex_label, sex_val, out_name in [
    ("Male",   0, "Figure_4_Male.png"),
    ("Female", 1, "Figure_4_Female.png"),
]:
    print()
    print("=" * 75)
    print(f"  SEX STRATUM: {sex_label.upper()}")
    print("=" * 75)

    # ── Cox models for this sex ───────────────────────────────────────────────
    cohort_sex = cohort[cohort["sex_female"] == sex_val].copy()
    print(f"  Cohort size: {len(cohort_sex):,}")

    results_rows = []
    for lag in LANDMARK_LAGS:
        lm_year = 2022 - lag
        if lm_year not in YEAR_WAVE:
            continue
        wave_n = YEAR_WAVE[lm_year]
        print(f"\n--- Lag {lag}yr  (landmark: {lm_year}, wave {wave_n}) ---")

        eligible = cohort_sex[
            cohort_sex["afib_onset_wave"].isna() |
            (cohort_sex["afib_onset_wave"] > lm_year)
        ].copy()

        has_onset        = eligible["afib_onset_wave"].notna()
        eligible["tstop"] = eligible["afib_onset_wave"].where(has_onset, 2022.0) - lm_year
        eligible["tstop"] = eligible["tstop"].clip(lower=0.01)
        eligible["event"] = (
            has_onset & (eligible["afib_onset_wave"] > lm_year)
        ).astype(int)

        eligible["age_lm"]   = eligible.get(f"age_w{wave_n}",   np.nan)
        eligible["bmi_lm"]   = eligible.get(f"bmi_w{wave_n}",   np.nan)
        eligible["hibpe_lm"] = eligible.get(f"hibpe_w{wave_n}", np.nan)
        eligible["diabe_lm"] = eligible.get(f"diabe_w{wave_n}", np.nan)
        if "sleep_apnea" not in eligible.columns:
            eligible["sleep_apnea"] = np.nan

        print(f"  n eligible={len(eligible):,}  events={eligible['event'].sum()}")

        dep_col_lm = f"dep_w{wave_n}"
        if dep_col_lm in eligible.columns:
            eligible["dep_lm"] = eligible[dep_col_lm]
            res = run_landmark_cox(eligible, "dep_lm", "Depression", lag, lm_year)
            if res:
                results_rows.append(res)

        sleep_col_lm = f"sleep_w{wave_n}"
        if wave_n not in SLEEP_BAD_WAVES and sleep_col_lm in eligible.columns:
            eligible["sleep_lm"] = eligible[sleep_col_lm]
            res = run_landmark_cox(eligible, "sleep_lm", "Sleep quality", lag, lm_year)
            if res:
                results_rows.append(res)

    df_res = pd.DataFrame(results_rows)

    # ── Re-aggregate trajectories for this sex ────────────────────────────────
    long_sex = long_df[long_df["sex_female"] == sex_val].copy()

    agg_rows = []
    for (group, lag), sub in long_sex.groupby(["group", "lag"]):
        for measure in ["dep", "sleep"]:
            vals = sub[measure].dropna()
            lo, hi = ci95(vals)
            agg_rows.append({"group": group, "lag": lag, "measure": measure,
                             "n": len(vals), "mean": vals.mean(),
                             "ci_lo": lo, "ci_hi": hi})
    agg = pd.DataFrame(agg_rows).sort_values(["measure", "group", "lag"])

    # ── Build 2×2 figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                              gridspec_kw={"hspace": 0.30, "wspace": 0.22})
    panel_letters = [["a", "b"], ["c", "d"]]

    traj_configs = [
        ("dep",   C_DEP,   "Depression",    "Depression score"),
        ("sleep", C_SLEEP, "Sleep Quality", "Sleep quality score"),
    ]

    for col_idx, (measure, case_col, title, ylabel) in enumerate(traj_configs):
        ax     = axes[0][col_idx]
        letter = panel_letters[0][col_idx]

        for group, col, ls, mk in [
            ("Cases (incident arrhythmia)", case_col, "-",  "o"),
            ("Controls (never arrhythmia)", C_CTRL,   "--", "s"),
        ]:
            sub = agg[(agg["measure"] == measure) & (agg["group"] == group)].sort_values("lag")
            if sub.empty:
                continue
            lags  = sub["lag"].values
            means = sub["mean"].values
            lo    = sub["ci_lo"].values
            hi    = sub["ci_hi"].values
            ns    = sub["n"].values

            ax.plot(lags, means, color=col, linestyle=ls, marker=mk,
                    markersize=5, linewidth=2, label=group)
            ax.fill_between(lags, lo, hi, color=col, alpha=0.15)

            if "Cases" in group:
                for lag_i, mean_i, n_i in zip(lags, means, ns):
                    ax.annotate(f"n={n_i:,}", (lag_i, mean_i),
                                textcoords="offset points", xytext=(0, 8),
                                fontsize=6.5, ha="center", color="black", alpha=0.8)

        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_xlabel("Years relative to arrhythmia onset (t0)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(LAGS_TO_PLOT)
        ax.set_xticklabels([str(l) for l in LAGS_TO_PLOT])
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
                ha="left", va="center", fontsize=13, fontweight="bold")

        # Inset: linear slope overlay
        axins = ax.inset_axes([0.70, 0.06, 0.24, 0.28])
        axins.set_facecolor("none")
        for group, col, ls in [
            ("Cases (incident arrhythmia)", case_col, "-"),
            ("Controls (never arrhythmia)", C_CTRL,   "--"),
        ]:
            sub_ins = agg[(agg["measure"] == measure) & (agg["group"] == group)].sort_values("lag")
            if sub_ins.empty:
                continue
            lags_ins  = sub_ins["lag"].values
            means_ins = sub_ins["mean"].values
            valid     = ~np.isnan(means_ins)
            axins.scatter(lags_ins[valid], means_ins[valid], color=col, s=12, zorder=3)
            if valid.sum() >= 2:
                slope, intercept, *_ = linregress(lags_ins[valid], means_ins[valid])
                x_fit = np.linspace(lags_ins[valid].min(), lags_ins[valid].max(), 100)
                axins.plot(x_fit, slope * x_fit + intercept, color=col, linestyle=ls, lw=1.5)
        axins.set_xticks([]); axins.set_yticks([])
        axins.set_title("Mean slope", fontsize=6.5, pad=2)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)
        axins.spines["left"].set_alpha(0.5)
        axins.spines["bottom"].set_alpha(0.5)

    hr_configs = [
        ("Depression",    C_DEP,   "Depression Score (CESD-8)",    "upper left"),
        ("Sleep quality", C_SLEEP, "Sleep Quality Score (4-item)", "upper left"),
    ]

    for col_idx, (pred, col, title, legend_loc) in enumerate(hr_configs):
        ax     = axes[1][col_idx]
        letter = panel_letters[1][col_idx]

        sub = df_res[df_res["predictor"] == pred].sort_values("lag_years") if len(df_res) > 0 else pd.DataFrame()
        if sub.empty:
            ax.set_title(title + "\n(no data)")
            ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
                    ha="left", va="center", fontsize=13, fontweight="bold")
            continue

        xs     = sub["lag_years"].tolist()
        hrs    = sub["HR"].tolist()
        ci_los = sub["CI_lo"].tolist()
        ci_his = sub["CI_hi"].tolist()

        ax.fill_between(xs, ci_los, ci_his, color=col, alpha=0.12, label="95% CI")
        ax.plot(xs, hrs, color=col, lw=2.5, marker="D", markersize=6, zorder=3)
        ax.axhline(1.0, color="black", lw=1.0, linestyle="--", label="HR=1")

        for xi, hr, ci_hi, row in zip(xs, hrs, ci_his, sub.itertuples()):
            if row.p < 0.05:
                ax.text(xi, ci_hi + 0.02, f"{hr:.2f}★\n(ev={row.events})",
                        ha="center", va="bottom", fontsize=7, color=col)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}yr" for x in xs], fontsize=8)
        ax.set_xlabel("Years prior to arrhythmia onset", fontsize=9)
        ax.set_ylabel("Hazard Ratio per SD", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.invert_xaxis()
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.06)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc=legend_loc)
        ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
                ha="left", va="center", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Trajectories Before Arrhythmia Onset — {sex_label}s Only (HRS 2010–2022)",
        fontsize=13, fontweight="bold"
    )

    fig.canvas.draw()
    for lbl in axes[0][1].get_yticklabels():
        if lbl.get_text() == "46":
            lbl.set_visible(False)

    out_path = OUT / out_name
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")

print("\nDone.")
