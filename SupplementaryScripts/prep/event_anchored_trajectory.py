"""
event_anchored_trajectory.py
=============================
Event-anchored trajectory: sleep quality and depression scores in the years
*before* incident arrhythmia (AF onset), averaged across cases and controls.

Design
------
For each incident case (afib_onset_wave not NaN, arrhythmia-free at 2010):
  - t0 = afib_onset_wave (year of first positive arrhythmia report)
  - Extract sleep/CESD score at t0-2, t0-4, t0-6, t0-8, t0-10 (HRS biennial)
  - Label each observation by lag = score_year - onset_year  (negative = before)

For controls (never developed arrhythmia):
  - Anchor to pseudo-event = 2022 (their last possible observation year)
  - Same lag extraction

Average mean +/- 95% CI across participants at each lag, separately for cases
and controls. Plot two panels: sleep and depression.

Outputs
-------
  data/event_anchored_scores.csv   — long-format: HHID_PN, group, lag, dep, sleep
  figures/event_anchored_traj.png  — mean trajectory plot (cases vs controls)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[2]
DATA_OUT  = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "analytic"
FIG_OUT   = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "figures"

RAND_DTA  = REPO_ROOT / "RawData" / "HRS" / "RandHRS" / "randhrs1992_2022v1.dta"
WIDE_CSV  = REPO_ROOT / "SF_OUTPUT" / "analytic" / "hrs_analytic_wide_fullcohort.csv"

_HRS = REPO_ROOT / "HRS"
_RAW = _HRS / "RawData" / "HRS"

# XC267 = "YEAR FIRST HAD ABNORMAL HEART RHYTHM" — one per wave, prefix varies
_C267_FWF = {
    2010: (_RAW / "HRS2010" / "H10C_R.da",             _RAW / "HRS2010" / "H10C_R.txt",             "M"),
    2012: (_RAW / "HRS2012" / "H12C_R.da",             _RAW / "HRS2012" / "H12C_R.txt",             "N"),
    2014: (_RAW / "HRS2014" / "H14C_R.da",             _RAW / "HRS2014" / "H14C_R.txt",             "O"),
    2016: (_HRS / "RawData" / "HRS" / "HMS2016" / "H16da" / "H16C_R.da",   _HRS / "RawData" / "HRS" / "HMS2016" / "H16cb" / "H16C_R.txt",  "P"),
    2018: (_RAW / "HRS2018" / "h18da" / "H18C_R.da",   _RAW / "HRS2018" / "h18cb" / "H18C_R.txt",  "Q"),
    2020: (_RAW / "HRS2020" / "h20da" / "H20C_R.da",   _RAW / "HRS2020" / "h20cb" / "H20C_R.txt",  "R"),
}
_C267_CSV = {2022: (_HRS / "RawData" / "HRS" / "HMS2022" / "H22csv" / "h22c_r.csv", "S")}

# ─────────────────────────────────────────────────────────────────────────────
# WAVE / YEAR CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WAVE_YEAR = {
    1: 1992, 2: 1994, 3: 1996, 4: 1998, 5: 2000,
    6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010,
    11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022,
}
YEAR_WAVE = {v: k for k, v in WAVE_YEAR.items()}

SLEEP_BAD_WAVES  = {9, 11}
SLEEP_GOOD_WAVES = [n for n in range(6, 17) if n not in SLEEP_BAD_WAVES]
DEP_WAVES        = list(range(2, 17))

# Lags (relative to onset) to visualise — negative = before onset
LAGS_TO_PLOT = [-10, -8, -6, -4, -2, 0]

import re as _re

def _parse_colspecs(cb_path):
    specs, pos = {}, 0
    vname_pat = _re.compile(r"^([A-Z][A-Z0-9_]+)\s+\S")
    meta_pat  = _re.compile(r"Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)", _re.IGNORECASE)
    cur = None
    with open(cb_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line and not line.startswith((" ", "=", "-", "{")):
                m = vname_pat.match(line)
                if m:
                    cur = m.group(1)
            m2 = meta_pat.search(line)
            if m2 and cur:
                w = int(m2.group(1))
                specs[cur] = (pos, pos + w)
                pos += w
                cur = None
    return specs


def _load_c267_fwf(da_path, cb_path, prefix):
    col = f"{prefix}C267"
    specs = _parse_colspecs(cb_path)
    if col not in specs or "HHID" not in specs or "PN" not in specs:
        return pd.DataFrame(columns=["hhidpn", "xc267_year"])
    need     = ["HHID", "PN", col]
    colspecs = [specs[v] for v in need]
    df = pd.read_fwf(da_path, colspecs=colspecs, header=None,
                     names=need, dtype=str, encoding="latin-1")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["HHID"]   = df["HHID"].str.zfill(6)
    df["PN"]     = df["PN"].str.zfill(3)
    df["hhidpn"] = pd.to_numeric(df["HHID"] + df["PN"], errors="coerce")
    df[col]      = pd.to_numeric(df[col], errors="coerce")
    df           = df[df[col].between(1900, 2022)].copy()
    return df[["hhidpn", col]].rename(columns={col: "xc267_year"})


def _load_c267_csv(csv_path, prefix):
    col = f"{prefix}C267".upper()
    df  = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = df.columns.str.upper()
    if "HHID" in df.columns and "PN" in df.columns:
        df["HHID"]   = df["HHID"].str.zfill(6)
        df["PN"]     = df["PN"].str.zfill(3)
        df["hhidpn"] = pd.to_numeric(df["HHID"] + df["PN"], errors="coerce")
    elif "HHIDPN" in df.columns:
        df["hhidpn"] = pd.to_numeric(df["HHIDPN"], errors="coerce")
    if col not in df.columns:
        return pd.DataFrame(columns=["hhidpn", "xc267_year"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df      = df[df[col].between(1900, 2022)].copy()
    return df[["hhidpn", col]].rename(columns={col: "xc267_year"})


def _nearest_biennial(year):
    candidates = [2010, 2012, 2014, 2016, 2018, 2020, 2022]
    return min(candidates, key=lambda y: abs(y - year))


print()
print("=" * 70)
print("  EVENT-ANCHORED TRAJECTORY — SLEEP & DEPRESSION BEFORE ARRHYTHMIA ONSET")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD XC267 — earliest self-reported arrhythmia diagnosis year
# ─────────────────────────────────────────────────────────────────────────────
print("\n[0] Loading XC267 (year first had abnormal heart rhythm) ...")
c267_frames = []
for yr, (da, cb, pfx) in _C267_FWF.items():
    if da.exists():
        c267_frames.append(_load_c267_fwf(da, cb, pfx))
for yr, (csv_path, pfx) in _C267_CSV.items():
    if csv_path.exists():
        c267_frames.append(_load_c267_csv(csv_path, pfx))

if c267_frames:
    c267_all = pd.concat(c267_frames, ignore_index=True)
    c267_min = (c267_all.groupby("hhidpn")["xc267_year"]
                .min().reset_index().rename(columns={"xc267_year": "first_arrhythmia_year"}))
    print(f"    {len(c267_min):,} participants with valid XC267 (range "
          f"{int(c267_min['first_arrhythmia_year'].min())}–"
          f"{int(c267_min['first_arrhythmia_year'].max())})")
else:
    c267_min = pd.DataFrame(columns=["hhidpn", "first_arrhythmia_year"])
    print("    WARNING: no XC267 files found — falling back to wave-level onset")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD RAND HRS — dep and sleep columns only
# ─────────────────────────────────────────────────────────────────────────────
dep_cols   = [f"r{n}cesd" for n in DEP_WAVES]
sleep_cols = []
for n in SLEEP_GOOD_WAVES:
    sleep_cols += [f"r{n}sleepfal", f"r{n}sleepwkn",
                   f"r{n}sleepwke",  f"r{n}sleeprt"]

rand_cols = ["hhidpn"] + dep_cols + sleep_cols

print(f"\n[1] Loading {len(rand_cols)} columns from RAND HRS ...")
rand = pd.read_stata(RAND_DTA, columns=rand_cols)
rand["hhidpn"] = rand["hhidpn"].astype(np.int64)
rand.set_index("hhidpn", inplace=True)
print(f"    Loaded {len(rand):,} participants")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCORE SLEEP BATTERY PER WAVE
#    Stata value labels look like "1.most.of.the.time" — strip to get numeric.
#    Items sleepfal/wkn/wke: 1=most of time (worst), 3=rarely (best)
#    Item sleeprt:           1=most rested (best),   3=rarely rested (worst)
# ─────────────────────────────────────────────────────────────────────────────

def _cat_to_num(series):
    """Convert Stata categorical labels like '1.most of the time' -> float."""
    return pd.to_numeric(series.astype(str).str.split(".").str[0], errors="coerce")

def _trouble(s):
    v = s.where(s.isin([1, 2, 3]))
    return (v - 3.0) / (1.0 - 3.0) * 10.0   # 1(worst)->10, 3(best)->0

def _rested(s):
    v = s.where(s.isin([1, 2, 3]))
    return (v - 1.0) / (3.0 - 1.0) * 10.0   # 1(best)->0, 3(worst)->10

print("\n[2] Scoring sleep battery per wave ...")
sleep_scores = {}
for n in SLEEP_GOOD_WAVES:
    yr = WAVE_YEAR[n]
    fal = _cat_to_num(rand[f"r{n}sleepfal"])
    wkn = _cat_to_num(rand[f"r{n}sleepwkn"])
    wke = _cat_to_num(rand[f"r{n}sleepwke"])
    rt  = _cat_to_num(rand[f"r{n}sleeprt"])
    raw = _trouble(fal) + _trouble(wkn) + _trouble(wke) + _rested(rt)  # 0-40
    n_valid = raw.notna().sum()
    score = raw / 40.0 * 100.0   # 0=best, 100=worst
    sleep_scores[yr] = score
    print(f"    Wave {n} ({yr}): {n_valid:,} valid sleep scores")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SCORE CESD PER WAVE (0-8 × 12.5 -> 0-100, higher = worse)
# ─────────────────────────────────────────────────────────────────────────────
dep_scores = {}
for n in DEP_WAVES:
    yr = WAVE_YEAR[n]
    raw = pd.to_numeric(rand.get(f"r{n}cesd"), errors="coerce")
    dep_scores[yr] = raw * 12.5

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD WIDE COHORT — get onset info
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] Loading wide cohort from {WIDE_CSV.name} ...")
wide = pd.read_csv(WIDE_CSV)
wide["HHID_PN"] = wide["HHID_PN"].astype(np.int64)
wide.set_index("HHID_PN", inplace=True)
print(f"    {len(wide):,} participants")

# Merge XC267 to refine onset year for incident cases
if not c267_min.empty:
    c267_min["hhidpn"] = c267_min["hhidpn"].astype(np.int64)
    wide = wide.merge(c267_min.set_index("hhidpn"), left_index=True, right_index=True, how="left")
    original_onset = wide["afib_onset_wave"].copy()
    def _refine_onset(row):
        wave_onset = row["afib_onset_wave"]
        xc267      = row["first_arrhythmia_year"]
        if pd.isna(wave_onset) or pd.isna(xc267):
            return wave_onset
        # Use exact XC267 year if it predates the wave-level onset
        return float(xc267) if xc267 < wave_onset else wave_onset
    wide["afib_onset_wave"] = wide.apply(_refine_onset, axis=1)
    refined = (wide["afib_onset_wave"] < original_onset).sum()
    print(f"    XC267 refined onset year for {refined:,} cases (earlier biennial wave)")
else:
    wide["first_arrhythmia_year"] = np.nan

# Arrhythmia-free at 2010 (eligible for incident analysis)
arrhythmia_free_2010 = wide["afib_2010"].fillna(0) != 1

# Incident cases: arrhythmia-free at 2010, onset after 2010
has_onset   = wide["afib_onset_wave"].notna()
cases_mask  = arrhythmia_free_2010 & has_onset & (wide["afib_onset_wave"] > 2010)
ctrl_mask   = arrhythmia_free_2010 & ~has_onset

cases  = wide[cases_mask].copy()
ctrls  = wide[ctrl_mask].copy()
print(f"    Incident cases (onset 2012-2022):  {len(cases):,}")
print(f"    Never-arrhythmia controls:         {len(ctrls):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXTRACT SCORES AT EACH LAG RELATIVE TO ANCHOR
# ─────────────────────────────────────────────────────────────────────────────
# Available biennial years with arrhythmia data
AFIB_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022]

def extract_lag_scores(group_df, anchor_col, group_label, lags):
    """
    For each participant, extract dep/sleep score at (anchor_year + lag).
    Anchor may be an exact year (from XC267); each lag target is snapped to
    the nearest available biennial score year.  Skips if the snap is >1.5 yr
    away from the target, or if a biennial year has already been used for this
    participant (prevents two lags mapping to the same observation).
    """
    avail = sorted(dep_scores.keys())
    rows = []
    for pid, row in group_df.iterrows():
        anchor = row[anchor_col]
        if pd.isna(anchor):
            continue
        anchor = float(anchor)
        used_score_years = set()
        for lag in lags:
            target     = anchor + lag
            score_year = min(avail, key=lambda y: abs(y - target))
            if abs(score_year - target) > 1.5 or score_year in used_score_years:
                continue
            used_score_years.add(score_year)
            dep   = dep_scores.get(score_year, pd.Series(dtype=float)).get(pid)
            sleep = sleep_scores.get(score_year, pd.Series(dtype=float)).get(pid)
            if dep is None and sleep is None:
                continue
            rows.append({
                "HHID_PN":     pid,
                "group":       group_label,
                "anchor_year": anchor,
                "score_year":  score_year,
                "lag":         lag,
                "dep":         dep if dep is not None else np.nan,
                "sleep":       sleep if sleep is not None else np.nan,
            })
    return pd.DataFrame(rows)


print("\n[4] Extracting lag-anchored scores ...")

# Cases: anchor = afib_onset_wave; lags -10 to 0
case_df = extract_lag_scores(cases, "afib_onset_wave", "Cases (incident arrhythmia)",
                              lags=[-10, -8, -6, -4, -2, 0])

# Controls: anchor = 2022 (pseudo-event); same lags
ctrls["_anchor"] = 2022
ctrl_df = extract_lag_scores(ctrls, "_anchor", "Controls (never arrhythmia)",
                             lags=[-10, -8, -6, -4, -2, 0])

long_df = pd.concat([case_df, ctrl_df], ignore_index=True)

# Remove lag=0 for controls (no meaningful anchor event)
long_df = long_df[~((long_df["group"] == "Controls (never arrhythmia)") & (long_df["lag"] == 0))]

long_df.to_csv(DATA_OUT / "event_anchored_scores.csv", index=False)
print(f"    Saved {len(long_df):,} rows -> data/event_anchored_scores.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. AGGREGATE: MEAN +/- 95% CI PER GROUP × LAG
# ─────────────────────────────────────────────────────────────────────────────
def ci95(s):
    s = s.dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan
    se = s.std(ddof=1) / np.sqrt(n)
    return s.mean() - 1.96 * se, s.mean() + 1.96 * se

print("\n[5] Aggregating mean trajectories ...")

agg_rows = []
for (group, lag), sub in long_df.groupby(["group", "lag"]):
    for measure in ["dep", "sleep"]:
        vals = sub[measure].dropna()
        lo, hi = ci95(vals)
        agg_rows.append({
            "group":   group,
            "lag":     lag,
            "measure": measure,
            "n":       len(vals),
            "mean":    vals.mean(),
            "ci_lo":   lo,
            "ci_hi":   hi,
        })

agg = pd.DataFrame(agg_rows).sort_values(["measure", "group", "lag"])

# Print summary
for measure in ["dep", "sleep"]:
    label = "Depression (CESD-8 -> 0-100)" if measure == "dep" else "Sleep quality (0-100)"
    print(f"\n  {label}:")
    sub = agg[agg["measure"] == measure].copy()
    print(sub[["group", "lag", "n", "mean", "ci_lo", "ci_hi"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from viz_style import PALETTE, apply_style, C_SLEEP, C_DEP
    apply_style()
    COL_CTRL = PALETTE.get("neutral", "#8E8E93")
except Exception:
    C_DEP    = "#7B5EA7"
    C_SLEEP  = "#E07828"
    COL_CTRL = "#8E8E93"

# Per-measure case colors: depression panel → purple, sleep panel → orange
CASE_COLORS = {"dep": C_DEP, "sleep": C_SLEEP}

MEASURE_LABELS = {
    "dep":   "Depression score (CESD-8 -> 0–100)",
    "sleep": "Sleep quality score (0–100, higher = worse)",
}
MEASURE_TITLES = {
    "dep":   "Depression",
    "sleep": "Sleep Quality",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(
    "Trajectories Before Arrhythmia Onset (Event-Anchored, HRS 2010–2022)",
    fontsize=13, fontweight="bold", y=1.01,
)

for ax, measure in zip(axes, ["dep", "sleep"]):
    COL_CASE = CASE_COLORS[measure]
    for group, col, ls, mk in [
        ("Cases (incident arrhythmia)",   COL_CASE, "-",  "o"),
        ("Controls (never arrhythmia)",   COL_CTRL, "--", "s"),
    ]:
        sub = agg[(agg["measure"] == measure) & (agg["group"] == group)].sort_values("lag")
        if sub.empty:
            continue

        lags   = sub["lag"].values
        means  = sub["mean"].values
        lo     = sub["ci_lo"].values
        hi     = sub["ci_hi"].values
        ns     = sub["n"].values

        ax.plot(lags, means, color=col, linestyle=ls, marker=mk,
                markersize=5, linewidth=2, label=group)
        ax.fill_between(lags, lo, hi, color=col, alpha=0.15)

        # Annotate n at each lag
        for lag_i, mean_i, n_i in zip(lags, means, ns):
            ax.annotate(f"n={n_i:,}", (lag_i, mean_i),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6.5, ha="center", color=col, alpha=0.8)

    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Years relative to arrhythmia onset (t0)", fontsize=10)
    ax.set_ylabel(MEASURE_LABELS[measure], fontsize=9)
    ax.set_title(MEASURE_TITLES[measure], fontsize=11, fontweight="bold")
    ax.set_xticks(LAGS_TO_PLOT)
    ax.set_xticklabels([str(l) for l in LAGS_TO_PLOT])
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    note = ("Controls (never arrhythmia) anchored to 2022. HRS biennial waves; "
            "sleep unavailable waves 9 & 11.")
    ax.annotate(note, xy=(0.01, -0.12), xycoords="axes fraction",
                fontsize=7, color="gray", va="top")

plt.tight_layout()
out_path = FIG_OUT / "event_anchored_traj.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[6] Figure saved -> {out_path}")

print("\nDone.")
