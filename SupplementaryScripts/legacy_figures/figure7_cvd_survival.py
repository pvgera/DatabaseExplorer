"""
figure7_cvd_survival.py
========================
Figure 7a — CVD-free survival curves anchored on first top-tertile
            Sleep Quality Burden (SQB) wave
Figure 7b — CVD-free survival curves anchored on first top-tertile
            Depression Burden (DB) wave

Entry / anchor (t = 0)
    The first wave at which a participant crosses from BELOW into the top
    tertile (>= 66.7th percentile of the cohort-wide pooled score
    distribution), while free of all CVD subtypes at that wave.  Requires
    an observed prior wave below the threshold ("captured transition") to
    exclude prevalent high-burden participants.  HRS is biennial; onset is
    interval-censored — anchored to the first observed above-threshold wave.
    Reported as "high relative burden," not "severe" — no validated clinical
    cutpoint exists for these instruments.

Exit
    First incident CVD event of any type, or censoring at the last observed
    wave (proxy: latest wave with non-missing age in RAND HRS), capped 2022.

Competing-risks design
    Each person contributes exactly ONE event to the dataset: the FIRST
    incident CVD type observed after anchor.  If >= 2 subtypes appear
    simultaneously at the same first-incident wave they are classified as
    "Multiple Events."  Cause-specific (KM) over-estimates cumulative
    incidence by treating competing events as non-informative censoring;
    Aalen-Johansen (AJ) CIF is the correct estimator and is used here.

Estimator
    Aalen-Johansen cumulative incidence function (lifelines
    AalenJohansenFitter, jitter_level=1e-5 for biennial ties).  Plotted as
    1 − CIF so it reads as a survival-style decline from 100%.  "Any CVD"
    reference curve uses Kaplan-Meier on the marginal event (event = any
    first CVD).  Curves are descriptive and unadjusted; they complement,
    not replace, the covariate-adjusted Cox models in Figs. 4-5.

CVD subtypes
    Stroke         — HMS Section C variable C053 == 1
    Heart Attack   — HMS Section C variable C040 == 1
    Heart Failure  — HMS Section C variable C048 == 1
    Arrhythmia     — afib flag (C266 | C270M1 | C270M2) in analytic-wide CSV
    Multiple Events — >= 2 subtypes first appear at the same wave
    Any CVD         — any of the above (KM reference, dashed)

Sensitivity (--sensitivity flag)
    Re-runs at the top quartile (>= 75th pct) to confirm separation is
    robust to the exact threshold choice.

Outputs
    results/CherryPicked2/Figure_7a.png
    results/CherryPicked2/Figure_7b.png
    results/CherryPicked2/Figure_7_combined.png
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from lifelines import AalenJohansenFitter, KaplanMeierFitter

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "CherryPicked2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
import engine.viz_style as V
V.apply()

_SENSITIVITY = "--sensitivity" in sys.argv

print("=" * 68)
print("  CVD-FREE SURVIVAL (AJ CIF)  SQB / DB ANCHOR")
print("=" * 68)

# ─── File paths ───────────────────────────────────────────────────────────────
WIDE_CSV   = ROOT / "results" / "data"       / "hrs_analytic_wide_fullcohort.csv"
SCORES_CSV = ROOT / "results" / "05_timelag" / "data" / "multiwave_scores.csv"

WAVE_C_FWF = {
    2010: (ROOT/"raw_data"/"HRS"/"waves"/"HRS2010"/"H10C_R.da",
           ROOT/"raw_data"/"HRS"/"waves"/"HRS2010"/"H10C_R.txt", "M"),
    2012: (ROOT/"raw_data"/"HRS"/"waves"/"HRS2012"/"H12C_R.da",
           ROOT/"raw_data"/"HRS"/"waves"/"HRS2012"/"H12C_R.txt", "N"),
    2014: (ROOT/"raw_data"/"HRS"/"waves"/"HRS2014"/"H14C_R.da",
           ROOT/"raw_data"/"HRS"/"waves"/"HRS2014"/"H14C_R.txt", "O"),
    2016: (ROOT/"raw_data"/"HRS"/"HMS2016"/"h16da"/"H16C_R.da",
           ROOT/"raw_data"/"HRS"/"HMS2016"/"h16cb"/"H16C_R.txt", "P"),
    2018: (ROOT/"raw_data"/"HRS"/"waves"/"HRS2018"/"h18da"/"H18C_R.da",
           ROOT/"raw_data"/"HRS"/"waves"/"HRS2018"/"h18cb"/"H18C_R.txt", "Q"),
    2020: (ROOT/"raw_data"/"HRS"/"waves"/"HRS2020"/"h20da"/"H20C_R.da",
           ROOT/"raw_data"/"HRS"/"waves"/"HRS2020"/"h20cb"/"H20C_R.txt", "R"),
}
WAVE_C_CSV = ROOT / "raw_data" / "HRS" / "HMS2022" / "H22csv" / "h22c_r.csv"
CSV_PREFIX = "S"

# ─── Wave / year mappings ─────────────────────────────────────────────────────
WAVE_YEAR    = {10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022}
ALL_YEARS    = [2010, 2012, 2014, 2016, 2018, 2020, 2022]
SLEEP_YEARS  = [2010, 2014, 2016, 2018, 2020, 2022]   # wave 11/2012 excluded
DEP_YEARS    = ALL_YEARS
YEAR_TO_WAVE = {v: k for k, v in WAVE_YEAR.items()}

FLAG_PREFIX = {
    "Stroke":        "stroke",
    "Heart Attack":  "ha",
    "Heart Failure": "hf",
    "Arrhythmia":    "arrhythmia",
}

# Competing-risks event codes (0 = censored)
CR_CODES = {
    "Stroke": 1, "Heart Attack": 2,
    "Heart Failure": 3, "Arrhythmia": 4, "Multiple Events": 5,
}

# ─── Colour palette (consistent 7a / 7b) ─────────────────────────────────────
CVD_COLORS = {
    "Any CVD":         "#58595B",   # charcoal reference
    "Stroke":          "#4A7098",   # steel blue
    "Heart Attack":    "#E87D3E",   # orange
    "Heart Failure":   "#7B5EA7",   # purple
    "Arrhythmia":      "#C8102E",   # crimson
    "Multiple Events": "#2DA84B",   # green
}
SUBTYPE_ORDER = [
    "Any CVD", "Stroke", "Heart Attack",
    "Heart Failure", "Arrhythmia", "Multiple Events",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_colspecs(cb_path):
    specs, pos = {}, 0
    vname_re = re.compile(r"^([A-Z][A-Z0-9_]+)\s+\S")
    meta_re  = re.compile(r"Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)", re.I)
    cur = None
    with open(cb_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line and not line[0].isspace() and line[0] not in "=-{":
                m = vname_re.match(line)
                if m:
                    cur = m.group(1)
            m2 = meta_re.search(line)
            if m2 and cur:
                w = int(m2.group(1))
                specs[cur] = (pos, pos + w)
                pos += w
                cur = None
    return specs


def _cvd_from_fwf(da_path, cb_path, prefix):
    specs     = _parse_colspecs(cb_path)
    flag_vars = [f"{prefix}C040", f"{prefix}C048", f"{prefix}C053"]
    year_vars = [f"{prefix}C043", f"{prefix}C064", f"{prefix}C264", f"{prefix}C267"]
    want  = flag_vars + year_vars
    avail = [v for v in want if v in specs]
    miss  = [v for v in want if v not in specs]
    if miss:
        print(f"  WARNING: {miss} not in codebook {cb_path.name}")
    need     = ["HHID", "PN"] + avail
    colspecs = [specs[v] for v in need]
    raw = pd.read_fwf(da_path, colspecs=colspecs, header=None,
                      names=need, dtype=str, encoding="latin-1")
    raw = raw.map(lambda x: x.strip() if isinstance(x, str) else x)
    raw["HHID"]   = raw["HHID"].str.zfill(6)
    raw["PN"]     = raw["PN"].str.zfill(3)
    raw["hhidpn"] = pd.to_numeric(raw["HHID"] + raw["PN"], errors="coerce")
    raw = raw.drop(columns=["HHID", "PN"]).set_index("hhidpn")
    out = pd.DataFrame(index=raw.index)
    for v in flag_vars:
        s = pd.to_numeric(raw.get(v, pd.Series(np.nan, index=raw.index)), errors="coerce")
        out[v] = s.map({1.0: 1.0, 5.0: 0.0})
    for v in year_vars:
        s = pd.to_numeric(raw.get(v, pd.Series(np.nan, index=raw.index)), errors="coerce")
        out[v] = s.where(s.lt(9998), other=np.nan)   # 9998=DK, 9999=RF -> NaN
    return out


def _cvd_from_csv(csv_path, prefix):
    flag_vars = [f"{prefix}C040", f"{prefix}C048", f"{prefix}C053"]
    year_vars = [f"{prefix}C043", f"{prefix}C064", f"{prefix}C264", f"{prefix}C267"]
    raw  = pd.read_csv(csv_path, dtype=str, low_memory=False)
    raw.columns = raw.columns.str.upper()
    if "HHID" in raw.columns and "PN" in raw.columns:
        raw["HHID"]   = raw["HHID"].str.strip().str.zfill(6)
        raw["PN"]     = raw["PN"].str.strip().str.zfill(3)
        raw["hhidpn"] = pd.to_numeric(raw["HHID"] + raw["PN"], errors="coerce")
    elif "HHIDPN" in raw.columns:
        raw["hhidpn"] = pd.to_numeric(raw["HHIDPN"], errors="coerce")
    raw = raw.set_index("hhidpn")
    out = pd.DataFrame(index=raw.index)
    for v in flag_vars:
        s = pd.to_numeric(raw.get(v, pd.Series(np.nan, index=raw.index)), errors="coerce")
        out[v] = s.map({1.0: 1.0, 5.0: 0.0})
    for v in year_vars:
        s = pd.to_numeric(raw.get(v, pd.Series(np.nan, index=raw.index)), errors="coerce")
        out[v] = s.where(s.lt(9998), other=np.nan)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 1a: Loading analytic wide dataset ...")
df_wide = pd.read_csv(WIDE_CSV)
df_wide["hhidpn"] = pd.to_numeric(df_wide["HHID_PN"], errors="coerce").astype("int64")
df_wide = df_wide.set_index("hhidpn")
print(f"  n = {len(df_wide):,}")

print("Step 1b: Loading multiwave scores ...")
df_scores = pd.read_csv(SCORES_CSV)
df_scores["hhidpn"] = pd.to_numeric(df_scores["HHID_PN"], errors="coerce").astype("int64")
df_scores = df_scores.set_index("hhidpn")

print("Step 1c: Loading CVD subtype flags from raw HMS Section C ...")
cvd_frames = {}
for year, (da_path, cb_path, prefix) in WAVE_C_FWF.items():
    print(f"  {year} ({prefix}C040/048/053) ...", end=" ", flush=True)
    if not da_path.exists():
        print(f"MISSING: {da_path}")
        continue
    frm = _cvd_from_fwf(da_path, cb_path, prefix)
    frm.columns = [f"ha_{year}", f"hf_{year}", f"stroke_{year}",
                   f"ha_yr_{year}", f"stroke_yr_{year}", f"hf_yr_{year}", f"arrhythmia_yr_{year}"]
    n_yr = frm[f"ha_yr_{year}"].notna().sum()
    print(f"n={len(frm):,}  HA={frm[f'ha_{year}'].eq(1).sum():,}  "
          f"HF={frm[f'hf_{year}'].eq(1).sum():,}  Stroke={frm[f'stroke_{year}'].eq(1).sum():,}  "
          f"dx_years_present={n_yr:,}")
    cvd_frames[year] = frm

print(f"  2022 ({CSV_PREFIX}C040/048/053) ...", end=" ", flush=True)
if WAVE_C_CSV.exists():
    frm22 = _cvd_from_csv(WAVE_C_CSV, CSV_PREFIX)
    frm22.columns = ["ha_2022", "hf_2022", "stroke_2022",
                     "ha_yr_2022", "stroke_yr_2022", "hf_yr_2022", "arrhythmia_yr_2022"]
    print(f"n={len(frm22):,}  HA={frm22['ha_2022'].eq(1).sum():,}  "
          f"HF={frm22['hf_2022'].eq(1).sum():,}  Stroke={frm22['stroke_2022'].eq(1).sum():,}")
    cvd_frames[2022] = frm22
else:
    print(f"MISSING: {WAVE_C_CSV}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ASSEMBLE UNIFIED WIDE DATASET
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 2: Assembling wide dataset ...")

df = pd.DataFrame(index=df_wide.index)

for yr in ALL_YEARS:
    df[f"age_{yr}"]        = df_wide.get(f"age_{yr}", pd.Series(np.nan, index=df.index))
    df[f"arrhythmia_{yr}"] = df_wide.get(f"afib_{yr}", pd.Series(np.nan, index=df.index))

for yr in SLEEP_YEARS:
    wn = YEAR_TO_WAVE[yr]
    col = f"sleep_w{wn}"
    df[f"sleep_{yr}"] = df_scores[col].reindex(df.index) if col in df_scores.columns else np.nan

for yr in DEP_YEARS:
    wn = YEAR_TO_WAVE[yr]
    col = f"dep_w{wn}"
    df[f"dep_{yr}"] = df_scores[col].reindex(df.index) if col in df_scores.columns else np.nan

for yr in ALL_YEARS:
    if yr in cvd_frames:
        frm = cvd_frames[yr].reindex(df.index)
        df[f"ha_{yr}"]             = frm[f"ha_{yr}"]
        df[f"hf_{yr}"]             = frm[f"hf_{yr}"]
        df[f"stroke_{yr}"]         = frm[f"stroke_{yr}"]
        df[f"ha_yr_{yr}"]          = frm.get(f"ha_yr_{yr}")
        df[f"hf_yr_{yr}"]          = frm.get(f"hf_yr_{yr}")
        df[f"stroke_yr_{yr}"]      = frm.get(f"stroke_yr_{yr}")
        df[f"arrhythmia_yr_{yr}"]  = frm.get(f"arrhythmia_yr_{yr}")
    else:
        df[f"ha_{yr}"]             = np.nan
        df[f"hf_{yr}"]             = np.nan
        df[f"stroke_{yr}"]         = np.nan
        df[f"ha_yr_{yr}"]          = np.nan
        df[f"hf_yr_{yr}"]          = np.nan
        df[f"stroke_yr_{yr}"]      = np.nan
        df[f"arrhythmia_yr_{yr}"]  = np.nan

# Last observed year for censoring (max year with non-NaN age)
_age_arr  = df[[f"age_{yr}" for yr in ALL_YEARS]].to_numpy()
_yr_arr   = np.where(np.isfinite(_age_arr), np.array(ALL_YEARS), np.nan)
_last_obs = np.nanmax(_yr_arr, axis=1)
df["last_obs_year"] = _last_obs

print(f"  {len(df):,} participants × {len(df.columns)} columns")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TERTILE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 3: Computing top-tertile thresholds ...")

sleep_pool = pd.concat([df[f"sleep_{yr}"] for yr in SLEEP_YEARS
                        if f"sleep_{yr}" in df.columns]).dropna()
dep_pool   = pd.concat([df[f"dep_{yr}"]   for yr in DEP_YEARS
                        if f"dep_{yr}"   in df.columns]).dropna()

sqb_thr = float(np.percentile(sleep_pool, 100 / 3 * 2))
db_thr  = float(np.percentile(dep_pool,   100 / 3 * 2))

print(f"  SQB top-tertile threshold (66.7th pct): {sqb_thr:.2f}  (n={len(sleep_pool):,} obs)")
print(f"  DB  top-tertile threshold (66.7th pct): {db_thr:.2f}  (n={len(dep_pool):,} obs)")

if _SENSITIVITY:
    sqb_thr_q = float(np.percentile(sleep_pool, 75))
    db_thr_q  = float(np.percentile(dep_pool,   75))
    print(f"  [sensitivity] SQB 75th pct: {sqb_thr_q:.2f}")
    print(f"  [sensitivity] DB  75th pct: {db_thr_q:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ANCHOR DETECTION (vectorised)
# ═══════════════════════════════════════════════════════════════════════════════

def _cvd_free_at(year):
    """Boolean Series: True if no CVD flag == 1 at this wave (cumulative ever-flags)."""
    return ~(
        df.get(f"stroke_{year}",     pd.Series(0, index=df.index)).eq(1) |
        df.get(f"ha_{year}",         pd.Series(0, index=df.index)).eq(1) |
        df.get(f"hf_{year}",         pd.Series(0, index=df.index)).eq(1) |
        df.get(f"arrhythmia_{year}", pd.Series(0, index=df.index)).eq(1)
    )


def find_anchor(score_years, score_prefix, threshold):
    """
    Vectorised anchor detection.

    For each participant: first year they cross from below into the top
    tertile (score >= threshold) while CVD-free.  Requires at least one
    prior OBSERVED wave with score < threshold ("ever_below" rolling flag
    to handle biennial gaps, including the missing 2012 sleep wave).

    Returns pd.Series[float] — anchor year (NaN = no valid anchor).
    """
    years_sorted = sorted(score_years)
    score_mat    = pd.DataFrame(
        {yr: df.get(f"{score_prefix}_{yr}", pd.Series(np.nan, index=df.index))
         for yr in years_sorted},
        index=df.index
    )
    anchor     = pd.Series(np.nan, index=df.index, dtype=float)
    ever_below = pd.Series(False, index=df.index)

    for i, yr in enumerate(years_sorted):
        s = score_mat[yr]
        if i > 0:
            new = (
                anchor.isna() &
                s.notna() & s.ge(threshold) &
                ever_below &
                _cvd_free_at(yr)
            )
            anchor = anchor.where(~new, float(yr))
        ever_below = ever_below | (s.notna() & s.lt(threshold))

    return anchor


print("\nStep 4: Detecting anchor waves ...")
df["anchor_sqb"] = find_anchor(SLEEP_YEARS, "sleep", sqb_thr)
df["anchor_db"]  = find_anchor(DEP_YEARS,   "dep",   db_thr)

for col, label in [("anchor_sqb", "SQB"), ("anchor_db", "DB")]:
    n = int(df[col].notna().sum())
    print(f"  {label} anchors: {n:,}")
    print(f"  {label} year distribution:\n"
          f"{df[col].value_counts(dropna=False).sort_index()}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD COMPETING-RISKS DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def _incident_onsets(sub, anc):
    """
    For anchored participants, return a DataFrame of event times (calendar year)
    for each CVD subtype's first post-anchor onset.

    Event time = self-reported diagnosis year when plausible
    (anchor_year <= dx_year <= wave_year); falls back to wave year otherwise.
    """
    onsets = {}
    for subtype, fp in FLAG_PREFIX.items():
        onset = pd.Series(np.nan, index=sub.index, dtype=float)
        for yr in ALL_YEARS:
            col    = f"{fp}_{yr}"
            yr_col = f"{fp}_yr_{yr}"
            if col not in sub.columns:
                continue
            is_first = (
                onset.isna() &
                pd.Series(float(yr), index=sub.index).gt(anc) &
                sub[col].eq(1.0)
            )
            if yr_col in sub.columns:
                dx_yr    = sub[yr_col]
                plausible = dx_yr.notna() & dx_yr.ge(anc) & dx_yr.le(float(yr))
                event_t  = np.where(is_first & plausible, dx_yr, float(yr))
                onset    = onset.where(~is_first, pd.Series(event_t, index=sub.index))
            else:
                onset = onset.where(~is_first, float(yr))
        onsets[subtype] = onset
    return pd.DataFrame(onsets, index=sub.index)


def build_cr_df(anchor_col, label):
    """
    Build a competing-risks dataset: one row per anchored participant.

    Columns
    -------
    duration      — years from anchor to first CVD event or censoring
    event_code    — 0 = censored; 1-5 = first CVD type (see CR_CODES)
    any_cvd_event — 1 if any first CVD event (for KM reference curve)
    anchor_yr     — calendar year of anchor (for reporting)
    """
    print(f"\nStep 5 ({label}): Building competing-risks dataset ...")
    anc  = df[anchor_col].dropna()
    sub  = df.loc[anc.index].copy()
    anc  = anc.copy()

    last_obs = sub["last_obs_year"].fillna(2022.0).clip(upper=2022.0)
    censor_t = (last_obs - anc).clip(lower=0.0)

    onset_df    = _incident_onsets(sub, anc)
    first_cvd   = onset_df.min(axis=1)         # NaN if no event
    has_any     = first_cvd.notna()

    # Duration: time from anchor to first CVD event (or censoring)
    duration = (first_cvd - anc).clip(lower=0.0).where(has_any, other=censor_t)

    # How many subtypes first appear simultaneously at the first CVD wave
    _eq     = onset_df.to_numpy() == first_cvd.values[:, None]
    n_simul = pd.Series(_eq.sum(axis=1), index=sub.index)

    # Assign event code
    event_code = pd.Series(0, index=sub.index, dtype=int)   # 0 = censored

    is_mult = has_any & n_simul.ge(2)
    event_code = event_code.where(~is_mult, 5)              # Multiple Events

    is_single = has_any & n_simul.eq(1)
    for subtype, code in [("Stroke", 1), ("Heart Attack", 2),
                           ("Heart Failure", 3), ("Arrhythmia", 4)]:
        is_this = is_single & (onset_df[subtype] == first_cvd)
        event_code = event_code.where(~is_this, code)

    cr = pd.DataFrame({
        "duration":      duration,
        "event_code":    event_code,
        "any_cvd_event": has_any.astype(int),
        "anchor_yr":     anc,
    }, index=sub.index)

    # Drop zero-follow-up non-events (no information)
    cr = cr[cr["duration"].gt(0.0) | cr["any_cvd_event"].eq(1)]

    print(f"  Anchored: {len(sub):,}  → kept (>0 follow-up): {len(cr):,}")
    print(f"  Anchor year distribution:\n"
          f"{cr['anchor_yr'].value_counts().sort_index()}")
    print(f"  Event type distribution:")
    for subtype, code in [("Any CVD (total)", -1)] + list(CR_CODES.items()):
        if code == -1:
            n = int(cr["any_cvd_event"].sum())
        else:
            n = int((cr["event_code"] == code).sum())
        pct = n / len(cr) * 100
        print(f"    {subtype:<22} {n:4d}  ({pct:.1f}%)")

    return cr


cr_sqb = build_cr_df("anchor_sqb", "SQB")
cr_db  = build_cr_df("anchor_db",  "DB")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def _panel(ax, cr_df, score_label, fig_tag):
    """
    Draw Aalen-Johansen 1-CIF curves per subtype + KM for 'Any CVD'.
    Includes an N-at-risk row below the x-axis.
    """
    n_total = len(cr_df)
    handles = []
    y_max   = 0.0

    ax.text(-0.06, 1.04, fig_tag, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")

    # ── Aalen-Johansen CIF per subtype (1−CIF on y-axis) ──────────────────
    for subtype in SUBTYPE_ORDER:
        if subtype == "Any CVD":
            continue
        code     = CR_CODES[subtype]
        n_events = int((cr_df["event_code"] == code).sum())
        if n_events < 5:
            print(f"  SKIP {subtype}: {n_events} events (< 5)")
            continue

        ajf = AalenJohansenFitter(calculate_variance=True, jitter_level=1e-5)
        ajf.fit(cr_df["duration"], cr_df["event_code"], event_of_interest=code)

        cif_col = f"CIF_{code}"
        t_cif   = ajf.cumulative_density_.index.values
        cif     = ajf.cumulative_density_[cif_col].values

        ci  = ajf.confidence_interval_cumulative_density_
        lo  = ci["AJ_estimate_lower_0.95"].values
        hi  = ci["AJ_estimate_upper_0.95"].values

        color = CVD_COLORS[subtype]
        ax.fill_between(t_cif, lo, hi, step="post", color=color, alpha=0.12, zorder=2)
        ax.step(t_cif, cif, where="post", color=color, lw=1.6, ls="-", zorder=3)

        y_max = max(y_max, float(np.nanmax(hi)))
        handles.append(
            mpatches.Patch(color=color, label=f"{subtype}  ({n_events})")
        )

    # ── KM for Any CVD reference (dashed charcoal) ────────────────────────
    kmf = KaplanMeierFitter()
    kmf.fit(cr_df["duration"], cr_df["any_cvd_event"])
    n_any = int(cr_df["any_cvd_event"].sum())

    t_s     = kmf.survival_function_.index.values
    ci_km   = kmf.confidence_interval_survival_function_
    cif_any = 1.0 - np.array(kmf.survival_function_.iloc[:, 0], dtype=float)
    lo_any  = 1.0 - np.array(ci_km.iloc[:, 1], dtype=float)
    hi_any  = 1.0 - np.array(ci_km.iloc[:, 0], dtype=float)

    ax.fill_between(t_s, lo_any, hi_any, step="post",
                    color=CVD_COLORS["Any CVD"], alpha=0.12, zorder=2)
    ax.step(t_s, cif_any, where="post",
            color=CVD_COLORS["Any CVD"], lw=2.2, ls="--", zorder=3)

    y_max = max(y_max, float(np.nanmax(hi_any)))
    handles.insert(0, mpatches.Patch(
        color=CVD_COLORS["Any CVD"],
        label=f"Any CVD  ({n_any})"
    ))

    # ── Axis formatting ────────────────────────────────────────────────────
    ax.set_xlabel("Years from first high-burden wave", labelpad=6)
    ax.set_ylabel("Cumulative incidence (%)", labelpad=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    y_ceil = 0.25
    ax.set_ylim(bottom=0, top=y_ceil)
    # xlim set AFTER N-at-risk text to prevent text x-coords from expanding axis

    ax.set_title(
        f"Cumulative Incidence of CVD — First High {score_label} Burden\n"
        f"Anchor: top tertile of cohort-pooled {score_label.lower()} burden; "
        f"CVD-free at anchor; n = {n_total:,}",
        fontsize=9.5, pad=8
    )

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(handles=handles, loc="upper left", fontsize=8.0,
              framealpha=0.93, edgecolor="#CCCCCC",
              title="First incident CVD type (AJ CIF)", title_fontsize=7.5)

    # ── Numbers at risk ────────────────────────────────────────────────────
    max_t     = cr_df["duration"].max()
    nar_times = [t for t in [2, 4, 6, 8, 10, 12] if t <= max_t + 0.5]
    nar_vals  = [int((cr_df["duration"] >= t).sum()) for t in nar_times]

    trans = ax.get_xaxis_transform()   # data-x, axes-fraction-y
    ax.text(1.7, -0.18, "N at risk:", ha="right", va="top",
            fontsize=7.5, color="#444444", transform=trans, clip_on=False)
    for t_n, n_n in zip(nar_times, nar_vals):
        ax.text(t_n, -0.18, str(n_n), ha="center", va="top",
                fontsize=7.5, color="#444444", transform=trans, clip_on=False)

    # Crop the flat 0-2 dead zone; first observable events appear at t=2 (biennial)
    ax.set_xlim(left=1.8, right=max_t + 0.5)
    ax.set_xticks([t for t in [2, 4, 6, 8, 10, 12] if t <= max_t + 0.5])

    # ── Footnote ──────────────────────────────────────────────────────────
    """ax.text(
        0.0, -0.29,
        "Entry: first observed wave in top tertile (observed transition from below required), "
        "free of all CVD subtypes.  "
        "Exit: first incident CVD type or last observed wave (censored, capped 2022).  "
        "Estimator: Aalen–Johansen CIF (1−CIF plotted); competing subtypes modelled, "
        "not censored.  Descriptive / unadjusted; complements Cox models (Figs. 4–5).  "
        "HRS biennial: event times are interval-censored within 2-year waves.",
        transform=ax.transAxes, fontsize=6.5, color="#555555", va="top"
    )"""


def make_figure(cr, score_label, fig_tag, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _panel(ax, cr, score_label, fig_tag)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_combined(cr_a, label_a, cr_b, label_b, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.5), sharey=False)
    _panel(axes[0], cr_a, label_a, "a")
    _panel(axes[1], cr_b, label_b, "b")
    fig.suptitle(
        "CVD-Free Survival After First High Sleep / Depression Burden  "
        "(Aalen–Johansen competing-risks CIF)",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.subplots_adjust(bottom=0.28, wspace=0.32)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("\nStep 5b: Comparing SQB vs DB event proportions (matched 8-year window) ...")
from scipy.stats import chi2_contingency

# Truncate DB at 8 years to match SQB maximum follow-up
cr_db_trunc = cr_db.copy()
late_event  = cr_db_trunc["duration"].gt(8) & cr_db_trunc["event_code"].gt(0)
late_censor = cr_db_trunc["duration"].gt(8) & cr_db_trunc["event_code"].eq(0)
cr_db_trunc.loc[late_event | late_censor, "duration"]  = 8.0
cr_db_trunc.loc[late_event, "event_code"]              = 0   # censor late events
cr_db_trunc.loc[late_event, "any_cvd_event"]           = 0

n_sqb_evt  = int(cr_sqb["any_cvd_event"].sum())
n_sqb_tot  = len(cr_sqb)
n_db_evt   = int(cr_db_trunc["any_cvd_event"].sum())
n_db_tot   = len(cr_db_trunc)

table = [[n_sqb_evt, n_sqb_tot - n_sqb_evt],
         [n_db_evt,  n_db_tot  - n_db_evt]]
chi2, p, dof, _ = chi2_contingency(table)

print(f"  SQB (full):        {n_sqb_evt}/{n_sqb_tot} = {n_sqb_evt/n_sqb_tot*100:.1f}%")
print(f"  DB  (trunc 8yr):   {n_db_evt}/{n_db_tot}  = {n_db_evt/n_db_tot*100:.1f}%")
print(f"  chi2={chi2:.3f}  p={p:.4f}  (2x2 contingency, matched 8-yr window)")
print(f"  DB subtype breakdown (truncated at 8yr):")
for subtype, code in CR_CODES.items():
    n = int((cr_db_trunc["event_code"] == code).sum())
    print(f"    {subtype:<22} {n:4d}  ({n/n_db_tot*100:.1f}%)")

print("\nStep 6: Plotting ...")
make_figure(cr_sqb,      "Sleep Quality", "a", OUT_DIR / "Figure_7a.png")
make_figure(cr_db_trunc, "Depression",    "b", OUT_DIR / "Figure_7b.png")
make_combined(cr_sqb, "Sleep Quality", cr_db_trunc, "Depression",
              OUT_DIR / "Figure_7_combined.png")

if _SENSITIVITY:
    print("\n[sensitivity] Re-running at top quartile ...")
    df["anchor_sqb_q"] = find_anchor(SLEEP_YEARS, "sleep", sqb_thr_q)
    df["anchor_db_q"]  = find_anchor(DEP_YEARS,   "dep",   db_thr_q)
    cr_sqb_q = build_cr_df("anchor_sqb_q", "SQB (quartile)")
    cr_db_q  = build_cr_df("anchor_db_q",  "DB  (quartile)")
    make_combined(cr_sqb_q, "Sleep Quality (quartile)",
                  cr_db_q,  "Depression (quartile)",
                  OUT_DIR / "Figure_7_sensitivity_quartile.png")

print("\n" + "=" * 68)
print("  DONE")
for f in ["Figure_7a.png", "Figure_7b.png", "Figure_7_combined.png"]:
    print(f"  {f} → {OUT_DIR / f}")
print("=" * 68)
