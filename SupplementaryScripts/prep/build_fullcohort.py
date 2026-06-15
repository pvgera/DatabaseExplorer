"""
build_fullcohort.py
===================
Phase 1 — Full-cohort analytic dataset (n ≈ 45 K).

Backbone: RAND HRS 1992-2022 longitudinal file — CESD, sleep, covariates.
Arrhythmia: raw Section C fixed-width .da files, waves 2010-2022 (7 waves).

Arrhythmia ascertainment
------------------------
  [prefix]C266 == 1  → EVER told by doctor: abnormal heart rhythm (primary)
  [prefix]C270M1/M2 == 1 → "Type heart disease = abnormal rhythm" catch-all
  Waves 2002-2008 have NO arrhythmia-specific question (introduced wave 10 / 2010).
  First positive wave = onset wave (earliest wave with either indicator positive).

Wave / file map
---------------
  2010 wave 10 M  HRS/Raw Data/HRS2010/H10C_R.da          vars: MC266, MC267, MC270M1, MC270M2
  2012 wave 11 N  HRS/Raw Data/HRS2012/H12C_R.da          vars: NC266, NC267, NC270M1, NC270M2
  2014 wave 12 O  HRS/Raw Data/HRS2014/H14C_R.da          vars: OC266, OC267, OC270M1, OC270M2
  2016 wave 13 P  HRS/HMS2016/h16da/H16C_R.da        vars: PC266, PC267, PC270M1, PC270M2
  2018 wave 14 Q  HRS/Raw Data/HRS2018/h18da/H18C_R.da    vars: QC266, QC267, QC270M1, QC270M2
  2020 wave 15 R  HRS/Raw Data/HRS2020/h20da/H20C_R.da    vars: RC266, RC267, RC270M1, RC270M2
  2022 wave 16 S  HRS/HMS2022/H22csv/h22c_r.csv      vars: SC266, SC267, SC270M1, SC270M2

Outputs
-------
  data/hrs_analytic_wide_fullcohort.csv   — one row per participant
  data/hrs_cox_long_fullcohort.csv        — counting-process format for analysis1_cox.py
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).parents[2]
OUT    = REPO / "SF_OUTPUT" / "analytic"   # same dir as pilot CSVs so Phase 2 scripts need only one-line path change
OUT.mkdir(parents=True, exist_ok=True)

RAND_DTA = REPO / "RawData" / "HRS" / "RandHRS" / "randhrs1992_2022v1.dta"

# (year, wave_n, prefix, .da path, codebook path)
WAVE_CONFIG = [
    (2010, 10, "M",
     REPO / "RawData" / "HRS" / "waves" / "HRS2010" / "H10C_R.da",
     REPO / "RawData" / "HRS" / "waves" / "HRS2010" / "H10C_R.txt"),
    (2012, 11, "N",
     REPO / "RawData" / "HRS" / "waves" / "HRS2012" / "H12C_R.da",
     REPO / "RawData" / "HRS" / "waves" / "HRS2012" / "H12C_R.txt"),
    (2014, 12, "O",
     REPO / "RawData" / "HRS" / "waves" / "HRS2014" / "H14C_R.da",
     REPO / "RawData" / "HRS" / "waves" / "HRS2014" / "H14C_R.txt"),
    (2016, 13, "P",
     REPO / "RawData" / "HRS" / "HMS2016" / "h16da" / "H16C_R.da",
     REPO / "RawData" / "HRS" / "HMS2016" / "h16cb" / "H16C_R.txt"),
    (2018, 14, "Q",
     REPO / "RawData" / "HRS" / "waves" / "HRS2018" / "h18da" / "H18C_R.da",
     REPO / "RawData" / "HRS" / "waves" / "HRS2018" / "h18cb" / "H18C_R.txt"),
    (2020, 15, "R",
     REPO / "RawData" / "HRS" / "waves" / "HRS2020" / "h20da" / "H20C_R.da",
     REPO / "RawData" / "HRS" / "waves" / "HRS2020" / "h20cb" / "H20C_R.txt"),
]
# 2022 uses the CSV release (no fixed-width parsing needed)
CSV_2022 = REPO / "RawData" / "HRS" / "HMS2022" / "H22csv" / "h22c_r.csv"

AFIB_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022]

print()
print("=" * 70)
print("  BUILD FULL-COHORT ANALYTIC DATASET — RAND HRS 45 K")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_colspecs(cb_path):
    """Return {VARNAME: (start_byte, end_byte)} from HRS fixed-width codebook."""
    specs = {}
    pos   = 0
    varname_pat = re.compile(r'^([A-Z][A-Z0-9_]+)\s+\S')
    meta_pat    = re.compile(r'Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)', re.I)
    cur = None
    with open(cb_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if (line and not line.startswith(" ") and not line.startswith("=")
                    and not line.startswith("-") and not line.startswith("{")):
                m = varname_pat.match(line)
                if m:
                    cur = m.group(1)
            m2 = meta_pat.search(line)
            if m2 and cur:
                specs[cur] = (pos, pos + int(m2.group(1)))
                pos += int(m2.group(1))
                cur = None
    return specs


def _load_da(da_path, cb_path, variables):
    """Load selected variables from HRS fixed-width .da file. Returns DataFrame indexed by hhidpn."""
    all_specs = _parse_colspecs(cb_path)
    need      = ["HHID", "PN"] + [v.upper() for v in variables if v.upper() in all_specs]
    missing   = [v for v in variables if v.upper() not in all_specs]
    if missing:
        print(f"    WARNING — not in codebook: {missing}")
    colspecs  = [all_specs[v] for v in need]
    df = pd.read_fwf(da_path, colspecs=colspecs, header=None,
                     names=need, dtype=str, encoding="latin-1")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["HHID"] = df["HHID"].str.zfill(6)
    df["PN"]   = df["PN"].str.zfill(3)
    df["hhidpn"] = pd.to_numeric(df["HHID"] + df["PN"], errors="coerce")
    return df.set_index("hhidpn")


def _afib_from_da(da_path, cb_path, prefix):
    """Load arrhythmia indicator and year-first from a Section C .da file."""
    v266  = f"{prefix}C266"
    v267  = f"{prefix}C267"
    v270a = f"{prefix}C270M1"
    v270b = f"{prefix}C270M2"
    df = _load_da(da_path, cb_path, [v266, v267, v270a, v270b])

    c266  = pd.to_numeric(df.get(v266,  pd.Series(dtype=str)), errors="coerce")
    c267  = pd.to_numeric(df.get(v267,  pd.Series(dtype=str)), errors="coerce")
    c270a = pd.to_numeric(df.get(v270a, pd.Series(dtype=str)), errors="coerce")
    c270b = pd.to_numeric(df.get(v270b, pd.Series(dtype=str)), errors="coerce")

    # arrhythmia positive = C266==1 OR C270M1==1 OR C270M2==1
    afib = ((c266 == 1) | (c270a == 1) | (c270b == 1)).astype(float)
    # Mark truly missing (no response at all) as NaN
    has_response = c266.notna() | c270a.notna() | c270b.notna()
    afib = afib.where(has_response)

    # Year first diagnosed: valid 1920-2022, 9998/9999 = DK/RF
    yr_valid = c267.where(c267.between(1920, 2022))

    return afib, yr_valid


def _afib_from_csv(csv_path, prefix):
    """Load arrhythmia from 2022 CSV release."""
    df = pd.read_csv(csv_path, dtype=str, encoding="latin-1")
    df["HHID"] = df["HHID"].str.strip().str.zfill(6)
    df["PN"]   = df["PN"].str.strip().str.zfill(3)
    df["hhidpn"] = pd.to_numeric(df["HHID"] + df["PN"], errors="coerce")
    df = df.set_index("hhidpn")

    v266  = prefix + "C266"
    v267  = prefix + "C267"
    v270a = prefix + "C270M1"
    v270b = prefix + "C270M2"

    def _get(col):
        return pd.to_numeric(df.get(col, pd.Series(dtype=str)), errors="coerce")

    c266  = _get(v266)
    c267  = _get(v267)
    c270a = _get(v270a)
    c270b = _get(v270b)

    afib = ((c266 == 1) | (c270a == 1) | (c270b == 1)).astype(float)
    has_response = c266.notna() | c270a.notna() | c270b.notna()
    afib = afib.where(has_response)
    yr_valid = c267.where(c267.between(1920, 2022))

    return afib, yr_valid


def _cat_to_num(series):
    """Convert Stata categorical labels like '1.most of the time' → float."""
    return pd.to_numeric(
        series.astype(str).str.split(".").str[0], errors="coerce"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAND HRS BACKBONE
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 1: Loading RAND HRS longitudinal file ...")

DEP_WAVES   = list(range(2, 17))           # waves 2-16 (1994-2022)
SLEEP_WAVES = [n for n in range(6, 17) if n not in (9, 11)]  # excl. sub-samples
COV_WAVES   = list(range(10, 17))          # covariate waves 10-16 (matching afib waves)

dep_cols   = [f"r{n}cesd" for n in DEP_WAVES]
sleep_cols = [f"r{n}sleep{s}" for n in SLEEP_WAVES for s in ("fal", "wkn", "wke", "rt")]
cov_cols   = (
    [f"r{n}agey_b" for n in COV_WAVES]
    + [f"r{n}bmi"   for n in COV_WAVES]
    + [f"r{n}hibpe" for n in COV_WAVES]
    + [f"r{n}diabe" for n in COV_WAVES]
    + [f"r{n}height" for n in COV_WAVES]
    + [f"r{n}weight" for n in COV_WAVES]
)

rand_cols = ["hhidpn", "ragender"] + dep_cols + sleep_cols + cov_cols
print(f"  Requesting {len(rand_cols)} columns ...")

df_rand = pd.read_stata(RAND_DTA, columns=rand_cols)
df_rand["hhidpn"] = df_rand["hhidpn"].astype("int64")
df_rand = df_rand.set_index("hhidpn")
print(f"  RAND HRS participants: {len(df_rand):,}")

# Sex
gender_raw    = _cat_to_num(df_rand["ragender"])
df_rand["sex_female"] = np.where(gender_raw == 2, 1.0, np.where(gender_raw == 1, 0.0, np.nan))
print(f"  sex_female available: {df_rand['sex_female'].notna().sum():,}  "
      f"female={df_rand['sex_female'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SCORE CESD AND SLEEP FROM RAND HRS
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Pre-scoring CESD and sleep at all waves ...")

WAVE_YEAR = {
    1: 1992,  2: 1994,  3: 1996,  4: 1998,  5: 2000,
    6: 2002,  7: 2004,  8: 2006,  9: 2008, 10: 2010,
   11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022,
}


def _score_cesd(wave_n):
    col = f"r{wave_n}cesd"
    if col not in df_rand.columns:
        return pd.Series(np.nan, index=df_rand.index)
    raw = pd.to_numeric(df_rand[col], errors="coerce")
    return raw.where(raw.between(0, 8)) * 12.5


def _score_sleep(wave_n):
    if wave_n in (9, 11):
        return pd.Series(np.nan, index=df_rand.index)
    cols = {s: f"r{wave_n}sleep{s}" for s in ("fal", "wkn", "wke", "rt")}
    if not all(c in df_rand.columns for c in cols.values()):
        return pd.Series(np.nan, index=df_rand.index)

    def trouble(s):
        v = _cat_to_num(df_rand[s]).where(lambda x: x.isin([1, 2, 3]))
        return (v - 3.0) / (1.0 - 3.0) * 10.0     # 1=worst→10, 3=best→0

    def rested(s):
        v = _cat_to_num(df_rand[s]).where(lambda x: x.isin([1, 2, 3]))
        return (v - 1.0) / (3.0 - 1.0) * 10.0     # 1=best→0, 3=worst→10

    s_fal = trouble(cols["fal"])
    s_wkn = trouble(cols["wkn"])
    s_wke = trouble(cols["wke"])
    s_rt  = rested(cols["rt"])

    total   = s_fal.fillna(0) + s_wkn.fillna(0) + s_wke.fillna(0) + s_rt.fillna(0)
    n_valid = s_fal.notna().astype(int) + s_wkn.notna().astype(int) + \
              s_wke.notna().astype(int) + s_rt.notna().astype(int)
    return (total / n_valid * 10.0).where(n_valid >= 2)


scored = {}
for n in DEP_WAVES:
    s = _score_cesd(n)
    scored[f"dep_w{n}"] = s
    print(f"  dep   w{n:2d} ({WAVE_YEAR[n]}): n={s.notna().sum():6,}")

for n in SLEEP_WAVES:
    s = _score_sleep(n)
    scored[f"sleep_w{n}"] = s
    print(f"  sleep w{n:2d} ({WAVE_YEAR[n]}): n={s.notna().sum():6,}")

df_scores = pd.DataFrame(scored)  # index = hhidpn

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COVARIATES FROM RAND HRS (per wave)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Extracting covariates from RAND HRS ...")

cov_data = {}
for n in COV_WAVES:
    yr = WAVE_YEAR[n]
    age = pd.to_numeric(df_rand.get(f"r{n}agey_b", pd.Series(dtype=float)), errors="coerce")
    cov_data[f"age_{yr}"] = age.where(age.between(30, 120))

    bmi = pd.to_numeric(df_rand.get(f"r{n}bmi", pd.Series(dtype=float)), errors="coerce")
    cov_data[f"bmi_{yr}"] = bmi.where(bmi.between(12, 75))

    hibpe = _cat_to_num(df_rand.get(f"r{n}hibpe", pd.Series(dtype=str)))
    cov_data[f"hibpe_{yr}"] = hibpe.where(hibpe.isin([0, 1]))

    diabe = _cat_to_num(df_rand.get(f"r{n}diabe", pd.Series(dtype=str)))
    cov_data[f"diabe_{yr}"] = diabe.where(diabe.isin([0, 1]))

    # Height in metres → cm (use 2016 height as stable; update if needed)
    ht_m  = pd.to_numeric(df_rand.get(f"r{n}height", pd.Series(dtype=float)), errors="coerce")
    ht_cm = (ht_m * 100.0).where(ht_m.between(1.0, 2.5))
    cov_data[f"height_cm_{yr}"] = ht_cm

    wt = pd.to_numeric(df_rand.get(f"r{n}weight", pd.Series(dtype=float)), errors="coerce")
    cov_data[f"weight_kg_{yr}"] = wt.where(wt.between(25, 300))

df_cov = pd.DataFrame(cov_data)  # index = hhidpn

# Convenience aliases for 2016 (required by analysis1_cox.py)
df_cov["age_2016"]    = df_cov["age_2016"]          # already named correctly
df_cov["bmi_2016"]    = df_cov["bmi_2016"]
df_cov["hypertension"] = df_cov["hibpe_2016"]
df_cov["diabetes"]     = df_cov["diabe_2016"]
df_cov["height_cm"]    = df_cov["height_cm_2016"]   # treat 2016 height as stable
df_cov["weight_kg"]    = df_cov["weight_kg_2016"]

print(f"  age_2016:    n={df_cov['age_2016'].notna().sum():,}  "
      f"mean={df_cov['age_2016'].mean():.1f}")
print(f"  hypertension: n={df_cov['hypertension'].notna().sum():,}  "
      f"prev={df_cov['hypertension'].mean()*100:.1f}%")
print(f"  diabetes:     n={df_cov['diabetes'].notna().sum():,}  "
      f"prev={df_cov['diabetes'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ARRHYTHMIA FROM SECTION C .da / .csv
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4: Parsing arrhythmia from Section C files (2010-2022) ...")

afib_frames = {}
yr_first_frames = {}

for year, wave_n, prefix, da_path, cb_path in WAVE_CONFIG:
    print(f"  {year} (wave {wave_n}, {prefix}C266) ...", end=" ")
    if not da_path.exists():
        print(f"FILE MISSING: {da_path}")
        continue
    afib, yr_first = _afib_from_da(da_path, cb_path, prefix)
    n_pos = int(afib.sum())
    n_tot = int(afib.notna().sum())
    print(f"n={n_tot:,}  arrhythmia={n_pos:,} ({n_pos/n_tot*100:.1f}%)")
    afib_frames[f"afib_{year}"]    = afib.rename(f"afib_{year}")
    yr_first_frames[f"yr_first_{year}"] = yr_first.rename(f"yr_first_{year}")

# 2022 from CSV
print(f"  2022 (wave 16, SC266) ...", end=" ")
if CSV_2022.exists():
    afib22, yr22 = _afib_from_csv(CSV_2022, "S")
    n_pos = int(afib22.sum())
    n_tot = int(afib22.notna().sum())
    print(f"n={n_tot:,}  arrhythmia={n_pos:,} ({n_pos/n_tot*100:.1f}%)")
    afib_frames["afib_2022"]    = afib22.rename("afib_2022")
    yr_first_frames["yr_first_2022"] = yr22.rename("yr_first_2022")
else:
    print(f"FILE MISSING: {CSV_2022}")

df_afib   = pd.DataFrame(afib_frames)      # index = hhidpn, cols = afib_2010..2022
df_yr_fst = pd.DataFrame(yr_first_frames)  # index = hhidpn

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b — SLEEP APNEA FROM 2016 SECTION C (first wave with this question)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4b: Extracting sleep apnea diagnosis from 2016 Section C ...")

_da16  = REPO / "RawData" / "HRS" / "HMS2016" / "h16da" / "H16C_R.da"
_cb16  = REPO / "RawData" / "HRS" / "HMS2016" / "h16cb" / "H16C_R.txt"
df_sa16 = _load_da(_da16, _cb16, ["PC291", "PC292"])
_pc291 = pd.to_numeric(df_sa16.get("PC291", pd.Series(dtype=str)), errors="coerce")
_pc292 = pd.to_numeric(df_sa16.get("PC292", pd.Series(dtype=str)), errors="coerce")
# 1 = told have sleep apnea; 0 = question answered but not apnea; NaN = no response
df_sleep_apnea = pd.Series(
    np.where(_pc292 == 1, 1.0, np.where(_pc291.notna(), 0.0, np.nan)),
    index=df_sa16.index, name="sleep_apnea"
)
n_apnea = int((df_sleep_apnea == 1).sum())
print(f"  Sleep apnea (PC292=1): n={n_apnea:,}  "
      f"prev={n_apnea / df_sleep_apnea.notna().sum() * 100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DETERMINE ONSET YEAR
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Determining arrhythmia onset ...")

# Earliest wave year with afib == 1
def _earliest_positive(row):
    for yr in AFIB_YEARS:
        col = f"afib_{yr}"
        if col in row.index and row[col] == 1:
            return float(yr)
    return np.nan

afib_onset_wave = df_afib.apply(_earliest_positive, axis=1)
afib_ever       = (df_afib[df_afib.columns] == 1).any(axis=1).astype(float)

print(f"  Onset wave distribution:")
print(afib_onset_wave.value_counts(dropna=False).sort_index().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ASSEMBLE WIDE DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6: Assembling wide dataset ...")

# Start from RAND HRS index (45 K participants)
df_wide = pd.DataFrame(index=df_rand.index)
df_wide["sex_female"] = df_rand["sex_female"]

# Covariates — year-based names (age_2010, age_2012, ..., age_2022)
# NOTE: dep_w*/sleep_w* columns are intentionally NOT included here.
# The multiwave script loads RAND HRS independently and joins its own scored
# dep/sleep data. Adding them here would create column conflicts in that join.
df_wide = df_wide.join(df_cov, how="left")

# Arrhythmia per wave
df_wide = df_wide.join(df_afib, how="left")
df_wide = df_wide.join(df_yr_fst, how="left")
df_wide = df_wide.join(df_sleep_apnea, how="left")
df_wide["afib_onset_wave"] = afib_onset_wave
df_wide["afib_ever"]       = afib_ever

# 2016-perspective flags (for analysis1_cox.py compatibility)
df_wide["afib_prevalent_2016"] = (
    df_wide["afib_onset_wave"].notna() & (df_wide["afib_onset_wave"] <= 2016)
).astype(float)

df_wide["afib_incident"] = (
    df_wide["afib_onset_wave"].notna() & (df_wide["afib_onset_wave"] > 2016)
).astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE WIDE CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 7: Saving wide CSV ...")

df_wide.index.name = "HHID_PN"      # backwards-compatible with existing scripts
wide_path = OUT / "hrs_analytic_wide_fullcohort.csv"
df_wide.reset_index().to_csv(wide_path, index=False)
print(f"  Saved: {wide_path.name}  ({len(df_wide):,} rows × {len(df_wide.columns)} cols)")

# Summary stats
n_total    = len(df_wide)
n_prev     = int(df_wide["afib_prevalent_2016"].sum())
n_incident = int(df_wide["afib_incident"].sum())
n_analytic = n_total - n_prev
print(f"\n  Total participants:        {n_total:,}")
print(f"  Prevalent arrhythmia 2016: {n_prev:,} ({n_prev/n_total*100:.1f}%)")
print(f"  Analytic (non-prevalent):  {n_analytic:,}")
print(f"  Incident arrhythmia:       {n_incident:,} ({n_incident/n_analytic*100:.1f}% of analytic)")
print(f"\n  Onset wave distribution:")
print(df_wide["afib_onset_wave"].value_counts(dropna=False).sort_index().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — BUILD LONG (COUNTING-PROCESS) CSV FOR analysis1_cox.py
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 8: Building Cox long format (2016 baseline, non-prevalent) ...")

analytic = df_wide[df_wide["afib_prevalent_2016"] == 0].copy()

# Add scored dep/sleep (not in df_wide to avoid multiwave script conflicts)
analytic["dep_2016"]   = df_scores["dep_w13"].reindex(analytic.index)
analytic["sleep_2016"] = df_scores["sleep_w13"].reindex(analytic.index)
analytic["dep_2022"]   = df_scores["dep_w16"].reindex(analytic.index)
analytic["sleep_2022"] = df_scores["sleep_w16"].reindex(analytic.index)
analytic["dep_change"] = analytic["dep_2022"] - analytic["dep_2016"]

analytic["tstart"] = 0.0
analytic["tstop"]  = np.where(
    analytic["afib_incident"] == 1,
    analytic["afib_onset_wave"] - 2016,
    6.0     # censored at 2022 (6 years from 2016)
)
analytic["tstop"] = analytic["tstop"].clip(lower=0.01)
analytic["event"] = analytic["afib_incident"].fillna(0).astype(int)

cox_cols = [
    "HHID_PN", "tstart", "tstop", "event",
    "dep_2016", "sleep_2016", "dep_change",
    "age_2016", "height_cm", "weight_kg", "hypertension", "diabetes",
    "sex_female", "bmi_2016", "sleep_apnea",
]

analytic.index.name = "HHID_PN"
cox_df = analytic.reset_index()[
    [c for c in cox_cols if c in analytic.reset_index().columns]
]

long_path = OUT / "hrs_cox_long_fullcohort.csv"
cox_df.to_csv(long_path, index=False)
print(f"  Saved: {long_path.name}  ({len(cox_df):,} rows, {int(cox_df['event'].sum())} events)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — 2010-BASELINE COX LONG CSV (preferred baseline: first arrhythmia wave)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 9: Building Cox long format — 2010 baseline (first arrhythmia wave) ...")

# Prevalent at 2010 = anyone with afib_2010 == 1 (first wave with the question;
# we can't know if they had arrhythmia before 2010, so they are excluded)
analytic10 = df_wide[df_wide.get("afib_2010", pd.Series(0, index=df_wide.index)) != 1].copy()

analytic10["dep_2010"]   = df_scores["dep_w10"].reindex(analytic10.index)
analytic10["sleep_2010"] = df_scores["sleep_w10"].reindex(analytic10.index)
analytic10["dep_2022"]   = df_scores["dep_w16"].reindex(analytic10.index)
analytic10["dep_change"] = analytic10["dep_2022"] - analytic10["dep_2010"]

analytic10["age_2010"]   = df_cov["age_2010"].reindex(analytic10.index)
analytic10["bmi_2010"]   = df_cov["bmi_2010"].reindex(analytic10.index)
analytic10["hypertension"] = df_cov["hibpe_2010"].reindex(analytic10.index)
analytic10["diabetes"]     = df_cov["diabe_2010"].reindex(analytic10.index)
analytic10["height_cm"]    = df_cov["height_cm_2010"].reindex(analytic10.index)
analytic10["weight_kg"]    = df_cov["weight_kg_2010"].reindex(analytic10.index)

# Incident = first positive AFTER 2010
has_onset = analytic10["afib_onset_wave"].notna()
incident  = has_onset & (analytic10["afib_onset_wave"] > 2010)

analytic10["tstart"] = 0.0
analytic10["tstop"]  = analytic10["afib_onset_wave"].where(incident, 2022.0) - 2010
analytic10["tstop"]  = analytic10["tstop"].clip(lower=0.01)
analytic10["event"]  = incident.astype(int)

cox10_cols = [
    "HHID_PN", "tstart", "tstop", "event",
    "dep_2010", "sleep_2010", "dep_change",
    "age_2010", "height_cm", "weight_kg", "hypertension", "diabetes",
    "sex_female", "bmi_2010",
]

analytic10.index.name = "HHID_PN"
cox10_df = analytic10.reset_index()[
    [c for c in cox10_cols if c in analytic10.reset_index().columns]
]

long10_path = OUT / "hrs_cox_long_fullcohort_2010base.csv"
cox10_df.to_csv(long10_path, index=False)
print(f"  Saved: {long10_path.name}  ({len(cox10_df):,} rows, {int(cox10_df['event'].sum())} events)")
print(f"  Follow-up window: 2010–2022 (12 years)")
print(f"  Excluded prevalent at 2010: {int((df_wide.get('afib_2010', pd.Series(0, index=df_wide.index)) == 1).sum()):,}")

print()
print("=" * 70)
print("  DONE — Phase 1 complete")
print(f"  Wide CSV:         {wide_path}")
print(f"  Long CSV (2016):  {long_path}")
print(f"  Long CSV (2010):  {long10_path}")
print("=" * 70)
