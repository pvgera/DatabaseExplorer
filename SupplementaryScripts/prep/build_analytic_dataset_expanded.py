"""
build_analytic_dataset_expanded.py
===================================
Expanded cohort: full HRS 2016 respondent population (ages 50+, no upper
age ceiling).  Replaces the HRS CSV (which was restricted to the
50-60 birth-year cohort) with direct parsing of the raw .da files.

Sources
-------
  HRS/HMS2016/h16da/   -- 2016 baseline (Sections A, B, C, D, PR)
  HRS/Raw Data/HRS2018/h18da/H18C_R.da  -- 2018 AFib follow-up
  HRS/Raw Data/HRS2020/h20da/H20C_R.da  -- 2020 AFib follow-up
  HRS/HMS2022/H22csv/h22c_r.csv    -- 2022 AFib follow-up (CSV)

AFib ascertainment (4-wave)
---------------------------
  PC270M1/M2 = 1  at 2016 -> prevalent, excluded from incident analysis
  QC270M1/M2 = 1  at 2018 -> onset wave 2018 (2 years post-baseline)
  RC270M1/M2 = 1  at 2020 -> onset wave 2020 (4 years)
  SC270M1/M2 = 1  at 2022 -> onset wave 2022 (6 years)
  Code 1 = "Abnormal heart rhythm (AFib / palpitations / pacemaker)"

Depression scoring (CES-D 8, H16D_R.da)
----------------------------------------
  Items PD110-PD117: binary 1=Yes 5=No
  Direct items (yes=burden): PD110 PD111 PD112 PD114 PD116 PD117
  Reversed items (yes=healthy): PD113 PD115
  Score = (sum normalised / max possible) * 100    [0-100, higher=worse]

Sleep scoring (H16C_R.da, PC083-PC086)
----------------------------------------
  3-point ordinal: 1=Most of the time  2=Sometimes  3=Rarely/never
  Symptom items (min=3, max=1): PC083 PC084 PC085
  Positive item (min=1, max=3): PC086 (rested)
  Score = (sum normalised / max possible) * 100    [0-100, higher=worse]

Outputs
-------
  data/hrs_analytic_wide_expanded.csv   -- one row per participant (full cohort)
  data/hrs_cox_long_expanded.csv        -- counting-process format
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[2]

H16_DA = REPO_ROOT / "RawData" / "HRS" / "HMS2016" / "h16da"
H16_CB = REPO_ROOT / "RawData" / "HRS" / "HMS2016" / "h16cb"

H18_DA_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2018" / "h18da" / "H18C_R.da"
H18_CB_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2018" / "h18cb" / "H18C_R.txt"

H20_DA_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2020" / "h20da" / "H20C_R.da"
H20_CB_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2020" / "h20cb" / "H20C_R.txt"

CSV_C22  = REPO_ROOT / "RawData" / "HRS" / "HMS2022" / "H22csv" / "h22c_r.csv"

OUT_DIR  = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "analytic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print()
print("=" * 70)
print("  BUILD EXPANDED ANALYTIC DATASET -- Full HRS 2016 Cohort (ages 50+)")
print("=" * 70)

# ---------------------------------------------------------------------------
# HELPERS  (same parsing logic as build_analytic_dataset.py)
# ---------------------------------------------------------------------------

def _parse_colspecs(cb_path):
    specs = {}; pos = 0
    varname_pat = re.compile(r'^([A-Z][A-Z0-9_]+)\s+\S')
    meta_pat    = re.compile(r'Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)', re.I)
    cur = None
    with open(cb_path, encoding='latin-1') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if (line and not line.startswith(' ') and not line.startswith('=')
                    and not line.startswith('-') and not line.startswith('{')):
                m = varname_pat.match(line)
                if m: cur = m.group(1)
            m2 = meta_pat.search(line)
            if m2 and cur:
                specs[cur] = (pos, pos + int(m2.group(1)))
                pos += int(m2.group(1)); cur = None
    return specs


def _load_da(da_path, cb_path, variables):
    all_specs = _parse_colspecs(cb_path)
    need      = ['HHID', 'PN'] + [v.upper() for v in variables if v.upper() in all_specs]
    missing   = [v for v in variables if v.upper() not in all_specs]
    if missing:
        print(f"    WARNING: not in codebook: {missing}")
    colspecs  = [all_specs[v] for v in need]
    df = pd.read_fwf(da_path, colspecs=colspecs, header=None,
                     names=need, dtype=str, encoding='latin-1')
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df['HHID']    = df['HHID'].str.zfill(6)
    df['PN']      = df['PN'].str.zfill(3)
    df['HHID_PN'] = pd.to_numeric(df['HHID'] + df['PN'], errors='coerce')
    return df.set_index('HHID_PN')


def _binary(series, missing=('8', '9', '98', '99', '', ' ', '.', 'nan')):
    out = pd.to_numeric(series, errors='coerce')
    return np.where(out == 1, 1.0, np.where(out == 5, 0.0, np.nan))


# ---------------------------------------------------------------------------
# STEP 1 -- 2016 DEMOGRAPHICS  (Section A: age; PR: sex)
# ---------------------------------------------------------------------------
print("\nStep 1: Loading 2016 demographics ...")

df_a16 = _load_da(H16_DA / 'H16A_R.da', H16_CB / 'H16A_R.txt', ['PA019'])
df_a16['age_2016'] = pd.to_numeric(df_a16['PA019'], errors='coerce')
df_a16 = df_a16[df_a16['age_2016'].between(50, 120)]   # HRS eligibility floor; no ceiling
print(f"  2016 Section A respondents age 50+: n={len(df_a16):,}")
print(f"  Age range: {int(df_a16['age_2016'].min())} - {int(df_a16['age_2016'].max())}  "
      f"mean={df_a16['age_2016'].mean():.1f}")

# Sex from PR_R
df_pr16 = _load_da(H16_DA / 'H16PR_R.da', H16_CB / 'H16PR_R.txt', ['PX060_R'])
sex_raw = pd.to_numeric(df_pr16['PX060_R'], errors='coerce')
df_pr16['sex_female'] = np.where(sex_raw == 2, 1.0, np.where(sex_raw == 1, 0.0, np.nan))
print(f"  Sex available: {df_pr16['sex_female'].notna().sum():,}  "
      f"female={df_pr16['sex_female'].mean()*100:.1f}%")

# Race from Section B
df_b16 = _load_da(H16_DA / 'H16B_R.da', H16_CB / 'H16B_R.txt', ['PB089M1M'])
race_raw = pd.to_numeric(df_b16['PB089M1M'], errors='coerce')
df_b16['race_white'] = np.where(race_raw == 1, 1.0,
                        np.where(race_raw.isin([2, 97, 98, 99]), 0.0, np.nan))
print(f"  Race available: {df_b16['race_white'].notna().sum():,}  "
      f"white={df_b16['race_white'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# STEP 2 -- 2016 DEPRESSION SCORE  (CES-D-8, Section D)
# ---------------------------------------------------------------------------
print("\nStep 2: Computing CES-D-8 depression score from 2016 Section D ...")

CESD_DIRECT   = ['PD110', 'PD111', 'PD112', 'PD114', 'PD116', 'PD117']  # yes=burden
CESD_REVERSED = ['PD113', 'PD115']                                         # yes=healthy

df_d16 = _load_da(H16_DA / 'H16D_R.da', H16_CB / 'H16D_R.txt',
                  CESD_DIRECT + CESD_REVERSED)

def _cesd_score(row, min_items=6):
    num = 0.0; denom = 0.0; answered = 0
    for var in CESD_DIRECT:
        if var not in row.index: continue
        try: code = int(float(row[var]))
        except: continue
        if code == 1:   val = 1.0
        elif code == 5: val = 0.0
        else: continue
        num += val * 10.0; denom += 10.0; answered += 1
    for var in CESD_REVERSED:
        if var not in row.index: continue
        try: code = int(float(row[var]))
        except: continue
        if code == 1:   val = 1.0   # happy → low burden
        elif code == 5: val = 0.0   # not happy → high burden
        else: continue
        # reversed: (val - 1)/(0 - 1)*10 = (1 - val)*10
        norm = (1.0 - val) * 10.0
        num += norm; denom += 10.0; answered += 1
    if answered < min_items or denom == 0:
        return np.nan
    return round((num / denom) * 100.0, 3)

dep_scores = df_d16.apply(_cesd_score, axis=1)
df_d16['dep_2016'] = dep_scores
valid_dep = df_d16['dep_2016'].notna().sum()
print(f"  Valid CES-D scores: {valid_dep:,} / {len(df_d16):,}  "
      f"mean={df_d16['dep_2016'].mean():.1f}")

# ---------------------------------------------------------------------------
# STEP 3 -- 2016 SLEEP SCORE  (Section C, PC083-PC086)
# ---------------------------------------------------------------------------
print("\nStep 3: Computing sleep quality score from 2016 Section C ...")

SLEEP_VARS_C16 = ['PC083', 'PC084', 'PC085', 'PC086',
                  # CHARGE-AF covariates (load in same pass)
                  'PC087', 'PC018', 'PC139', 'PC141', 'PC142',
                  # AFib 2016
                  'PC270M1', 'PC270M2',
                  # Sleep disorder diagnosis (introduced 2016)
                  'PC291', 'PC292']

df_c16 = _load_da(H16_DA / 'H16C_R.da', H16_CB / 'H16C_R.txt', SLEEP_VARS_C16)

SLEEP_SYMPTOM = {'PC083': (3, 1), 'PC084': (3, 1), 'PC085': (3, 1)}  # min=3, max=1
SLEEP_POSITIVE = {'PC086': (1, 3)}                                      # min=1, max=3

def _sleep_score(row, quality_threshold=0.5):
    num = 0.0; denom = 0.0
    max_possible = (len(SLEEP_SYMPTOM) + len(SLEEP_POSITIVE)) * 10.0
    for var, (minv, maxv) in {**SLEEP_SYMPTOM, **SLEEP_POSITIVE}.items():
        if var not in row.index: continue
        try: val = float(row[var])
        except: continue
        if val not in (1.0, 2.0, 3.0): continue
        norm = (val - minv) / (maxv - minv) * 10.0
        num += norm; denom += 10.0
    sq_power = denom / max_possible if max_possible > 0 else 0
    if sq_power < quality_threshold or denom == 0:
        return np.nan
    return round((num / denom) * 100.0, 3)

df_c16['sleep_2016'] = df_c16.apply(_sleep_score, axis=1)
valid_sleep = df_c16['sleep_2016'].notna().sum()
print(f"  Valid sleep scores: {valid_sleep:,} / {len(df_c16):,}  "
      f"mean={df_c16['sleep_2016'].mean():.1f}")

# Sleep apnea: PC291 = told have sleep disorder; PC292 = 1 → sleep apnea specifically
pc291_raw = pd.to_numeric(df_c16.get('PC291', pd.Series(dtype=str)), errors='coerce')
pc292_raw = pd.to_numeric(df_c16.get('PC292', pd.Series(dtype=str)), errors='coerce')
df_c16['sleep_apnea'] = np.where(
    pc292_raw == 1, 1.0,
    np.where(pc291_raw.notna(), 0.0, np.nan)
)
n_apnea = int((df_c16['sleep_apnea'] == 1).sum())
print(f"  Sleep apnea (PC292=1): n={n_apnea:,}  "
      f"prev={n_apnea/df_c16['sleep_apnea'].notna().sum()*100:.1f}%")

# CHARGE-AF covariates
df_c16['hypertension'] = pd.Series(_binary(df_c16.get('PC087', pd.Series(dtype=str))),
                                    index=df_c16.index)
df_c16['diabetes']     = pd.Series(_binary(df_c16.get('PC018', pd.Series(dtype=str))),
                                    index=df_c16.index)

wt_raw = pd.to_numeric(df_c16.get('PC139', pd.Series(dtype=str)), errors='coerce')
ht_ft  = pd.to_numeric(df_c16.get('PC141', pd.Series(dtype=str)), errors='coerce')
ht_in  = pd.to_numeric(df_c16.get('PC142', pd.Series(dtype=str)), errors='coerce')
df_c16['weight_kg'] = np.where(wt_raw.between(50, 600), wt_raw * 0.453592, np.nan)
ht_total_in = ht_ft * 12 + ht_in
df_c16['height_cm'] = np.where(ht_total_in.between(48, 90), ht_total_in * 2.54, np.nan)
ht_m = df_c16['height_cm'] / 100.0
bmi_raw = df_c16['weight_kg'] / (ht_m ** 2)
df_c16['bmi_2016'] = np.where(bmi_raw.between(12, 75), bmi_raw, np.nan)

# AFib at 2016 (prevalent)
afib16_m1 = pd.to_numeric(df_c16.get('PC270M1', pd.Series(dtype=str)), errors='coerce')
afib16_m2 = pd.to_numeric(df_c16.get('PC270M2', pd.Series(dtype=str)), errors='coerce')
df_c16['afib_2016'] = ((afib16_m1 == 1) | (afib16_m2 == 1)).astype(int)
print(f"  AFib at 2016 (prevalent): n={df_c16['afib_2016'].sum():,}  "
      f"prev={df_c16['afib_2016'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# STEP 4 -- AFib FOLLOW-UP (2018, 2020, 2022)
# ---------------------------------------------------------------------------
print("\nStep 4: Loading AFib status at 2018, 2020, 2022 ...")

# 2018
df_c18 = _load_da(H18_DA_C, H18_CB_C, ['QC270M1', 'QC270M2'])
a18_m1 = pd.to_numeric(df_c18.get('QC270M1', pd.Series(dtype=str)), errors='coerce')
a18_m2 = pd.to_numeric(df_c18.get('QC270M2', pd.Series(dtype=str)), errors='coerce')
df_c18['afib_2018'] = ((a18_m1 == 1) | (a18_m2 == 1)).astype(int)
print(f"  2018 respondents: {len(df_c18):,}  AFib 2018: {df_c18['afib_2018'].sum():,}")

# 2020
df_c20 = _load_da(H20_DA_C, H20_CB_C, ['RC270M1', 'RC270M2'])
a20_m1 = pd.to_numeric(df_c20.get('RC270M1', pd.Series(dtype=str)), errors='coerce')
a20_m2 = pd.to_numeric(df_c20.get('RC270M2', pd.Series(dtype=str)), errors='coerce')
df_c20['afib_2020'] = ((a20_m1 == 1) | (a20_m2 == 1)).astype(int)
print(f"  2020 respondents: {len(df_c20):,}  AFib 2020: {df_c20['afib_2020'].sum():,}")

# 2022
df22 = pd.read_csv(CSV_C22, dtype=str, encoding='latin-1')
df22['HHID'] = df22['HHID'].str.strip().str.zfill(6)
df22['PN']   = df22['PN'].str.strip().str.zfill(3)
df22['HHID_PN'] = pd.to_numeric(df22['HHID'] + df22['PN'], errors='coerce')
df22 = df22.set_index('HHID_PN')
a22_m1 = pd.to_numeric(df22.get('SC270M1', pd.Series(dtype=str)), errors='coerce')
a22_m2 = pd.to_numeric(df22.get('SC270M2', pd.Series(dtype=str)), errors='coerce')
df22['afib_2022'] = ((a22_m1 == 1) | (a22_m2 == 1)).astype(int)
print(f"  2022 respondents: {len(df22):,}  AFib 2022: {df22['afib_2022'].sum():,}")

# ---------------------------------------------------------------------------
# STEP 5 -- ASSEMBLE WIDE DATASET
# ---------------------------------------------------------------------------
print("\nStep 5: Assembling analytic wide dataset ...")

df = df_a16[['age_2016']].copy()

# Merge all sections (left join: 2016 baseline is master)
df = df.join(df_pr16[['sex_female']], how='left')
df = df.join(df_b16[['race_white']],  how='left')
df = df.join(df_d16[['dep_2016']],    how='left')
df = df.join(df_c16[['sleep_2016', 'hypertension', 'diabetes',
                      'weight_kg', 'height_cm', 'bmi_2016', 'afib_2016',
                      'sleep_apnea']], how='left')

# AFib follow-up
df = df.join(df_c18[['afib_2018']], how='left')
df = df.join(df_c20[['afib_2020']], how='left')
df = df.join(df22[['afib_2022']],   how='left')

# Fill missing follow-up AFib as unknown (NaN), not zero
# A missing value means the participant was not seen at that wave

# Determine onset wave (incident AFib logic)
def get_onset(row):
    if row.get('afib_2016', 0) == 1:
        return 2016   # prevalent
    for yr, col in [(2018, 'afib_2018'), (2020, 'afib_2020'), (2022, 'afib_2022')]:
        val = row.get(col, np.nan)
        if val == 1:
            return yr
    return np.nan   # no AFib observed

df['afib_onset_wave']     = df.apply(get_onset, axis=1)
df['afib_prevalent_2016'] = (df['afib_onset_wave'] == 2016).astype(int)
df['afib_incident']       = (
    df['afib_onset_wave'].notna() & (df['afib_onset_wave'] > 2016)
).astype(int)

# Event time: years from 2016 baseline
# Participants seen at a wave without AFib are censored at their last seen wave
# For simplicity: censor at last wave they participated (2018, 2020, or 2022)
def get_tstop(row):
    if pd.notna(row['afib_onset_wave']) and row['afib_onset_wave'] > 2016:
        return float(row['afib_onset_wave'] - 2016)
    # Censor at latest wave observed
    for yr, col in [(2022, 'afib_2022'), (2020, 'afib_2020'), (2018, 'afib_2018')]:
        if pd.notna(row.get(col, np.nan)):
            return float(yr - 2016)
    return 6.0   # default censor 2022

df['event_time_yrs'] = df.apply(get_tstop, axis=1)

print(f"  Full expanded cohort: n={len(df):,}")
print(f"  Prevalent AFib 2016: {df['afib_prevalent_2016'].sum():,}")
print(f"  Incident AFib (post-2016): {df['afib_incident'].sum():,}")
print(f"  Non-AFib (censored): {((df['afib_incident']==0) & (df['afib_prevalent_2016']==0)).sum():,}")
print(f"\n  Key variable completeness:")
for col in ['dep_2016', 'sleep_2016', 'age_2016', 'height_cm', 'weight_kg',
            'hypertension', 'diabetes', 'sex_female', 'race_white']:
    n = df[col].notna().sum()
    print(f"    {col:20s}: {n:,} / {len(df):,}  ({n/len(df)*100:.1f}%)")

print(f"\n  Age distribution of full cohort:")
for lo, hi in [(50,60),(61,70),(71,80),(81,120)]:
    n = df['age_2016'].between(lo,hi).sum()
    n_afib = (df['age_2016'].between(lo,hi) & (df['afib_incident']==1)).sum()
    pct = n/len(df)*100
    print(f"    {lo}-{hi}: n={n:,} ({pct:.1f}%)  incident AFib={n_afib:,}")

# ---------------------------------------------------------------------------
# STEP 6 -- COUNTING PROCESS DATASET
# ---------------------------------------------------------------------------
print("\nStep 6: Building counting-process Cox dataset ...")

# Exclude prevalent AFib 2016
df_cox = df[df['afib_prevalent_2016'] == 0].copy()
df_cox['tstart'] = 0.0
df_cox['tstop']  = df_cox['event_time_yrs'].clip(lower=0.01)
df_cox['event']  = df_cox['afib_incident']

# Derived: dep_change not available without 2022 dep (not collected here)
# age_group for stratification
df_cox['age_group'] = pd.cut(df_cox['age_2016'],
                              bins=[49, 60, 70, 80, 120],
                              labels=['50-60', '61-70', '71-80', '81+'])
df_cox['age_z'] = (df_cox['age_2016'] - df_cox['age_2016'].mean()) / df_cox['age_2016'].std()

# z-score dep and sleep for HRs
for col in ['dep_2016', 'sleep_2016']:
    mu, sd = df_cox[col].mean(), df_cox[col].std()
    if sd > 0:
        df_cox[col + '_z'] = (df_cox[col] - mu) / sd

cox_vars = ['tstart', 'tstop', 'event', 'age_2016', 'age_z', 'age_group',
            'dep_2016', 'dep_2016_z', 'sleep_2016', 'sleep_2016_z',
            'sex_female', 'race_white', 'height_cm', 'weight_kg', 'bmi_2016',
            'hypertension', 'diabetes', 'sleep_apnea']
df_cox_out = df_cox[cox_vars].reset_index()

print(f"  Expanded Cox dataset: {len(df_cox_out):,} rows")
print(f"  Incident AFib events: {int(df_cox_out['event'].sum())}")
print(f"  Censored: {int((df_cox_out['event']==0).sum())}")

# ---------------------------------------------------------------------------
# STEP 7 -- SAVE
# ---------------------------------------------------------------------------
print("\nStep 7: Saving outputs ...")

wide_path = OUT_DIR / "hrs_analytic_wide_expanded.csv"
cox_path  = OUT_DIR / "hrs_cox_long_expanded.csv"

df.reset_index().to_csv(wide_path, index=False)
df_cox_out.to_csv(cox_path, index=False)

# Also write to canonical results/data/ location used by figure scripts
SHARED_DIR = REPO_ROOT / "SF_OUTPUT" / "analytic"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
df_cox_out.to_csv(SHARED_DIR / "hrs_cox_long_expanded.csv", index=False)

print(f"  -> {wide_path.name}  ({len(df):,} rows)")
print(f"  -> {cox_path.name}   ({len(df_cox_out):,} rows)")
print(f"  -> results/data/hrs_cox_long_expanded.csv  (shared copy)")

print()
print("=" * 70)
print("  EXPANDED DATASET ASSEMBLY COMPLETE")
print("=" * 70)
print(f"\n  Original cohort (50-60 only): ~4,400")
print(f"  Expanded cohort (50+):         {len(df):,}")
print(f"  Incident AFib events:           {int(df_cox_out['event'].sum())}")
