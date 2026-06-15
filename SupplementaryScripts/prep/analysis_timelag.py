"""
analysis_timelag.py
===================
Landmark Analysis: Depression and Sleep Quality HR by Time Lag Before AFib

Scores depression (CESD-8) and sleep quality (4-item) at 2018 and 2020 from
the newly discovered full-wave .da files, then runs three landmark Cox models:

  2016 landmark → outcome AFib 2018/2020/2022  (up to 6-yr follow-up)
  2018 landmark → outcome AFib 2020/2022       (up to 4-yr follow-up)
  2020 landmark → outcome AFib 2022            (2-yr follow-up)

The HR at each landmark reflects the association when the measurement is
taken at different distances from the eventual AF event. If the HR is larger
at the 2020 landmark than the 2016 landmark, it suggests these factors are
stronger near-term (prodromal) markers rather than distant causal ones.

CHARGE-AF adjustment: age updated per wave (+2/+4 yrs from 2016 baseline),
height/weight/hypertension/diabetes carried forward from 2016 baseline.

Note: 2020 landmark has only ~78 events (all 2022-onset), so is underpowered
for detecting modest HRs; results are descriptive.
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[2]
DATA_OUT  = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "analytic"
FIG_OUT   = Path(__file__).resolve().parents[2] / "SF_OUTPUT" / "figures"

WIDE_CSV  = REPO_ROOT / "SF_OUTPUT" / "analytic" / "hrs_analytic_wide.csv"
OBJ3_CSV  = REPO_ROOT / "SF_OUTPUT" / "analytic" / \
            "OBJ3_afib_eventtime_clusters.csv"

HRS2018_DA_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2018" / "h18da" / "H18C_R.da"
HRS2018_CB_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2018" / "h18cb" / "H18C_R.txt"
HRS2018_DA_D = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2018" / "h18da" / "H18D_R.da"
HRS2018_CB_D = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2018" / "h18cb" / "H18D_R.txt"

HRS2020_DA_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2020" / "h20da" / "H20C_R.da"
HRS2020_CB_C = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2020" / "h20cb" / "H20C_R.txt"
HRS2020_DA_D = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2020" / "h20da" / "H20D_R.da"
HRS2020_CB_D = REPO_ROOT / "RawData" / "HRS" / "waves" / "HRS2020" / "h20cb" / "H20D_R.txt"

print()
print("=" * 70)
print("  ANALYSIS: Landmark Time-Lag  --  Dep/Sleep HR vs Wave")
print("=" * 70)

# ---------------------------------------------------------------------------
# HRS .DA FILE HELPERS
# ---------------------------------------------------------------------------

def _parse_colspecs(cb_path):
    """Parse fixed-width column positions from HRS ASCII codebook."""
    specs = {}
    pos = 0
    varname_pat = re.compile(r'^([A-Z][A-Z0-9_]+)\s+\S')
    meta_pat    = re.compile(r'Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)', re.I)
    cur = None
    with open(cb_path, encoding='latin-1') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if (line and not line.startswith(' ') and not line.startswith('=')
                    and not line.startswith('-') and not line.startswith('{')):
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
    """Load selected variables from HRS fixed-width .da file."""
    all_specs = _parse_colspecs(cb_path)
    need      = ['HHID', 'PN'] + [v.upper() for v in variables if v.upper() in all_specs]
    missing_v = [v for v in variables if v.upper() not in all_specs]
    if missing_v:
        print(f"    WARNING: not in codebook: {missing_v}")
    colspecs = [all_specs[v] for v in need]
    df = pd.read_fwf(da_path, colspecs=colspecs, header=None,
                     names=need, dtype=str, encoding='latin-1')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['HHID']    = df['HHID'].str.zfill(6)
    df['PN']      = df['PN'].str.zfill(3)
    df['HHID_PN'] = pd.to_numeric(df['HHID'] + df['PN'], errors='coerce')
    return df.set_index('HHID_PN')


# ---------------------------------------------------------------------------
# SCORING FUNCTIONS
# ---------------------------------------------------------------------------
MISSING_CODES = {'8', '9', '98', '99', '998', '999', '', ' ', '.', 'nan'}


def _to_numeric_clean(series):
    """Convert HRS string codes to float; set missing codes to NaN."""
    out = series.copy()
    out[out.isin(MISSING_CODES)] = np.nan
    return pd.to_numeric(out, errors='coerce')


def score_sleep(df_raw, prefix):
    """
    Score 4-item HRS sleep quality from Section C.
    prefix: 'PC' (2016), 'QC' (2018), 'RC' (2020)
    3-point ordinal: 1=Most of time  2=Sometimes  3=Rarely/never
    Returns Series indexed by HHID_PN, 0=best sleep, 100=worst sleep.
    """
    v83 = prefix + '083'
    v84 = prefix + '084'
    v85 = prefix + '085'
    v86 = prefix + '086'

    symptom_vars = [v83, v84, v85]   # higher raw value = less trouble = better
    rested_var   = v86               # higher raw value = less rested = worse

    results = {}
    for idx, row in df_raw.iterrows():
        items, total_weight = [], 0.0
        valid = 0

        for var in symptom_vars:
            val = _to_numeric_clean(pd.Series([row.get(var, np.nan)])).iloc[0]
            if pd.notna(val) and val in (1.0, 2.0, 3.0):
                # min_effect=3, max_effect=1 → Most(1)=10, Rarely(3)=0
                norm = (val - 3.0) / (1.0 - 3.0) * 10.0
                items.append(norm)
                total_weight += 10.0
                valid += 1

        val = _to_numeric_clean(pd.Series([row.get(rested_var, np.nan)])).iloc[0]
        if pd.notna(val) and val in (1.0, 2.0, 3.0):
            # min_effect=1, max_effect=3 → Most rested(1)=0, Rarely rested(3)=10
            norm = (val - 1.0) / (3.0 - 1.0) * 10.0
            items.append(norm)
            total_weight += 10.0
            valid += 1

        if valid >= 2 and total_weight > 0:
            results[idx] = round(sum(items) / total_weight * 100.0, 3)

    return pd.Series(results, name=f'sleep_{prefix[:2].lower()}')


def score_depression(df_raw, prefix):
    """
    Score CESD-8 from Section D.
    prefix: 'PD' (2016), 'QD' (2018), 'RD' (2020)
    Binary: 1=YES, 5=NO
    Returns Series indexed by HHID_PN, 0=no symptoms, 100=max burden.
    """
    # Direct items (YES = burden): 110, 111, 112, 114, 116, 117
    direct  = [prefix + s for s in ('110', '111', '112', '114', '116', '117')]
    # Reverse items (YES = healthy): 113 (happy), 115 (enjoyed life)
    reverse = [prefix + s for s in ('113', '115')]
    all_items = direct + reverse

    results = {}
    for idx, row in df_raw.iterrows():
        norms, answered = [], 0

        for var in direct:
            raw = str(row.get(var, '')).strip().lstrip('0') or '0'
            if raw in MISSING_CODES:
                continue
            val = 1.0 if raw == '1' else (0.0 if raw == '5' else None)
            if val is None:
                continue
            norms.append(val * 10.0)
            answered += 1

        for var in reverse:
            raw = str(row.get(var, '')).strip().lstrip('0') or '0'
            if raw in MISSING_CODES:
                continue
            val = 1.0 if raw == '1' else (0.0 if raw == '5' else None)
            if val is None:
                continue
            # YES (healthy) → 0 burden; NO (unhealthy) → 10 burden
            norms.append((1.0 - val) * 10.0)
            answered += 1

        if answered >= 6:
            results[idx] = round(sum(norms) / (len(all_items) * 10.0) * 100.0, 3)

    return pd.Series(results, name=f'dep_{prefix[:2].lower()}')


# ---------------------------------------------------------------------------
# SCORE 2018
# ---------------------------------------------------------------------------
print("\nScoring 2018 sleep (QC083-086) ...")
df18c = _load_da(HRS2018_DA_C, HRS2018_CB_C,
                 ['QC083', 'QC084', 'QC085', 'QC086'])
sleep_2018 = score_sleep(df18c, 'QC')
print(f"  Valid sleep scores 2018: {sleep_2018.notna().sum():,}  "
      f"mean={sleep_2018.mean():.1f}")

print("Scoring 2018 depression (QD110-117) ...")
df18d = _load_da(HRS2018_DA_D, HRS2018_CB_D,
                 ['QD110', 'QD111', 'QD112', 'QD113', 'QD114', 'QD115', 'QD116', 'QD117'])
dep_2018 = score_depression(df18d, 'QD')
print(f"  Valid dep scores 2018:   {dep_2018.notna().sum():,}  "
      f"mean={dep_2018.mean():.1f}")

# ---------------------------------------------------------------------------
# SCORE 2020
# ---------------------------------------------------------------------------
print("\nScoring 2020 sleep (RC083-086) ...")
df20c = _load_da(HRS2020_DA_C, HRS2020_CB_C,
                 ['RC083', 'RC084', 'RC085', 'RC086'])
sleep_2020 = score_sleep(df20c, 'RC')
print(f"  Valid sleep scores 2020: {sleep_2020.notna().sum():,}  "
      f"mean={sleep_2020.mean():.1f}")

print("Scoring 2020 depression (RD110-117) ...")
df20d = _load_da(HRS2020_DA_D, HRS2020_CB_D,
                 ['RD110', 'RD111', 'RD112', 'RD113', 'RD114', 'RD115', 'RD116', 'RD117'])
dep_2020 = score_depression(df20d, 'RD')
print(f"  Valid dep scores 2020:   {dep_2020.notna().sum():,}  "
      f"mean={dep_2020.mean():.1f}")

# ---------------------------------------------------------------------------
# LOAD BASE DATASET
# ---------------------------------------------------------------------------
print("\nLoading analytic wide dataset ...")
df_wide = pd.read_csv(WIDE_CSV).set_index('HHID_PN')
print(f"  n={len(df_wide):,}  incident AFib={df_wide['afib_incident'].sum()}")
print(f"  afib_onset_wave distribution:")
print(df_wide['afib_onset_wave'].value_counts(dropna=False).sort_index().to_string())

# Attach 2018 and 2020 scores
df_wide = df_wide.join(sleep_2018.rename('sleep_2018'), how='left')
df_wide = df_wide.join(dep_2018.rename('dep_2018'),   how='left')
df_wide = df_wide.join(sleep_2020.rename('sleep_2020'), how='left')
df_wide = df_wide.join(dep_2020.rename('dep_2020'),   how='left')

print(f"\n  2018 sleep matched to cohort: {df_wide['sleep_2018'].notna().sum():,}")
print(f"  2018 dep   matched to cohort: {df_wide['dep_2018'].notna().sum():,}")
print(f"  2020 sleep matched to cohort: {df_wide['sleep_2020'].notna().sum():,}")
print(f"  2020 dep   matched to cohort: {df_wide['dep_2020'].notna().sum():,}")

# Save intermediate dataset
df_wide.reset_index().to_csv(DATA_OUT / "wide_with_intermediate_waves.csv", index=False)

# ---------------------------------------------------------------------------
# BUILD LANDMARK DATASETS
# ---------------------------------------------------------------------------
CHARGE_BASE = ['height_cm', 'weight_kg', 'hypertension', 'diabetes']


def build_landmark(df, landmark_year, dep_col, sleep_col):
    """
    Build a Cox-ready dataset for a given landmark year.
    Includes only participants who:
      - Were AFib-free before landmark_year
      - Have valid dep and sleep scores at landmark_year
    Outcome: AFib onset strictly after landmark_year, censored at 2022.
    Time axis: years from landmark_year.
    """
    # Exclude prevalent AFib at or before the landmark wave
    eligible = df[
        df['afib_prevalent_2016'] == 0
    ].copy()

    if landmark_year >= 2018:
        eligible = eligible[
            ~(eligible['afib_onset_wave'] < landmark_year)
        ]
    if landmark_year >= 2020:
        eligible = eligible[
            ~(eligible['afib_onset_wave'] < landmark_year)
        ]

    # Also exclude people who had AFib AT the landmark wave
    # (their exposure was measured simultaneously with the event)
    eligible = eligible[
        ~(eligible['afib_onset_wave'] == landmark_year)
    ]

    # Age at landmark
    eligible['age_lm'] = eligible['age_2016'] + (landmark_year - 2016)

    # Time and event
    eligible['tstop'] = (
        eligible['afib_onset_wave']
        .where(eligible['afib_onset_wave'] > landmark_year, 2022.0)
        - landmark_year
    ).clip(lower=0.01)
    eligible['event'] = (
        eligible['afib_incident'] == 1
    ) & (eligible['afib_onset_wave'] > landmark_year)
    eligible['event'] = eligible['event'].astype(int)

    # Keep needed columns and drop rows missing scores or covariates
    keep = [dep_col, sleep_col, 'age_lm'] + CHARGE_BASE + ['tstop', 'event']
    sub  = eligible[keep].dropna()

    # Z-score predictors within this landmark sample
    for col in [dep_col, sleep_col, 'age_lm'] + CHARGE_BASE:
        mu, sd = sub[col].mean(), sub[col].std()
        if sd > 0:
            sub[col + '_z'] = (sub[col] - mu) / sd

    return sub


# ---------------------------------------------------------------------------
# RUN COX MODELS AT EACH LANDMARK
# ---------------------------------------------------------------------------
LANDMARKS = [
    (2016, 'dep_2016',  'sleep_2016',  'dep_2016_z',  'sleep_2016_z'),
    (2018, 'dep_2018',  'sleep_2018',  'dep_2018_z',  'sleep_2018_z'),
    (2020, 'dep_2020',  'sleep_2020',  'dep_2020_z',  'sleep_2020_z'),
]

CHARGE_Z = ['age_lm_z', 'height_cm_z', 'weight_kg_z', 'hypertension_z', 'diabetes_z']

results_rows = []

print()
print("=" * 70)
print("  LANDMARK COX MODELS")
print("=" * 70)

for lm_year, dep_raw, sleep_raw, dep_z, sleep_z in LANDMARKS:
    print(f"\n--- Landmark {lm_year} ---")
    sub = build_landmark(df_wide, lm_year, dep_raw, sleep_raw)
    n, ev = len(sub), int(sub['event'].sum())
    print(f"  n={n:,}  events={ev}")

    if ev < 5:
        print("  SKIP: too few events")
        continue

    for pred_z, pred_label in [(dep_z, 'Depression'), (sleep_z, 'Sleep quality')]:
        covs = [pred_z] + CHARGE_Z
        cc   = sub[covs + ['tstop', 'event']].dropna()
        if cc['event'].sum() < 5:
            print(f"  SKIP {pred_label}: too few complete cases")
            continue
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(cc, duration_col='tstop', event_col='event',
                    formula=' + '.join(covs))
            row_s = cph.summary.loc[pred_z]
            hr    = np.exp(row_s['coef'])
            ci_lo = np.exp(row_s['coef lower 95%'])
            ci_hi = np.exp(row_s['coef upper 95%'])
            p     = row_s['p']
            print(f"  {pred_label:15s}: HR={hr:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]  p={p:.3f}  "
                  f"n={len(cc)}, ev={int(cc['event'].sum())}")
            results_rows.append({
                'landmark_year': lm_year,
                'predictor':     pred_label,
                'HR':            round(hr, 4),
                'CI_lo':         round(ci_lo, 4),
                'CI_hi':         round(ci_hi, 4),
                'p':             round(p, 4),
                'n':             len(cc),
                'events':        int(cc['event'].sum()),
            })
        except Exception as e:
            print(f"  ERROR {pred_label}: {e}")

df_res = pd.DataFrame(results_rows)
df_res.to_csv(DATA_OUT / "timelag_cox_results.csv", index=False)
print(f"\n  -> Results saved ({len(df_res)} rows)")

# ---------------------------------------------------------------------------
# FIGURE 1: Forest plot — HR at each landmark, side by side
# ---------------------------------------------------------------------------
print("\nGenerating figures ...")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')

C_DEP   = '#C8102E'
C_SLEEP = '#2C5F8A'
LANDMARK_LABELS = {2016: '2016\n(≤6 yr follow-up)', 2018: '2018\n(≤4 yr follow-up)',
                   2020: '2020\n(2 yr follow-up)'}

y_ticks, y_labels = [], []
y = 0
for lm_year in [2016, 2018, 2020]:
    sub_res = df_res[df_res['landmark_year'] == lm_year]
    for _, row in sub_res.iterrows():
        col   = C_DEP if row['predictor'] == 'Depression' else C_SLEEP
        label = f"{LANDMARK_LABELS[lm_year]}  {row['predictor']}"
        sig   = '★' if row['p'] < 0.05 else ''
        ax.plot([row['CI_lo'], row['CI_hi']], [y, y], color=col, lw=2.5, zorder=2)
        ax.scatter([row['HR']], [y], color=col, s=90, zorder=3, marker='D')
        ax.text(row['CI_hi'] + 0.02, y,
                f"HR={row['HR']:.3f} [{row['CI_lo']:.3f}–{row['CI_hi']:.3f}]  "
                f"p={row['p']:.3f}{sig}  (ev={row['events']})",
                va='center', ha='left', fontsize=8.5, color=col)
        y_ticks.append(y)
        y_labels.append(f"  {row['predictor']}")
        y += 1
    if lm_year != 2020:
        ax.axhline(y - 0.5, color='#cccccc', lw=0.8, linestyle=':')
        ax.text(-0.15, y + 0.2, f'Landmark {lm_year}', fontsize=8,
                color='gray', ha='left', transform=ax.get_yaxis_transform())
        y += 0.8

ax.axvline(1.0, color='black', lw=1.2, linestyle='--', zorder=1)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Hazard Ratio for AFib per SD  (adjusted for CHARGE-AF covariates)', fontsize=10)
ax.set_title('Landmark Analysis: Depression and Sleep Quality HR by Measurement Wave\n'
             'Does proximity to AF onset strengthen the association?',
             fontsize=11)
ax.set_xlim(0.4, max(df_res['CI_hi'].max() + 1.0, 2.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_DEP, label='Depression'),
                   Patch(color=C_SLEEP, label='Sleep quality')],
          fontsize=9, loc='lower right')

plt.tight_layout()
fig.savefig(FIG_OUT / "timelag_forest.png", dpi=150, bbox_inches='tight')
plt.close()
print("  -> timelag_forest.png")

# ---------------------------------------------------------------------------
# FIGURE 2: HR trend lines across landmarks
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.patch.set_facecolor('white')

x_labels = ['2016\n(6 yr)', '2018\n(4 yr)', '2020\n(2 yr)']
x_pos    = [0, 1, 2]

for ax, pred, col, title in zip(
        axes,
        ['Depression', 'Sleep quality'],
        [C_DEP, C_SLEEP],
        ['Depression Score (CESD-8)', 'Sleep Quality Score (4-item)']):

    sub = df_res[df_res['predictor'] == pred].sort_values('landmark_year')
    if sub.empty:
        ax.set_title(title)
        continue

    xs = [x_pos[i] for i, yr in enumerate([2016, 2018, 2020]) if yr in sub['landmark_year'].values]
    hrs    = sub['HR'].tolist()
    ci_lo  = sub['CI_lo'].tolist()
    ci_hi  = sub['CI_hi'].tolist()
    events = sub['events'].tolist()

    ax.fill_between(xs, ci_lo, ci_hi, color=col, alpha=0.15)
    ax.plot(xs, hrs, color=col, lw=2.5, marker='D', markersize=8, zorder=3)
    ax.axhline(1.0, color='black', lw=1.0, linestyle='--')

    for xi, hr, lo, hi, ev, row in zip(xs, hrs, ci_lo, ci_hi, events, sub.itertuples()):
        sig = '★' if row.p < 0.05 else ''
        ax.text(xi, hi + 0.04, f'{hr:.2f}{sig}\n(ev={ev})',
                ha='center', va='bottom', fontsize=8, color=col)

    ax.set_xticks(x_pos[:len(xs)])
    ax.set_xticklabels([x_labels[i] for i in range(len(xs))], fontsize=9)
    ax.set_xlabel('Measurement wave  (follow-up window)', fontsize=9)
    ax.set_ylabel('Hazard Ratio for AFib per SD', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('HR Trend Across Landmark Waves: Does AF Risk Signal Sharpen Near the Event?',
             fontsize=11)
plt.tight_layout()
fig.savefig(FIG_OUT / "timelag_trend.png", dpi=150, bbox_inches='tight')
plt.close()
print("  -> timelag_trend.png")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(df_res.to_string(index=False))
print()
print("NOTE: 2020 landmark has only 2-yr follow-up and ~78 events max.")
print("Power to detect HR<1.2 is limited; treat 2020 results as descriptive.")
