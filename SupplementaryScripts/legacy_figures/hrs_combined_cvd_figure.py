"""
hrs_combined_cvd_figure.py
==========================
Single combined figure:
  Left col  (2 rows): CVD incidence bar charts — depression (top) + sleep (bottom)
  Right 2×3          : CVD type breakdown — depression row / sleep row × Improved|Stable|Worsened

Layout mirrors the screenshot layout exactly.

Outputs:
  HRS/results/longitudinal/combined_cvd_figure_2010_2022.png
  results/CherryPicked2/HRS_combined_cvd_figure.png
"""

import re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
import engine.viz_style as V
from engine.paths import HRS_RAND_DTA, HRS_WAVES, HRS_HMS2016, HRS_HMS2022
V.apply()

OUT_LONG   = _REPO / "HRS" / "results" / "longitudinal"
OUT_CHERRY = _REPO / "results" / "CherryPicked2"
OUT_LONG.mkdir(parents=True, exist_ok=True)
OUT_CHERRY.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
RAND_DTA     = HRS_RAND_DTA
WAVES        = {10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022}
SLOPE_THRESH = 1.5
CESD_SCALE   = 12.5
TRAJ_ORDER   = ["Improved", "Stable", "Worsened"]
FIG_DPI      = 150

CVD_SUBTYPES    = ["Stroke / TIA", "Heart Failure", "Heart Attack", "Arrhythmia"]
ARRHYTHMIA_CODE = 1
BAR_COLORS = {
    "Stroke / TIA":  "#AAAAAA",
    "Heart Failure": "#AAAAAA",
    "Heart Attack":  "#AAAAAA",
    "Arrhythmia":    "#111111",
    "Multiple Events": "#555555",
}
X_LABELS       = ["Stroke\n/TIA", "Heart\nFailure", "Heart\nAttack", "Arrhythmia", "Multiple\nEvents"]

WAVE_C_FWF = {
    2010: (HRS_WAVES / "HRS2010" / "H10C_R.da",               HRS_WAVES / "HRS2010" / "H10C_R.txt",               "M"),
    2012: (HRS_WAVES / "HRS2012" / "H12C_R.da",               HRS_WAVES / "HRS2012" / "H12C_R.txt",               "N"),
    2014: (HRS_WAVES / "HRS2014" / "H14C_R.da",               HRS_WAVES / "HRS2014" / "H14C_R.txt",               "O"),
    2016: (HRS_HMS2016 / "h16da" / "H16C_R.da",               HRS_HMS2016 / "h16cb" / "H16C_R.txt",               "P"),
    2018: (HRS_WAVES / "HRS2018" / "h18da" / "H18C_R.da",     HRS_WAVES / "HRS2018" / "h18cb" / "H18C_R.txt",     "Q"),
    2020: (HRS_WAVES / "HRS2020" / "h20da" / "H20C_R.da",     HRS_WAVES / "HRS2020" / "h20cb" / "H20C_R.txt",     "R"),
}
WAVE_C_CSV = {2022: (HRS_HMS2022 / "H22csv" / "h22c_r.csv", "S")}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_colspecs(cb_path):
    specs, pos = {}, 0
    vname_pat = re.compile(r"^([A-Z][A-Z0-9_]+)\s+\S")
    meta_pat  = re.compile(r"Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)", re.IGNORECASE)
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


def _load_c_fwf(da_path, cb_path, prefix, ids):
    specs = _parse_colspecs(cb_path)
    want  = ([f"{prefix}C083", f"{prefix}C084", f"{prefix}C085", f"{prefix}C086"]
             + [f"{prefix}C040", f"{prefix}C048", f"{prefix}C053",
                f"{prefix}C270M1", f"{prefix}C270M2"])
    need     = ["HHID", "PN"] + [v for v in want if v in specs]
    colspecs = [specs[v] for v in need]
    df = pd.read_fwf(da_path, colspecs=colspecs, header=None,
                     names=need, dtype=str, encoding="latin-1")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["HHID"]   = df["HHID"].str.zfill(6)
    df["PN"]     = df["PN"].str.zfill(3)
    df["hhidpn"] = pd.to_numeric(df["HHID"] + df["PN"], errors="coerce")
    df = df.drop(columns=["HHID", "PN"])
    for col in [v for v in want if v in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["hhidpn"].isin(ids)].copy()


def _load_c_csv(csv_path, prefix, ids):
    want = ([f"{prefix}C083", f"{prefix}C084", f"{prefix}C085", f"{prefix}C086"]
            + [f"{prefix}C040", f"{prefix}C048", f"{prefix}C053",
               f"{prefix}C270M1", f"{prefix}C270M2"])
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = df.columns.str.upper()
    if "HHID" in df.columns and "PN" in df.columns:
        df["HHID"]   = df["HHID"].str.zfill(6)
        df["PN"]     = df["PN"].str.zfill(3)
        df["hhidpn"] = pd.to_numeric(df["HHID"] + df["PN"], errors="coerce")
    elif "HHIDPN" in df.columns:
        df["hhidpn"] = pd.to_numeric(df["HHIDPN"], errors="coerce")
    avail = [v for v in want if v in df.columns]
    for col in avail:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["hhidpn"].isin(ids)][["hhidpn"] + avail].copy()


def _score_sleep(row, prefix):
    num, denom = 0.0, 0.0
    for item in [f"{prefix}C083", f"{prefix}C084", f"{prefix}C085"]:
        val = row.get(item, np.nan)
        if pd.isna(val) or val not in (1.0, 2.0, 3.0):
            continue
        num   += (val - 3.0) / (1.0 - 3.0) * 10.0
        denom += 10.0
    val = row.get(f"{prefix}C086", np.nan)
    if not pd.isna(val) and val in (1.0, 2.0, 3.0):
        num   += (val - 1.0) / (3.0 - 1.0) * 10.0
        denom += 10.0
    if denom < 20.0:
        return np.nan
    return round((num / denom) * 100.0, 3)


def _mean_pairwise_rate(row, year_cols):
    obs = [(yr, row[col]) for yr, col in year_cols if not pd.isna(row.get(col))]
    if len(obs) < 2:
        return np.nan
    rates = [(obs[i+1][1] - obs[i][1]) / ((obs[i+1][0] - obs[i][0]) / 2)
             for i in range(len(obs) - 1)]
    return float(np.mean(rates))


def _to_traj(rate):
    if pd.isna(rate):   return np.nan
    if rate > SLOPE_THRESH:  return "Worsened"
    if rate < -SLOPE_THRESH: return "Improved"
    return "Stable"


def _chi2_str(cohort):
    ct = pd.crosstab(cohort["trajectory"], cohort["any_new_cvd"])
    if ct.shape != (3, 2):
        return None, None, None
    chi2, p, dof, _ = chi2_contingency(ct.reindex(TRAJ_ORDER))
    if   p < 0.001: p_str = "p < 0.001"
    elif p < 0.01:  p_str = "p < 0.01"
    elif p < 0.05:  p_str = "p < 0.05"
    else:           p_str = f"p = {p:.3g}"
    return chi2, dof, p_str


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("Loading RAND HRS data ...")
rand_cols = ["hhidpn", "r10agey_e"]
for w in WAVES:
    rand_cols += [f"r{w}cesd", f"r{w}heart"]
df_rand = pd.read_stata(RAND_DTA, columns=rand_cols, convert_categoricals=False)
print(f"  {len(df_rand):,} total")

for w, yr in WAVES.items():
    df_rand[f"dep_{yr}"] = df_rand[f"r{w}cesd"] * CESD_SCALE

cohort_base = df_rand[
    (df_rand["r10heart"] == 0.0) & df_rand["r10agey_e"].between(50, 70)
].copy()
print(f"  CVD-free at 2010, age 50–70: {len(cohort_base):,}")

# C-section files
print("\nLoading C-section files ...")
all_ids, c_frames = set(cohort_base["hhidpn"].dropna()), {}
for yr, (da, cb, pfx) in WAVE_C_FWF.items():
    if not da.exists():
        print(f"  WARNING: {da} missing — skipping {yr}")
        continue
    df_c = _load_c_fwf(da, cb, pfx, all_ids)
    df_c["wave_year"] = yr
    df_c["_pfx"]      = pfx
    c_frames[yr]      = df_c
    print(f"  {yr}: {len(df_c):,} records")
for yr, (csv_path, pfx) in WAVE_C_CSV.items():
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} missing — skipping {yr}")
        continue
    df_c = _load_c_csv(csv_path, pfx, all_ids)
    df_c["wave_year"] = yr
    df_c["_pfx"]      = pfx
    c_frames[yr]      = df_c
    print(f"  {yr}: {len(df_c):,} records (CSV)")

# Sleep scores
print("\nScoring sleep quality ...")
sleep_scores = {}
for yr, df_c in c_frames.items():
    pfx    = df_c["_pfx"].iloc[0]
    scores = df_c.apply(lambda row: _score_sleep(row, pfx), axis=1)
    s      = pd.Series(scores.values, index=df_c["hhidpn"].values, name=f"sleep_{yr}")
    valid  = s.notna().sum()
    if valid / len(df_c) < 0.10:
        print(f"  {yr}: low coverage — skipped")
        continue
    sleep_scores[yr] = s
    print(f"  {yr}: {valid:,} valid  mean={s.mean():.1f}")

# Follow-up waves for any-CVD outcome
_FOLLOW_WAVES = [w for w in WAVES if w > 10]

def _new_cvd(row):
    return int(any(row[f"r{w}heart"] == 1.0 for w in _FOLLOW_WAVES))


# ── Depression cohort ──────────────────────────────────────────────────────────
dep_cohort = cohort_base[cohort_base["dep_2010"].notna()].copy()
_DEP_COLS  = [(yr, f"dep_{yr}") for _, yr in sorted(WAVES.items())]
dep_cohort["dep_slope"]  = dep_cohort.apply(lambda r: _mean_pairwise_rate(r, _DEP_COLS), axis=1)
dep_cohort["trajectory"] = dep_cohort["dep_slope"].apply(_to_traj)
dep_cohort = dep_cohort[dep_cohort["trajectory"].isin(TRAJ_ORDER)].copy()
dep_cohort["new_cvd"] = dep_cohort.apply(_new_cvd, axis=1)
print(f"\n  Depression cohort: {len(dep_cohort):,}")

# ── Sleep cohort ───────────────────────────────────────────────────────────────
sleep_cohort = cohort_base.copy()
for yr, s in sleep_scores.items():
    sleep_cohort = sleep_cohort.merge(
        s.reset_index().rename(columns={"index": "hhidpn"}), on="hhidpn", how="left")
sleep_cohort = sleep_cohort[sleep_cohort["sleep_2010"].notna()].copy()
_SLEEP_COLS  = [(yr, f"sleep_{yr}") for yr in sorted(sleep_scores.keys())]
sleep_cohort["sleep_slope"] = sleep_cohort.apply(lambda r: _mean_pairwise_rate(r, _SLEEP_COLS), axis=1)
sleep_cohort["trajectory"]  = sleep_cohort["sleep_slope"].apply(_to_traj)
sleep_cohort = sleep_cohort[sleep_cohort["trajectory"].isin(TRAJ_ORDER)].copy()
sleep_cohort["new_cvd"] = sleep_cohort.apply(_new_cvd, axis=1)
print(f"  Sleep cohort: {len(sleep_cohort):,}")

# ── CVD subtype flags ──────────────────────────────────────────────────────────
def _build_cvd_flags(cohort):
    ids          = set(cohort["hhidpn"].dropna())
    df_follow    = pd.concat([df for yr, df in c_frames.items() if yr > 2010], ignore_index=True)
    df_follow    = df_follow[df_follow["hhidpn"].isin(ids)].copy()
    arr_cols     = [c for c in df_follow.columns if "C270M1" in c or "C270M2" in c]
    if arr_cols:
        df_follow["arrhythmia"] = (df_follow[arr_cols] == ARRHYTHMIA_CODE).any(axis=1).astype(int)
    else:
        df_follow["arrhythmia"] = 0
    def _any(stem):
        m = [c for c in df_follow.columns if stem in c]
        if not m:
            return pd.Series(0, index=sorted(ids))
        sub = df_follow[["hhidpn"] + m].copy()
        sub["_hit"] = (sub[m] == 1).any(axis=1)
        return sub.groupby("hhidpn")["_hit"].any().astype(int)
    ref = pd.DataFrame({"hhidpn": sorted(ids)})
    for subtype, stem in [("Stroke / TIA","C053"),("Heart Attack","C040"),("Heart Failure","C048")]:
        ref[subtype] = _any(stem).reindex(ref["hhidpn"]).fillna(0).astype(int).values
    ref["Arrhythmia"] = (df_follow.groupby("hhidpn")["arrhythmia"]
                         .any().astype(int).reindex(ref["hhidpn"]).fillna(0).astype(int).values)
    ref["any_new_cvd"] = (ref[CVD_SUBTYPES].sum(axis=1) > 0).astype(int)
    return ref

print("\nBuilding CVD subtype flags ...")
for label, cohort in [("Depression", dep_cohort), ("Sleep", sleep_cohort)]:
    flags = _build_cvd_flags(cohort)
    cohort.drop(columns=[c for c in CVD_SUBTYPES + ["any_new_cvd"] if c in cohort.columns],
                inplace=True, errors="ignore")
    cohort.drop(columns=["any_new_cvd"], inplace=True, errors="ignore")
    merged = cohort.merge(flags[["hhidpn"] + CVD_SUBTYPES + ["any_new_cvd"]], on="hhidpn", how="left")
    for col in CVD_SUBTYPES + ["any_new_cvd"]:
        merged[col] = merged[col].fillna(0).astype(int)
    if label == "Depression":
        dep_cohort  = merged
    else:
        sleep_cohort = merged
    print(f"  {label}: {len(merged):,}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
# ── Baseline skew by trajectory ────────────────────────────────────────────────
dep_base = dep_cohort.groupby("trajectory")["dep_2010"].mean().reindex(TRAJ_ORDER)
slp_base = sleep_cohort.groupby("trajectory")["sleep_2010"].mean().reindex(TRAJ_ORDER)
print("\nBaseline (2010) mean score by trajectory:")
print("  Depression (0–125): " + "  ".join(f"{t}={dep_base[t]:.1f}" for t in TRAJ_ORDER))
print("  Sleep burden (0–100): " + "  ".join(f"{t}={slp_base[t]:.1f}" for t in TRAJ_ORDER))

print("\nBuilding combined figure ...")

fig = plt.figure(figsize=(22, 11))
gs  = fig.add_gridspec(2, 4,
                        width_ratios=[1.1, 1.3, 1.3, 1.3],
                        wspace=0.28, hspace=0.30)

ax_dep_inc = fig.add_subplot(gs[0, 0])
ax_slp_inc = fig.add_subplot(gs[1, 0])
axes_right = [[fig.add_subplot(gs[row, col + 1]) for col in range(3)] for row in range(2)]

# Panel letter sequence: a–d (row 0), e–h (row 1)
panel_letters = [
    [ax_dep_inc] + axes_right[0],
    [ax_slp_inc] + axes_right[1],
]
for row_idx, row_axes in enumerate(panel_letters):
    for col_idx, ax in enumerate(row_axes):
        letter = chr(ord("a") + row_idx * 4 + col_idx)
        ax.text(0.02, 0.98, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

# ── Left column: CVD incidence bar charts ─────────────────────────────────────
for ax, cohort, domain_color in [
    (ax_dep_inc,  dep_cohort,   V.C_DEP),
    (ax_slp_inc, sleep_cohort,  V.C_SLEEP),
]:
    cvd_traj = (cohort.groupby("trajectory")["any_new_cvd"]
                .agg(n="count", cvd_n="sum")
                .reindex(TRAJ_ORDER))
    cvd_traj["pct"] = cvd_traj["cvd_n"] / cvd_traj["n"] * 100

    bars = ax.bar(
        TRAJ_ORDER, cvd_traj["pct"],
        color=[V.TRAJ_COLORS[t] for t in TRAJ_ORDER],
        width=0.5, edgecolor="white", linewidth=0.5,
    )
    for bar, (ttype, row) in zip(bars, cvd_traj.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{row['pct']:.1f}%",
                ha="center", va="bottom", fontsize=12)

    chi2, dof, p_str = _chi2_str(cohort)
    if chi2 is not None:
        ax.text(0.97, 0.97, f"χ² = {chi2:.2f}, df = {dof}, {p_str}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", linewidth=1.0))

    ax.set_ylabel("New CVD incidence (%)", fontsize=12)
    ax.set_ylim(0, cvd_traj["pct"].max() * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Right 2×3: CVD type breakdown ─────────────────────────────────────────────
for cohort, row_idx in [(dep_cohort, 0), (sleep_cohort, 1)]:
    # For row_max, use exclusive counts (each CVD person in exactly one bin)
    def _exclusive_counts(sub):
        sub_cvd = sub[sub["any_new_cvd"] == 1]
        n_sub   = sub_cvd[CVD_SUBTYPES].sum(axis=1)
        excl    = [int(((sub_cvd[c] == 1) & (n_sub == 1)).sum()) for c in CVD_SUBTYPES]
        multi   = int((n_sub > 1).sum())
        return excl + [multi]

    row_max = max(
        max(_exclusive_counts(cohort[cohort["trajectory"] == t]))
        for t in TRAJ_ORDER
    )
    for col_idx, ttype in enumerate(TRAJ_ORDER):
        ax  = axes_right[row_idx][col_idx]
        sub = cohort[cohort["trajectory"] == ttype]
        n_total = len(sub)
        n_cvd   = int(sub["any_new_cvd"].sum())
        pct_cvd = n_cvd / n_total * 100 if n_total else 0

        # Exclusive bins: single-subtype only; multi = 2+ subtypes
        sub_cvd   = sub[sub["any_new_cvd"] == 1]
        n_sub     = sub_cvd[CVD_SUBTYPES].sum(axis=1)
        counts    = [int(((sub_cvd[c] == 1) & (n_sub == 1)).sum()) for c in CVD_SUBTYPES]
        n_multi   = int((n_sub > 1).sum())
        pcts      = [c / n_cvd * 100 if n_cvd else 0 for c in counts]
        pct_multi = n_multi / n_cvd * 100 if n_cvd else 0

        all_counts = counts + [n_multi]
        all_pcts   = pcts   + [pct_multi]
        all_colors = [BAR_COLORS[c] for c in CVD_SUBTYPES] + [BAR_COLORS["Multiple Events"]]

        bars = ax.bar(range(len(all_counts)), all_counts,
                      color=all_colors,
                      width=0.55, edgecolor="white", linewidth=0.4)
        for bar, n, pct in zip(bars, all_counts, all_pcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + row_max * 0.02,
                    f"n={n}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(range(len(X_LABELS)))
        ax.set_xticklabels(X_LABELS, fontsize=11)
        ax.set_ylim(0, row_max * 1.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Trajectory label only on top row (bottom row aligns with top)
        if row_idx == 0:
            ax.set_title(ttype, color=V.TRAJ_COLORS[ttype], fontweight="bold", fontsize=14)

        ax.text(0.5, 0.97,
                f"n={n_cvd:,} new CVD  ({pct_cvd:.1f}% of {n_total:,})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#555555")

# ── Overall suptitle ──────────────────────────────────────────────────────────
fig.suptitle(
    "CVD Incidence by score trajectory (HRS 2010–2022)",
    fontsize=16, fontweight="bold", y=0.95,
)

# ── Row headers centered above each row (added after tight_layout) ────────────
fig.tight_layout()
pos_dep = ax_dep_inc.get_position()
pos_slp = ax_slp_inc.get_position()
# Row 0 header: just above the depression row axes
fig.text(0.5, pos_dep.y1 + 0.012, "Depression Score",
         ha="center", va="bottom", fontsize=15, fontweight="bold")
# Row 1 header: midpoint of the gap between the two rows
fig.text(0.5, (pos_dep.y0 + pos_slp.y1) / 2, "Sleep Quality Score",
         ha="center", va="center", fontsize=15, fontweight="bold")

# ── Baseline-skew footnote ────────────────────────────────────────────────────
_dep_note = "  ".join(
    f"{t}: {dep_base[t]:.1f}" for t in TRAJ_ORDER
)
_slp_note = "  ".join(
    f"{t}: {slp_base[t]:.1f}" for t in TRAJ_ORDER
)
_footnote = (
    "Note: Trajectory groups reflect overall 2010–2022 slope direction (|Δ| > 1.5 units/2 yr threshold). "
    "Baseline (2010) mean scores by group — "
    f"Depression (0–125 scale): {_dep_note};  "
    f"Sleep burden (0–100 scale): {_slp_note}. "
    "Participants classified as 'Worsened' tend to begin with lower baseline burden, "
    "while those classified as 'Improved' tend to begin with higher baseline burden — "
    "a floor/ceiling artifact of slope-based classification."
)
fig.subplots_adjust(bottom=0.10)
fig.text(
    0.5, 0.01, _footnote,
    ha="center", va="bottom", fontsize=8, color="#444444",
    wrap=True,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
              edgecolor="#cccccc", linewidth=0.8),
)

out1 = OUT_LONG   / "combined_cvd_figure_2010_2022.png"
out2 = OUT_CHERRY / "HRS_combined_cvd_figure.png"
for out_path in [out1, out2]:
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  -> saved {out_path}")
plt.close(fig)
print("\nDone.")
