"""
StatsHelpers.py
===============
Shared helpers for the analysis layer: a single place for repo paths, common
statistical utilities, and the medication exposure-group machinery used by the
drug-class figures. Imported by the prep scripts and by StatsAnal.ipynb.

    import sys; sys.path.insert(0, "SupplementaryScripts")
    from StatsHelpers import REPO_ROOT, RAW, SCORED, ANALYTIC, FIGS
    from StatsHelpers import significance_label, ci95, build_exposure_groups, run_pairwise_mwu
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository layout — one source of truth for every analysis script.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW       = REPO_ROOT / "RawData"          # raw NHANES / HRS files (git-ignored)
SCORED    = REPO_ROOT / "SF_OUTPUT"        # ScoringFunctions.py outputs
ANALYTIC  = SCORED / "analytic"            # prep-script intermediate CSVs
FIGS      = SCORED / "figures"             # rendered figures
for _d in (ANALYTIC, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# Make ScoringFunctions importable from the repo root (for its loaders / math).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# General statistical utilities
# ---------------------------------------------------------------------------
def significance_label(p: float) -> str:
    """Human-readable significance label for a p-value."""
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return "p < 0.01 **"
    if p < 0.05:
        return "p < 0.05 *"
    return f"p = {p:.3f} (NS)"


def ci95(arr) -> tuple[float, float, float, int]:
    """Return (mean, lo, hi, n) for a 95% normal-approx confidence interval."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        m = float(arr.mean()) if n else float("nan")
        return m, m, m, n
    se = arr.std(ddof=1) / np.sqrt(n)
    m = float(arr.mean())
    return m, m - 1.96 * se, m + 1.96 * se, n


def mwu_effect_r(u: float, n1: int, n2: int) -> float:
    """Rank-biserial-style effect size r = |Z| / sqrt(N) for Mann-Whitney U."""
    from scipy.stats import norm
    n = n1 * n2
    mean_u = n / 2.0
    sd_u = np.sqrt(n * (n1 + n2 + 1) / 12.0)
    if sd_u == 0:
        return float("nan")
    z = (u - mean_u) / sd_u
    return abs(z) / np.sqrt(n1 + n2)


# ---------------------------------------------------------------------------
# Medication exposure groups (non-exclusive) + pairwise Mann-Whitney U.
# Ported from the study's NHANES analysis; used by the drug-class figure.
# ---------------------------------------------------------------------------
def build_exposure_groups(df: pd.DataFrame,
                          score_col: str,
                          classes_col: str = "MedicationClasses",
                          reference_label: str = "none (reference)",
                          min_exposure_n: int = 15) -> dict:
    """Non-exclusive drug-class score pools.

    Each participant contributes their score to every class they take. The
    reference pool is participants with no classified medication. Classes with
    fewer than `min_exposure_n` valid scores are dropped.
    """
    from MedicationClassifier import DRUG_CLASS_MAP
    valid = df[df[score_col].notna()].copy()
    groups: dict = {}
    ref = valid[valid[classes_col] == "none"][score_col].values
    if len(ref) >= min_exposure_n:
        groups[reference_label] = ref
    for cls in sorted(DRUG_CLASS_MAP.keys()):
        mask = valid[classes_col].apply(
            lambda s, c=cls: c in s.split(",") if isinstance(s, str) else False)
        scores = valid.loc[mask, score_col].values
        if len(scores) >= min_exposure_n:
            groups[cls] = scores
    return groups


def run_pairwise_mwu(groups: dict,
                     reference_key: str = "none (reference)") -> pd.DataFrame:
    """Pairwise Mann-Whitney U of each group vs the reference, Holm-corrected."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    ref = groups.get(reference_key)
    if ref is None:
        return pd.DataFrame()
    out = []
    for k in [g for g in groups if g != reference_key]:
        u, p = mannwhitneyu(groups[k], ref, alternative="two-sided")
        out.append({"DrugClass": k, "n_class": len(groups[k]), "n_ref": len(ref),
                    "U": round(u, 1), "p_raw": p,
                    "effect_r": round(mwu_effect_r(u, len(groups[k]), len(ref)), 3)})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    reject, p_holm, _, _ = multipletests(df["p_raw"].values, method="holm")
    df["p_holm"] = p_holm
    df["significant"] = reject

    def _sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    df["label"] = df["p_holm"].apply(_sig)
    return df.sort_values("p_holm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# HRS Core .DA loader (re-exported from ScoringFunctions for the notebook).
# ---------------------------------------------------------------------------
def load_hrs_da(da_path, variables):
    """Load HRS Core fixed-width .DA via ScoringFunctions' loader (needs codebook)."""
    import ScoringFunctions as sf
    sf.RAW_DIR = RAW
    return sf._load_da(Path(da_path), list(variables))
