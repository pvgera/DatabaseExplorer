"""
ScoringFunctions.py
===================
A single, variable-dictionary-driven scoring engine for public-health
questionnaire data (NHANES, HRS, and any dataset you describe in the dictionary).

It does three things:

  1.  READS your variable dictionary           ->  VariableDict.xlsx (repo root)
  2.  LOADS the raw data files it points to     ->  RawData/  (.xpt / .dta / .DA)
  3.  SCORES composites with a Sum-Score Model  ->  SF_OUTPUT/<your_name>.csv

There is no other script to run. Everything that used to live across several
dataset-specific files (NHANES loader, HRS .DA loader, HRS RAND loader, the
normalisation helpers, and the generic scorer) is consolidated here and driven
entirely by the dictionary. To change WHAT is scored, HOW items are coded, or
which direction counts as "more burden", you edit VariableDict.xlsx — not this
file.

----------------------------------------------------------------------------
THE SUM-SCORE MODEL
----------------------------------------------------------------------------
For one participant and one composite (e.g. "SleepQualityBurden"):

                   sum over valid items of  (r - min) / (max - min)
    score  =  --------------------------------------------------------  x 100
                              y  (number of valid items)

where r is the participant's response, and (min, max) are the per-item
"minimum effect" / "maximum effect" bounds from the dictionary. Higher score =
more burden. Reverse-coded items are handled simply by setting min > max for
that row (see VD_README.md). Items are left un-weighted — this is the same
Sum-Score Model used in the associated study (Schlechter et al., 2022).

A composite is only returned if the participant answered at least
`min_items_fraction` of that composite's items; otherwise the score is NaN.

----------------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------------
  1. Fill in VariableDict.xlsx           (see VD_README.md)
  2. Put raw files in RawData/           (see RawData/RD_README.md)
  3. Edit the DATASETS dict below to pick which datasets to score and to
     choose your output file names.
  4. Run:   python ScoringFunctions.py

----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION  — edit this block
# ============================================================================
ROOT       = Path(__file__).resolve().parent
DICT_PATH  = ROOT / "VariableDict.xlsx"     # the variable dictionary (template name — do not rename)
RAW_DIR    = ROOT / "RawData"               # where your downloaded raw files live
OUTPUT_DIR = ROOT / "SF_OUTPUT"             # scored CSVs land here

# One entry per dataset you want to score.
#   key          : must match the `dataset` column in VariableDict.xlsx
#   output_name  : the CSV file written to SF_OUTPUT/  (name it whatever you like)
#   id           : the participant-id column for this dataset
#                    NHANES        -> "SEQN"
#                    HRS RAND .dta -> "HHIDPN"
#                    HRS Core .DA  -> "HHID_PN"  (built automatically from HHID + PN)
#
# Comment out a dataset to skip it.
DATASETS = {
    "NHANES_pilot": {"output_name": "nhanes_scored.csv", "id": "SEQN"},
    "HRS_RAND":     {"output_name": "hrs_rand_scored.csv", "id": "HHIDPN"},
    "HRS_2016":     {"output_name": "hrs_2016_scored.csv", "id": "HHID_PN"},
    "HRS_2022":     {"output_name": "hrs_2022_scored.csv", "id": "HHID_PN"},
}

# ============================================================================
# NORMALISATION & MISSING-VALUE HELPERS  (the scoring math lives here)
# ============================================================================

# NHANES "refused / don't know" sentinel codes, across question widths.
NHANES_MISSING: frozenset = frozenset({
    7, 9, 77, 99, 777, 999, 7777, 9999,
    7.0, 9.0, 77.0, 99.0, 777.0, 999.0, 7777.0, 9999.0,
})

# Values always treated as missing in HRS-style data unless overridden per row.
HRS_MISSING_BASE: frozenset = frozenset({
    np.nan, "", " ", ".", "Blank", "8", "9", 8, 9, 8.0, 9.0,
})


def normalize_item(val: float, minv: float, maxv: float) -> float:
    """Normalise one response to a 0-10 burden scale.

    Works for normal scales (minv < maxv) and reverse-coded scales (minv > maxv).
    Returns NaN if val is NaN.
    """
    if np.isnan(val):
        return np.nan
    return (val - minv) / (maxv - minv) * 10.0


def score_pct(numerator: float, denominator: float) -> float:
    """Convert a weighted sum to a 0-100% score. NaN if denominator <= 0."""
    if denominator <= 0:
        return np.nan
    return round((numerator / denominator) * 100.0, 3)


def clean_val(val, valid_min=None, valid_max=None) -> float:
    """Return NaN for missing / sentinel / out-of-range values, else a float.

    Handles NHANES coded-missing (7, 9, 77, ...), the XPT byte sentinel that
    pandas reads as ~5.4e-79 for a true 0 response, and optional range bounds.
    """
    if pd.isna(val):
        return np.nan
    try:
        v = float(val)
    except (TypeError, ValueError):
        return np.nan
    if 0 < abs(v) < 1e-50:           # XPT reads a "0" response as ~5.4e-79
        v = 0.0
    if v in NHANES_MISSING:
        return np.nan
    if valid_min is not None and v < valid_min:
        return np.nan
    if valid_max is not None and v > valid_max:
        return np.nan
    return v


def parse_missing_set(raw, base=None) -> set:
    """Parse the dictionary's `missing_values` cell into a set of missing codes."""
    result = set(HRS_MISSING_BASE) if base is None else set(base)
    if pd.isna(raw):
        return result
    for token in str(raw).replace("(", "").replace(")", "").split(","):
        token = token.strip().strip('"').strip("'")
        if not token:
            continue
        try:
            result.add(float(token))
            result.add(int(float(token)))
            result.add(str(int(float(token))))
        except Exception:
            result.add(token)
    return result


# ============================================================================
# VARIABLE DICTIONARY
# ============================================================================

# Canonical column names this engine expects, after header normalisation.
_REQUIRED_COLS = {
    "composite_name", "variable_name", "dataset", "file_name",
    "composite_type", "min_effect", "max_effect", "min_items_fraction",
}


def load_dictionary(path: Path = DICT_PATH) -> pd.DataFrame:
    """Read VariableDict.xlsx and normalise its column headers.

    Headers are lower-cased and spaces -> underscores, so "Composite Name",
    "composite_name", and "Composite_Name" all work.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Variable dictionary not found at {path}. "
            f"Fill in the VariableDict.xlsx template (see VD_README.md)."
        )
    # Read the "Dictionary" sheet if present (the template ships a separate
    # "Legend" sheet); otherwise fall back to the first sheet.
    xls = pd.ExcelFile(path)
    sheet = "Dictionary" if "Dictionary" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = (
        df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    # An "include" column is optional; default everything to included.
    if "include" not in df.columns:
        df["include"] = "include"
    df["include"] = df["include"].fillna("include").astype(str).str.strip().str.lower()

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"VariableDict.xlsx is missing required columns: {sorted(missing)}.\n"
            f"Found columns: {sorted(df.columns)}\nSee VD_README.md."
        )
    # Drop separator / comment rows with no variable name.
    df = df[df["variable_name"].notna()].copy()
    df["composite_type"] = df["composite_type"].fillna("passthrough").astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def dataset_rows(vd: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Included rows for one dataset."""
    mask = (vd["dataset"] == dataset) & (vd["include"] != "exclude")
    rows = vd[mask].copy()
    if rows.empty:
        raise ValueError(
            f"No included rows for dataset={dataset!r}. "
            f"Datasets present in the dictionary: {sorted(vd['dataset'].dropna().unique())}"
        )
    return rows


# ============================================================================
# RAW-FILE LOADERS  (dispatch by file extension)
# ============================================================================

def _parse_codebook_colspecs(cb_path: Path) -> dict:
    """Parse an HRS ASCII codebook (.txt) -> {VARNAME: (start, end_exclusive)}.

    HRS Core .DA files are fixed-width with no header; the column positions come
    from the matching codebook .txt (e.g. H16C_R.txt sits beside H16C_R.da).
    """
    import re
    specs, pos, current = {}, 0, None
    varname_pat = re.compile(r"^([A-Z][A-Z0-9_]+)\s+\S")
    meta_pat = re.compile(r"Type:\s*(?:Numeric|Character)\s+Width:\s*(\d+)", re.IGNORECASE)
    with open(cb_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line and line[0] not in " =-{":
                m = varname_pat.match(line)
                if m:
                    current = m.group(1)
            m2 = meta_pat.search(line)
            if m2 and current:
                width = int(m2.group(1))
                specs[current] = (pos, pos + width)
                pos += width
                current = None
    return specs


def _load_da(da_path: Path, variables: list[str]) -> pd.DataFrame:
    """Load HRS Core fixed-width .DA. Requires a same-stem .txt codebook beside it.

    Always includes HHID + PN and builds the HHID_PN join key.
    """
    cb_path = da_path.with_suffix(".txt")
    if not cb_path.exists():
        # HRS downloads often separate data (h16da/) from codebooks (h16cb/);
        # search RawData for a same-stem .txt anywhere under it.
        matches = list(RAW_DIR.rglob(cb_path.name))
        if matches:
            cb_path = matches[0]
        else:
            raise FileNotFoundError(
                f"Codebook not found for {da_path.name}: expected {cb_path.name} "
                f"beside it or anywhere under {RAW_DIR}. HRS Core .DA files need "
                f"their codebook .txt to locate columns (see RawData/RD_README.md)."
            )
    specs = _parse_codebook_colspecs(cb_path)
    need = ["HHID", "PN"] + [v.upper() for v in variables]
    need = [v for v in dict.fromkeys(need) if v in specs]      # dedupe, keep order
    colspecs = [specs[v] for v in need]
    df = pd.read_fwf(da_path, colspecs=colspecs, header=None, names=need,
                     dtype=str, encoding="latin-1")
    for col in df.columns:
        df[col] = df[col].str.strip()
    df["HHID"] = df["HHID"].str.zfill(6)
    df["PN"] = df["PN"].str.zfill(3)
    df["HHID_PN"] = df["HHID"] + df["PN"]
    # Convert non-key columns to numeric where possible.
    for col in df.columns:
        if col not in ("HHID", "PN", "HHID_PN"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def _load_dta(path: Path) -> pd.DataFrame:
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_file(file_name: str, variables: list[str]) -> pd.DataFrame:
    """Dispatch to the right reader based on the file extension."""
    path = RAW_DIR / file_name
    if not path.exists():
        # also try a recursive search so users can nest by cohort/wave folders
        matches = list(RAW_DIR.rglob(file_name))
        if not matches:
            raise FileNotFoundError(
                f"Raw file {file_name!r} not found under {RAW_DIR}. "
                f"See RawData/RD_README.md for placement."
            )
        path = matches[0]
    ext = path.suffix.lower()
    if ext == ".xpt":
        return _load_xpt(path)
    if ext == ".dta":
        return _load_dta(path)
    if ext == ".da":
        return _load_da(path, variables)
    raise ValueError(f"Unsupported file type {ext!r} for {file_name}. "
                     f"Supported: .xpt (NHANES), .dta (HRS RAND), .da (HRS Core).")


# ============================================================================
# SCORING
# ============================================================================

def _clean_for(dataset: str, raw, missing_set: set | None):
    """Apply the right missing-value rule for a dataset and return a float/NaN."""
    if dataset.upper().startswith("NHANES"):
        return clean_val(raw)
    if raw in missing_set or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def score_composite(merged: pd.DataFrame, items: pd.DataFrame, dataset: str) -> pd.Series:
    """Sum-Score one composite across `merged` rows. Returns a 0-100 Series."""
    fracs = items["min_items_fraction"].dropna()
    threshold = float(fracs.mode().iloc[0]) if not fracs.empty else 0.5
    n_items = len(items)

    is_nhanes = dataset.upper().startswith("NHANES")
    missing_sets = {} if is_nhanes else {
        r["variable_name"]: parse_missing_set(r.get("missing_values"))
        for _, r in items.iterrows()
    }

    def _row_score(row) -> float:
        weighted_sum, n_valid = 0.0, 0
        for _, it in items.iterrows():
            var = it["variable_name"]
            if var not in row:
                continue
            val = _clean_for(dataset, row[var], missing_sets.get(var))
            if np.isnan(val):
                continue
            norm = normalize_item(val, float(it["min_effect"]), float(it["max_effect"]))
            if np.isnan(norm):
                continue
            weighted_sum += norm
            n_valid += 1
        if n_items == 0 or (n_valid / n_items) < threshold:
            return np.nan
        return score_pct(weighted_sum, n_valid * 10.0)

    return merged.apply(_row_score, axis=1)


def passthrough_value(merged: pd.DataFrame, row: pd.Series, dataset: str) -> pd.Series:
    """Return one cleaned (un-scored) variable — e.g. age, sex, a disease flag."""
    var = row["variable_name"]
    if var not in merged.columns:
        return pd.Series(np.nan, index=merged.index)
    if dataset.upper().startswith("NHANES"):
        return merged[var].apply(lambda v: clean_val(v))
    mset = parse_missing_set(row.get("missing_values"))
    return merged[var].apply(lambda v: _clean_for(dataset, v, mset))


def score_dataset(vd: pd.DataFrame, dataset: str, id_col: str) -> pd.DataFrame:
    """Load every file this dataset needs, merge on the id, and build a wide table
    of pass-through variables + scored composites."""
    rows = dataset_rows(vd, dataset)
    print(f"\n=== {dataset}  ({len(rows)} dictionary rows) ===")

    # 1) Load each needed file once, requesting only the variables we need.
    files = {}
    for fn, grp in rows.groupby("file_name"):
        try:
            files[fn] = load_file(str(fn), grp["variable_name"].astype(str).tolist())
            print(f"  loaded {fn}: {len(files[fn]):,} rows")
        except FileNotFoundError as e:
            print(f"  SKIP {fn}: {e}")

    if not files:
        print(f"  No files loaded for {dataset}; skipping.")
        return pd.DataFrame()

    # 2) Merge all files on the id column.
    merged = None
    for fn, df in files.items():
        if id_col not in df.columns:
            print(f"  WARNING: id column {id_col!r} not in {fn}; available: "
                  f"{[c for c in df.columns[:8]]}...")
            continue
        keep = [id_col] + [c for c in df.columns if c != id_col]
        merged = df[keep] if merged is None else merged.merge(
            df[keep], on=id_col, how="outer", suffixes=("", f"_{fn}")
        )
    if merged is None:
        print(f"  Could not build a merged frame for {dataset} (id {id_col!r} missing).")
        return pd.DataFrame()

    out = pd.DataFrame({id_col: merged[id_col]})

    # 3) Pass-through variables (un-scored: age, sex, flags, ...).
    pass_rows = rows[rows["composite_type"] == "passthrough"]
    for _, r in pass_rows.iterrows():
        if str(r["variable_name"]) == id_col:
            continue
        colname = str(r["composite_name"]) if pd.notna(r["composite_name"]) else str(r["variable_name"])
        out[colname] = passthrough_value(merged, r, dataset).values

    # 4) Scored composites (Sum-Score Model).
    score_rows = rows[rows["composite_type"] == "score"]
    for comp, items in score_rows.groupby("composite_name"):
        out[str(comp)] = score_composite(merged, items, dataset).values
        n_valid = out[str(comp)].notna().sum()
        print(f"  scored {comp}: {n_valid:,} valid / {len(out):,} rows "
              f"({len(items)} items)")

    return out


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 68)
    print("  ScoringFunctions.py — Sum-Score engine")
    print(f"  Dictionary : {DICT_PATH}")
    print(f"  Raw data   : {RAW_DIR}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("=" * 68)

    vd = load_dictionary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    any_written = False
    for dataset, cfg in DATASETS.items():
        if dataset not in vd["dataset"].values:
            print(f"\n(skip) {dataset}: not present in the dictionary.")
            continue
        scored = score_dataset(vd, dataset, cfg["id"])
        if scored.empty:
            continue
        out_path = OUTPUT_DIR / cfg["output_name"]
        scored.to_csv(out_path, index=False)
        print(f"  -> wrote {out_path.relative_to(ROOT)}  ({len(scored):,} rows, "
              f"{scored.shape[1]} cols)")
        any_written = True

    print("\n" + "=" * 68)
    if any_written:
        print(f"  Done. Scored files are in {OUTPUT_DIR.relative_to(ROOT)}/")
    else:
        print("  Nothing was written. Check that RawData/ holds your files and that\n"
              "  DATASETS keys match the `dataset` column in VariableDict.xlsx.")
    print("=" * 68)


if __name__ == "__main__":
    sys.exit(main())
