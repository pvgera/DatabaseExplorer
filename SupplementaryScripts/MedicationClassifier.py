"""
MedicationClassifier.py
=======================
Maps free-text prescription drug names to broad drug classes, and rolls those
classes up to the participant level.

This is NOT part of scoring — the study did not score medication use. It is a
helper for the medication analysis (drug class vs. sleep / depression burden;
see StatsAnal.ipynb, Figure 3). It is kept separate from ScoringFunctions.py on
purpose.

Typical use (NHANES RXQ_RX file, one row per drug per participant):

    import pandas as pd
    from MedicationClassifier import participant_drug_classes, classify_drug

    rx = pd.read_sas("RawData/RXQ_RX_H.xpt", format="xport")
    rx.columns = rx.columns.str.upper()
    rx["drug_class"] = rx["RXDDRUG"].apply(classify_drug)

    per_person = participant_drug_classes(rx, id_col="SEQN", drug_col="RXDDRUG")
    # -> SEQN | MedClass_primary | MedicationClasses ("beta blocker,statin")

Classes are grouped into "brain" (psychiatric) vs. "heart" (cardiovascular) by
the BRAIN_CLASSES / HEART_CLASSES sets below — edit those to taste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Drug-name -> class keyword map. Matching is case-insensitive substring.
# Add brand names or extra generics as needed.
# ---------------------------------------------------------------------------
DRUG_CLASS_MAP: dict[str, list[str]] = {
    "antiarrhythmic":         ["amiodarone", "flecainide", "propafenone", "sotalol",
                               "dronedarone", "quinidine", "disopyramide", "mexiletine"],
    "beta blocker":           ["metoprolol", "atenolol", "carvedilol", "bisoprolol",
                               "propranolol", "nadolol", "labetalol", "nebivolol",
                               "acebutolol", "betaxolol", "esmolol", "pindolol", "timolol"],
    "oral anticoagulant":     ["warfarin", "apixaban", "rivaroxaban", "dabigatran",
                               "edoxaban", "enoxaparin", "coumadin"],
    "ssri":                   ["fluoxetine", "sertraline", "paroxetine", "escitalopram",
                               "citalopram", "fluvoxamine"],
    "snri":                   ["venlafaxine", "duloxetine", "desvenlafaxine", "levomilnacipran"],
    "antidepressant (other)": ["bupropion", "mirtazapine", "trazodone", "nefazodone",
                               "amitriptyline", "nortriptyline", "imipramine", "doxepin"],
    "anxiolytic":             ["alprazolam", "lorazepam", "diazepam", "clonazepam",
                               "buspirone", "oxazepam", "temazepam", "triazolam"],
    "antipsychotic":          ["quetiapine", "risperidone", "aripiprazole", "olanzapine",
                               "haloperidol", "clozapine", "ziprasidone", "lurasidone"],
    "arb":                    ["losartan", "valsartan", "irbesartan", "olmesartan",
                               "candesartan", "telmisartan", "azilsartan"],
    "ace inhibitor":          ["lisinopril", "enalapril", "ramipril", "captopril",
                               "benazepril", "quinapril", "fosinopril", "perindopril"],
    "diuretic":               ["furosemide", "hydrochlorothiazide", "spironolactone",
                               "chlorthalidone", "indapamide", "torsemide", "bumetanide"],
    "statin":                 ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
                               "lovastatin", "fluvastatin", "pitavastatin"],
}

# Brain (psychiatric) vs. heart (cardiovascular) groupings used by the figures.
BRAIN_CLASSES: set[str] = {"ssri", "snri", "anxiolytic",
                           "antidepressant (other)", "antipsychotic"}
HEART_CLASSES: set[str] = {"ace inhibitor", "antiarrhythmic", "arb", "beta blocker",
                           "diuretic", "oral anticoagulant", "statin"}


def classify_drug(drug_name) -> str:
    """Return the drug class for one drug name, or 'unclassified'."""
    if isinstance(drug_name, bytes):
        drug_name = drug_name.decode("utf-8", errors="replace")
    if not isinstance(drug_name, str):
        return "unclassified"
    dn = drug_name.lower().strip()
    for cls, keywords in DRUG_CLASS_MAP.items():
        if any(kw in dn for kw in keywords):
            return cls
    return "unclassified"


def participant_drug_classes(df_rx: pd.DataFrame,
                             id_col: str = "SEQN",
                             drug_col: str = "RXDDRUG") -> pd.DataFrame:
    """Roll a long prescription table up to one row per participant.

    Returns columns: <id_col> | MedClass_primary | MedicationClasses
    where MedicationClasses is a comma-separated, de-duplicated list of the
    classified classes that participant takes. Participants with no classified
    drug are simply absent (merge with how='left' and fillna('none')).
    """
    by_person: dict = {}
    for _, row in df_rx.iterrows():
        pid = row[id_col]
        cls = classify_drug(row.get(drug_col, np.nan))
        if cls != "unclassified":
            by_person.setdefault(pid, []).append(cls)
    out = []
    for pid, classes in by_person.items():
        out.append({id_col: pid,
                    "MedClass_primary": classes[0],
                    "MedicationClasses": ",".join(sorted(set(classes)))})
    return pd.DataFrame(out)


def classify_combination(med_classes: str) -> str:
    """Collapse a 'beta blocker,statin' string to 'none' / 'single' / 'a + b'."""
    if not isinstance(med_classes, str):
        return "none"
    parts = sorted(p.strip() for p in med_classes.split(",")
                   if p.strip() and p.strip().lower() != "none")
    if not parts:
        return "none"
    if len(parts) == 1:
        return "single"
    return " + ".join(parts)


def drug_group(cls: str) -> str:
    """Return 'brain', 'heart', or 'other' for a drug class."""
    if cls in BRAIN_CLASSES:
        return "brain"
    if cls in HEART_CLASSES:
        return "heart"
    return "other"
