# RawData — what goes here

This folder is where you place the **raw datasets**. Nothing in it is committed
to git (everything except this README is ignored), and **no participant data
ships with this repository**. You obtain the data yourself, under each program's
data-use terms.

## Where to download

| Dataset | Source | Files you need |
|---------|--------|----------------|
| **HRS RAND** (longitudinal) | <https://hrsdata.isr.umich.edu/> | `randhrs1992_2022v1.dta` |
| **HRS Core** (per-wave items) | <https://hrsdata.isr.umich.edu/> | per-wave `.DA` data files **and** their codebook `.txt` files (2010–2022) |
| **NHANES** 2013–2018 | <https://wwwn.cdc.gov/nchs/nhanes/> | `.xpt` modules per cycle: `DEMO`, `DPQ`, `SLQ`, `MCQ`, `RXQ_RX`, `DIQ`, `BPQ`, `HSQ`, `HUQ` |

## How to lay it out

`ScoringFunctions.py` searches `RawData/` **recursively**, so the exact nesting
is flexible — but a clean layout helps. A suggested structure:

```
RawData/
├── NHANES/
│   ├── 2013-2014/   DEMO_H.xpt, DPQ_H.xpt, SLQ_H.xpt, RXQ_RX_H.xpt, ...
│   ├── 2015-2016/   *_I.xpt
│   └── 2017-2018/   *_J.xpt
└── HRS/
    ├── RandHRS/     randhrs1992_2022v1.dta
    └── waves/
        ├── HRS2016/ H16C_R.da, H16D_R.da, H16PR_R.da, H16C_R.txt, H16D_R.txt, ...
        └── ...      (one folder per wave 2010–2022)
```

## ⚠️ HRS Core `.DA` files need their codebook `.txt`

Fixed-width `.DA` files have **no column headers** — the column positions live in
the matching codebook `.txt` (e.g. `H16C_R.txt` for `H16C_R.da`). HRS distributes
these in separate folders (`h16da/` for data, `h16cb/` for codebooks). That's
fine: the loader searches all of `RawData/` for the matching `.txt` by name. Just
make sure **both** the `.da` and its `.txt` are somewhere under `RawData/`.

## Then what

1. Fill in `../VariableDict.xlsx` (see `../VD_README.md`).
2. Run `python ../ScoringFunctions.py` → scores land in `../SF_OUTPUT/`.
3. For the longitudinal analyses, run the prep pipeline
   (see `../SupplementaryScripts/SS_README.md`).
