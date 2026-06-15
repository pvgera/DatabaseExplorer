# Filling in the Variable Dictionary

`VariableDict.xlsx` is the control panel for the whole project. You describe
your questionnaire items here, and `ScoringFunctions.py` does the rest — no
coding required. Open it in **Excel or Google Sheets**.

> **Do not rename the file.** `ScoringFunctions.py` looks for `VariableDict.xlsx`
> in the repo root by that exact name.

The workbook has two sheets:

- **Dictionary** — the sheet the engine reads. One row per variable.
- **Legend** — a description of every column (also summarised below).

The Dictionary sheet ships with **example rows** (highlighted amber, marked
`include = exclude` so they are ignored). They demonstrate every pattern. Once
you understand them, delete them and add your own.

---

## The two kinds of row

Every row is one of two types, set in the **`composite_type`** column:

| `composite_type` | meaning | examples |
|------------------|---------|----------|
| `score` | an item that gets summed into a composite score (0–100) | each PHQ-9 / CESD-8 / sleep question |
| `passthrough` | a variable emitted as its cleaned raw value, **not** scored | id, age, sex, BMI, a disease flag |

Rows that share the same **`composite_name`** and are typed `score` are summed
together into one score. So all ten PHQ-9 items get `composite_name =
DepressionBurden`, `composite_type = score`.

---

## The columns

| column | required? | what to put |
|--------|-----------|-------------|
| `composite_name` | ✅ | The score this row feeds (e.g. `DepressionBurden`), or the output column name for a pass-through. |
| `variable_name` | ✅ | The **exact** raw column name in the source file (e.g. `DPQ010`). For HRS `.DA`, the name as it appears in the codebook `.txt`. |
| `dataset` | ✅ | A label you choose (e.g. `NHANES_pilot`). Must match a key in the `DATASETS` block of `ScoringFunctions.py`. |
| `file_name` | ✅ | The raw file holding this variable (e.g. `DPQ_H.xpt`). The engine finds it under `RawData/`. |
| `role` | — | Free-text tag for your own grouping. Not used by the math. |
| `composite_type` | ✅ | `score` or `passthrough` (see above). |
| `include` | — | `include` (default) or `exclude` (keep the row but skip it when scoring). |
| `min_effect` | score rows | The response value meaning **least** burden. |
| `max_effect` | score rows | The response value meaning **most** burden. |
| `min_items_fraction` | score rows | Fraction (0–1) of a composite's items a participant must answer to get a score. Use the same value on every row of that composite. |
| `missing_values` | — | Comma-separated codes to treat as missing (e.g. `8, 9, Blank`). NHANES sentinels (7, 9, 77, 99, …) are automatic. |
| `wave` | — | Optional study wave/year. |
| `definition` | — | Human-readable description. |
| `notes` | — | Anything else. |

---

## Worked example: a 3-item sleep composite

Say your sleep instrument has three questions, each answered 0–4:

1. *"How often do you have trouble falling asleep?"* (0 = never … 4 = always)
2. *"How often do you wake during the night?"* (0 = never … 4 = always)
3. *"How often do you feel rested?"* (0 = never … 4 = always)

Questions 1 and 2: a **higher** answer means **worse** sleep → normal coding,
`min_effect = 0`, `max_effect = 4`.

Question 3 is **reverse-coded**: feeling rested is *good*, so a higher answer
means *less* burden. Flip the bounds → `min_effect = 4`, `max_effect = 0`. **No
code change needed** — the formula inverts automatically.

| composite_name | variable_name | composite_type | min_effect | max_effect | min_items_fraction |
|---|---|---|---|---|---|
| SleepQualityBurden | SLQ_FALL  | score | 0 | 4 | 0.67 |
| SleepQualityBurden | SLQ_WAKE  | score | 0 | 4 | 0.67 |
| SleepQualityBurden | SLQ_RESTED| score | 4 | 0 | 0.67 |

A participant who answers ≥ 2 of the 3 items (0.67) gets a 0–100 burden score;
fewer than 2 → `NaN`.

---

## Reverse coding — the one rule to remember

> **Higher answer = more burden?** → `min_effect < max_effect` (normal)
> **Higher answer = less burden?** → `min_effect > max_effect` (reverse)

This is exactly how the associated study handled its "minimum effect value" /
"maximum effect value" columns.

---

## After you fill it in

1. Make sure each `dataset` label matches a key in the `DATASETS` block of
   `ScoringFunctions.py`.
2. Put your raw files in `RawData/` — see
   [RawData/RD_README.md](RawData/RD_README.md).
3. Run `python ScoringFunctions.py`. Scores land in `SF_OUTPUT/`.

See [SF_README.md](SF_README.md) for what the engine does and the scoring
equation.
