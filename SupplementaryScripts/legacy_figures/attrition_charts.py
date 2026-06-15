"""
attrition_charts.py
===================
Three-panel participant attrition flowchart.
  a — Sleep vs Depression descriptive cohort  (Fig 1)
        HRS per-person means from RAND pooled only
  b — NHANES pooled 2013-2018, main analytic cohort  (Fig 2)
  c — HRS RAND longitudinal, Cox analysis pipeline   (Fig 5, 6)

Panel labels, fonts, and colours match CherryPicked2 convention:
  - C_COVARIATE steel blue for main box borders
  - C_SLEEP amber for exclusion box borders
  - Bold panel letters at (-0.05, 1.04) relative to each axis
    (matching sleep_dep_scatter.py and nhanes_combined_drug_ci.py)

Key numbers verified from data files:
  HRS scatter: 30,033 unique indiv -> 28,753 with >=1 wave both valid
  NHANES main: 8,535 -> 7,511 valid dep -> 5,042 valid dep+sleep
  HRS Cox    : 42,839 non-prevalent -> 18,693 age>=50 -> 17,492 complete cases

Output: results/CherryPicked2/attrition_charts.png
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engine.viz_style as V

_EDIT = "--edit" in sys.argv
V.apply()

OUT     = Path(__file__).parent
FIG_DPI = 300

# ─── project colours ─────────────────────────────────────────────────────────
C_MAIN_FILL   = "#FFFFFF"
C_MAIN_BORDER = V.C_COVARIATE    # #4A7098 steel blue
C_EXCL_FILL   = "#FFF8F0"
C_EXCL_BORDER = V.C_SLEEP        # #E07828 amber
C_FINAL_FILL  = "#F0F4F8"
C_FINAL_BDR   = "#C8102E"        # crimson for emphasis
C_ARROW       = "#333333"
C_MAIN_TXT    = "#1A1A1A"
C_SUB_TXT     = "#4A4A4A"
C_EXCL_TXT    = "#7A3800"

MAIN_FS  = 8.0
SUB_FS   = 7.3
EXCL_FS  = 7.0
PANEL_FS = 14    # matches Figure 1 / Figure 2

CYCLE_COLORS = {
    "2013-2014": "#5B9BD5",
    "2015-2016": "#E07828",
    "2017-2018": "#2DA84B",
}


# ─── drawing primitives ───────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, title, subtitle=None,
         fc=C_MAIN_FILL, ec=C_MAIN_BORDER, lw=1.5, is_final=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    y_t = cy + (0.026 if subtitle else 0)
    ax.text(cx, y_t, title,
            ha="center", va="center",
            fontsize=MAIN_FS, fontweight="bold" if is_final else "normal",
            color=C_MAIN_TXT, zorder=4, multialignment="center")
    if subtitle:
        ax.text(cx, cy - 0.026, subtitle,
                ha="center", va="center",
                fontsize=SUB_FS, color=C_SUB_TXT, zorder=4)


def _excl(ax, cx, cy, w, h, text):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=C_EXCL_FILL, edgecolor=C_EXCL_BORDER,
        linewidth=1.2, zorder=3))
    ax.text(cx, cy, text,
            ha="center", va="center", fontsize=EXCL_FS,
            color=C_EXCL_TXT, zorder=4, multialignment="center")


def _arrow(ax, cx, y_from, y_to):
    ax.annotate("",
        xy=(cx, y_to + 0.007), xytext=(cx, y_from - 0.007),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4), zorder=5)


def _branch(ax, cx, y_mid, excl_cx, ew):
    ax.plot([cx, excl_cx - ew/2 - 0.006], [y_mid, y_mid],
            color=C_ARROW, lw=1.2, zorder=5)
    ax.annotate("",
        xy=(excl_cx - ew/2, y_mid), xytext=(cx + 0.005, y_mid),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2), zorder=5)


def _label(ax, letter):
    """Bold panel letter outside top-left — matches Figure 1 style."""
    ax.text(-0.05, 1.04, letter,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PANEL_FS, fontweight="bold", color=C_MAIN_TXT)


# ─── figure: 3 equal columns ──────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 4.5), facecolor="white")
ax_a = fig.add_subplot(1, 3, 1)
ax_b = fig.add_subplot(1, 3, 2)
ax_c = fig.add_subplot(1, 3, 3)

for ax in (ax_a, ax_b, ax_c):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

plt.subplots_adjust(wspace=0.0, left=0.02, right=0.98, top=0.82, bottom=0.12)

MW = 0.56; MH = 0.082; EW = 0.30; ECX = 0.87; MCX = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL a — HRS per-person means  (Figure 1a, scatter)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_a
_label(ax, "a")
ax.text(MCX, 1.02, "HRS Pooled",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=C_MAIN_BORDER)

MHA  = MH + 0.01
MHF_A = 0.135

A_Y = [0.74, 0.44]

# ── single intermediate box ───────────────────────────────────────────────────
_box(ax, MCX, A_Y[0], MW, MHA,
     "RAND HRS pooled\nAges 50+", "n = 30,033  (unique individuals)")

# ── final box ─────────────────────────────────────────────────────────────────
fy_a = A_Y[1]
ax.add_patch(mpatches.FancyBboxPatch(
    (MCX - MW/2, fy_a - MHF_A/2), MW, MHF_A,
    boxstyle="round,pad=0.015",
    facecolor=C_FINAL_FILL, edgecolor=C_FINAL_BDR,
    linewidth=2.2, zorder=3))
ax.text(MCX, fy_a + MHF_A/2 - 0.025,
        "-1,280 for incomplete DB and/or SQB",
        ha="center", va="center",
        fontsize=7.5, color="#888888", style="italic", zorder=4)
ax.text(MCX, fy_a + 0.008, "HRS descriptive cohort",
        ha="center", va="center",
        fontsize=MAIN_FS, fontweight="bold", color=C_MAIN_TXT, zorder=4)
ax.text(MCX, fy_a - MHF_A/2 + 0.028, "n = 28,753  |  6 waves pooled",
        ha="center", va="center",
        fontsize=SUB_FS, color=C_SUB_TXT, zorder=4)

_arrow(ax, MCX, A_Y[0] - MHA/2, fy_a + MHF_A/2)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL b — NHANES pooled  (Figure 2)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_b
_label(ax, "b")
ax.text(MCX, 1.02, "NHANES Pooled",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=C_MAIN_BORDER)

MH2  = MH + 0.01
MHF_B = 0.135

B_Y = [0.74, 0.44]

# ── single intermediate box ───────────────────────────────────────────────────
_box(ax, MCX, B_Y[0], MW + 0.04, MH2,
     "NHANES pooled\nAges 50+", "n = 8,535")

# ── final box ─────────────────────────────────────────────────────────────────
fy_b = B_Y[1]
ax.add_patch(mpatches.FancyBboxPatch(
    (MCX - (MW + 0.04)/2, fy_b - MHF_B/2), MW + 0.04, MHF_B,
    boxstyle="round,pad=0.015",
    facecolor=C_FINAL_FILL, edgecolor=C_FINAL_BDR,
    linewidth=2.2, zorder=3))
ax.text(MCX, fy_b + MHF_B/2 - 0.025,
        "-3,493 for incomplete DB and/or SQB",
        ha="center", va="center",
        fontsize=7.5, color="#888888", style="italic", zorder=4)
ax.text(MCX, fy_b + 0.008, "Primary NHANES cohort",
        ha="center", va="center",
        fontsize=MAIN_FS, fontweight="bold", color=C_MAIN_TXT, zorder=4)
ax.text(MCX, fy_b - MHF_B/2 + 0.028, "n = 5,042  (cross-sectional)",
        ha="center", va="center",
        fontsize=SUB_FS, color=C_SUB_TXT, zorder=4)

_arrow(ax, MCX, B_Y[0] - MH2/2, fy_b + MHF_B/2)

ax.text(0.01, 0.50, "Cross-sectional\n(no follow-up)",
        ha="left", va="center", fontsize=7, color="#888888",
        rotation=90, transform=ax.transAxes)



# ═══════════════════════════════════════════════════════════════════════════════
# PANEL c — Landmark Cox Regression
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_c
_label(ax, "c")
ax.text(MCX, 1.02, "Landmark Cox Regression",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=C_MAIN_BORDER)

# 3 boxes, evenly spaced
C_Y = [0.80, 0.60, 0.40]
MHC = 0.10

c_nodes = [
    (C_Y[0], "RAND HRS 1992-2022 longitudinal file", "45,234 total participants"),
    (C_Y[1], "Exclude arrhythmia prevalent\nat 2010 baseline", "n = 42,839"),
    (C_Y[2], "Age >= 50 at 2010 baseline",            "n = 18,693"),
]
for (cy, title, sub) in c_nodes:
    fin = (cy == C_Y[2])
    _box(ax, MCX, cy, MW, MHC, title, sub,
         fc=C_FINAL_FILL if fin else C_MAIN_FILL,
         ec=C_FINAL_BDR  if fin else C_MAIN_BORDER,
         lw=2.2 if fin else 1.5, is_final=fin)

for i in range(len(C_Y) - 1):
    _arrow(ax, MCX, C_Y[i] - MHC/2, C_Y[i+1] + MHC/2)


# ─── shared caption ───────────────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    "Figure. Participant attrition by analysis.  "
    "a: HRS descriptive scatter cohort (Fig 1a) — RAND pooled per-person means, n=28,753.  "
    "b: NHANES pooled 2013-2018, cross-sectional main cohort (Figs 1b, 2).  "
    "c: HRS RAND longitudinal Cox pipeline (2010 baseline, 12-yr follow-up, Figs 5 & 6).  "
    "Dep = CESD-8 (HRS) / DPQ (NHANES).  Sleep = composite burden score.  "
    "NHANES RIDAGEYR topcoded at 80 (80+ group included).",
    ha="center", va="bottom", fontsize=7.5, color="#555555"
)

out_path = OUT / "attrition_charts.png"
fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path}")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close()
