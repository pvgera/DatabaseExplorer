"""
sleep_dep_scatter.py
Runs HRS and NHANES through the *same* plot function so both figures
are guaranteed to have identical formatting.

HRS  : per-person means across pooled waves, coloured by wave count
NHANES: one row per participant (cross-sectional), coloured by cycle

Outputs:
  results/CherryPicked2/HRS_sleep_dep_scatter.png
  results/CherryPicked2/NHANES_sleep_dep_scatter.png
  results/CherryPicked2/combined_sleep_dep_scatter.png
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engine.viz_style as V

_EDIT = "--edit" in sys.argv
V.apply()

OUT     = ROOT / "results" / "CherryPicked2"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DPI = 150

# ──────────────────────────────────────────────────────────────────────────────
# Shared plot function — draws into a provided Axes object
# ──────────────────────────────────────────────────────────────────────────────

def _draw_scatter(ax, sleep_vals, dep_vals, point_colors,
                  legend_handles, legend_title,
                  title, x_label, y_label, n_label,
                  panel_letter=None):
    np.random.seed(42)
    jx = np.asarray(sleep_vals) + np.random.uniform(-2.5, 2.5, size=len(sleep_vals))
    jy = np.asarray(dep_vals)   + np.random.uniform(-2.5, 2.5, size=len(dep_vals))

    rho, p = spearmanr(sleep_vals, dep_vals)
    sm     = lowess(dep_vals, sleep_vals, frac=0.4, return_sorted=True)

    if   p < 0.001: sig = "p < 0.001 ***"
    elif p < 0.01:  sig = f"p = {p:.3f} **"
    elif p < 0.05:  sig = f"p = {p:.3f} *"
    else:           sig = f"p = {p:.3f} ns"

    ax.scatter(jx, jy, c=point_colors, alpha=0.35, s=12)
    ax.plot(sm[:, 0], sm[:, 1], color=V.C_LOWESS, lw=2)

    lowess_line = Line2D([0], [0], color=V.C_LOWESS, lw=2, label="LOWESS")
    ax.legend(handles=legend_handles + [lowess_line],
              fontsize=11, loc="upper right",
              title=legend_title, title_fontsize=10)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    ax.text(0.03, 0.97,
            f"Spearman ρ = {rho:.3f}\n{sig}\nn={n_label}",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", linewidth=1.2, alpha=0.85))

    if panel_letter:
        ax.text(-0.05, 1.04, panel_letter,
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=14, fontweight="bold")


def sleep_dep_scatter(sleep_vals, dep_vals, point_colors,
                      legend_handles, legend_title,
                      title, x_label, y_label,
                      n_label, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_scatter(ax, sleep_vals, dep_vals, point_colors,
                  legend_handles, legend_title,
                  title, x_label, y_label, n_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# HRS — per-person means, coloured by number of waves observed
# ──────────────────────────────────────────────────────────────────────────────

print("Loading HRS ...")
hrs_raw = pd.read_csv(ROOT / "output" / "hrs_rand_pooled_output.csv")
hrs_raw = hrs_raw[hrs_raw["DepressionScore"].notna() & hrs_raw["SleepQualityScore"].notna()]

per_person = (
    hrs_raw.groupby("HHID_PN")
    .agg(mean_dep=("DepressionScore",  "mean"),
         mean_slp=("SleepQualityScore", "mean"),
         n_waves  =("Wave",             "count"))
    .dropna(subset=["mean_dep", "mean_slp"])
    .reset_index()
)
print(f"  HRS individuals: {len(per_person):,}")

WAVE_COLORS = {1: "#d9d9d9", 2: "#bdbdbd", 3: "#969696",
               4: "#636363", 5: "#252525", 6: V.CAT_PALETTE[1]}
point_colors_hrs = [WAVE_COLORS.get(int(n), "#252525")
                    for n in per_person["n_waves"]]

wave_patches = [mpatches.Patch(color=WAVE_COLORS[k], alpha=0.8,
                               label=f"{k} wave{'s' if k > 1 else ''}")
                for k in sorted(WAVE_COLORS)]

n_indiv = len(per_person)
hrs_kwargs = dict(
    sleep_vals    = per_person["mean_slp"].values,
    dep_vals      = per_person["mean_dep"].values,
    point_colors  = point_colors_hrs,
    legend_handles= wave_patches,
    legend_title  = "Waves observed",
    title         = (f"Sleep Quality vs Depression — per-person means\n"
                     f"HRS pooled 50+, 2010–2022  |  n = {n_indiv:,}"),
    x_label       = "Mean Sleep Quality Score (%)",
    y_label       = "Mean Depression Score (%)",
    n_label       = f"{n_indiv:,} individuals",
)

sleep_dep_scatter(**hrs_kwargs, out_path=OUT / "HRS_sleep_dep_scatter.png")

# ──────────────────────────────────────────────────────────────────────────────
# NHANES — cross-sectional, coloured by cycle
# ──────────────────────────────────────────────────────────────────────────────

print("Loading NHANES ...")
nh = pd.read_csv(ROOT / "output" / "nhanes_pooled_output.csv")
nh = nh[nh["DepressionScore"].notna() & nh["SleepQualityScore"].notna()].copy()
print(f"  NHANES participants: {len(nh):,}")

CYCLE_COLORS = {
    "2013-2014": "#5B9BD5",
    "2015-2016": "#E07828",
    "2017-2018": "#2DA84B",
}
present_cycles  = [c for c in ["2013-2014", "2015-2016", "2017-2018"]
                   if c in nh["Cycle"].values]
point_colors_nh = [CYCLE_COLORS[c] for c in nh["Cycle"]]

cycle_patches = [mpatches.Patch(color=CYCLE_COLORS[c], alpha=0.8, label=c)
                 for c in present_cycles]

n_nh = len(nh)
nh_kwargs = dict(
    sleep_vals    = nh["SleepQualityScore"].values,
    dep_vals      = nh["DepressionScore"].values,
    point_colors  = point_colors_nh,
    legend_handles= cycle_patches,
    legend_title  = "NHANES cycle",
    title         = f"Sleep Quality vs Depression\nNHANES pooled 50+  |  n = {n_nh:,}",
    x_label       = "Sleep Quality Score (%)",
    y_label       = "Depression Score (%)",
    n_label       = f"{n_nh:,}",
)

sleep_dep_scatter(**nh_kwargs, out_path=OUT / "NHANES_sleep_dep_scatter.png")

# ──────────────────────────────────────────────────────────────────────────────
# Combined side-by-side figure
# ──────────────────────────────────────────────────────────────────────────────

print("Building combined figure ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

_draw_scatter(axes[0], panel_letter="a", **hrs_kwargs)
_draw_scatter(axes[1], panel_letter="b", **nh_kwargs)

fig.tight_layout()
combined_path = OUT / "combined_sleep_dep_scatter.png"
fig.savefig(combined_path, dpi=FIG_DPI, bbox_inches="tight")
if _EDIT:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parents[2]))
    from tools.figure_editor import launch
    launch(fig)
plt.close(fig)
print(f"  Saved -> {combined_path}")

print("\nDone.")
