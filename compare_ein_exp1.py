import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

EIN_PATH = "/Volumes/Expansion/features and curves 141026/features.parquet"
EXP1_PATH = "/Volumes/My Book/exp_1/sw/feeding_features/features.parquet"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ID_COLUMNS = {"channel", "feeding_datetime", "date", "feeding_window"}

# Paper name for each code column written by extract_mountain_features.
CODE_TO_NAME = {
    "baseline": "Baseline",
    "peak_height": "Aqua-MAFRI",
    "plateau_rise_gradient": "Plateau-rise gradient",
    "plateau_fall_gradient": "Plateau-fall gradient",
    "rise_fall_gradient_ratio": "Rise-to-fall gradient ratio",
    "rise_AUC": "Rise AUC",
    "fall_AUC": "Fall AUC",
    "rise_fall_AUC_ratio": "Rise-to-fall AUC ratio",
}

FEATURE_ORDER = list(CODE_TO_NAME.values())


def feature_frame(path):
    """Numeric features only, with paper names as columns."""
    raw = pd.read_parquet(path)
    renamed = {}
    unknown = []
    for col in raw.columns:
        if col in ID_COLUMNS:
            continue
        name = CODE_TO_NAME.get(col, col if col in CODE_TO_NAME.values() else None)
        if name is None:
            unknown.append(col)
            continue
        renamed[name] = pd.to_numeric(raw[col], errors="coerce")
    frame = pd.DataFrame(renamed)
    ordered = [name for name in FEATURE_ORDER if name in frame.columns]
    extra = [name for name in frame.columns if name not in ordered]
    times = pd.to_datetime(raw["feeding_datetime"]).reset_index(drop=True)
    return frame[ordered + extra], times, unknown


def finite_values(series):
    values = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def shared_bins(left, right):
    pooled = np.concatenate([left, right])
    if pooled.size == 0:
        return np.array([0.0, 1.0])
    lo = float(np.min(pooled))
    hi = float(np.max(pooled))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([lo - 0.5, hi + 0.5])
    # One far outlier otherwise empties the bars. Keep the bulk of both samples.
    if pooled.size >= 20:
        p_lo, p_hi = np.percentile(pooled, [1.0, 99.0])
        if p_hi > p_lo and (hi - lo) > 8.0 * (p_hi - p_lo):
            lo, hi = float(p_lo), float(p_hi)
    n_bins = int(np.clip(np.sqrt(pooled.size), 12, 30))
    return np.linspace(lo, hi, n_bins + 1)


PANEL_FEATURES = [
    ("(a)", "Aqua-MAFRI"),
    ("(b)", "Baseline"),
    ("(c)", "Rise AUC"),
    ("(d)", "Fall AUC"),
]
RAS_COLOR = "#3D2BFF"
PREVIOUS_COLOR = "#FF6A00"
FONT_SIZE = 13
# Rise AUC already ends below this. Fall AUC outliers above it are left off
# so panels (c) and (d) share the same right edge.
AUC_XMAX = 50.0


def add_kde(ax, values, color):
    if values.size < 2 or float(np.std(values)) == 0.0:
        return
    kde = gaussian_kde(values)
    xs = np.linspace(float(values.min()), float(values.max()), 256)
    ax.plot(xs, kde(xs), color=color, linewidth=2.2, zorder=3)


def hedges_g(left, right):
    """Bias-corrected standardized mean difference, RAS minus previous."""
    na, nb = len(left), len(right)
    mean_diff = float(left.mean() - right.mean())
    pooled = np.sqrt(
        ((na - 1) * left.var(ddof=1) + (nb - 1) * right.var(ddof=1)) / (na + nb - 2)
    )
    if pooled == 0 or not np.isfinite(pooled):
        return mean_diff, np.nan
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return mean_diff, mean_diff / pooled * correction


def welch_ci(left, right):
    """95% CI for the mean difference, RAS minus previous."""
    na, nb = len(left), len(right)
    mean_diff = float(left.mean() - right.mean())
    va, vb = float(left.var(ddof=1)), float(right.var(ddof=1))
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    critical = float(stats.t.ppf(0.975, df))
    return mean_diff - critical * se, mean_diff + critical * se, df


def cliffs_delta(left, right, mann_whitney_u):
    """Positive when RAS values tend to be larger than the previous setup."""
    return (2.0 * float(mann_whitney_u) / (len(left) * len(right))) - 1.0


def hodges_lehmann(left, right):
    """Median of all pairwise differences, RAS minus previous."""
    return float(np.median(left[:, None] - right[None, :]))


def star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def magnitude_g(value):
    absolute = abs(value)
    if not np.isfinite(absolute):
        return "undefined"
    if absolute < 0.2:
        return "negligible"
    if absolute < 0.5:
        return "small"
    if absolute < 0.8:
        return "medium"
    return "large"


def magnitude_delta(value):
    """Romano cutoffs for Cliff's delta."""
    absolute = abs(value)
    if not np.isfinite(absolute):
        return "undefined"
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.33:
        return "small"
    if absolute < 0.474:
        return "medium"
    return "large"


def summarize(values):
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "iqr": (float(q1), float(q3)),
    }


def report_panel_tests(ein, exp1, label):
    """Two-sample tests for the four plotted features. Groups are independent."""
    names = [name for _, name in PANEL_FEATURES]
    rows = []
    for name in names:
        left = finite_values(ein[name])
        right = finite_values(exp1[name])
        ras = summarize(left)
        previous = summarize(right)
        mean_diff, g = hedges_g(left, right)
        ci_low, ci_high, welch_df = welch_ci(left, right)
        welch = stats.ttest_ind(left, right, equal_var=False)
        mann = stats.mannwhitneyu(left, right, alternative="two-sided")
        ks = stats.ks_2samp(left, right, alternative="two-sided", method="auto")
        levene = stats.levene(left, right, center="median")
        delta = cliffs_delta(left, right, mann.statistic)
        shift = hodges_lehmann(left, right)
        shapiro_ras = stats.shapiro(left) if left.size <= 5000 else None
        shapiro_prev = stats.shapiro(right) if right.size <= 5000 else None
        rows.append(
            {
                "name": name,
                "ras": ras,
                "previous": previous,
                "mean_diff": mean_diff,
                "ci": (ci_low, ci_high),
                "welch_df": welch_df,
                "welch_p": float(welch.pvalue),
                "g": g,
                "mann_u": float(mann.statistic),
                "mann_p": float(mann.pvalue),
                "delta": delta,
                "shift": shift,
                "ks_d": float(ks.statistic),
                "ks_p": float(ks.pvalue),
                "levene_p": float(levene.pvalue),
                "shapiro_ras_p": None if shapiro_ras is None else float(shapiro_ras.pvalue),
                "shapiro_prev_p": None if shapiro_prev is None else float(shapiro_prev.pvalue),
            }
        )

    n_tests = len(rows)
    print("\n" + "=" * 72)
    print(f"RAS vs Previous Experimental Setup — {label}")
    print("RAS is Ein. Previous Experimental Setup is Exp1.")
    print("Difference is RAS minus previous. Positive means RAS is higher.")
    print(f"Bonferroni across {n_tests} features (alpha 0.05 / {n_tests} = {0.05 / n_tests:.4f}).")
    print("=" * 72)

    for row in rows:
        ras, previous = row["ras"], row["previous"]
        mann_adj = min(1.0, row["mann_p"] * n_tests)
        welch_adj = min(1.0, row["welch_p"] * n_tests)
        ks_adj = min(1.0, row["ks_p"] * n_tests)
        direction = "higher" if row["mean_diff"] > 0 else "lower"
        print(f"\n{row['name']}")
        print(
            f"  RAS        n={ras['n']:<4}  mean={ras['mean']:.4g}  sd={ras['sd']:.4g}  "
            f"median={ras['median']:.4g}  IQR={ras['iqr'][0]:.4g}–{ras['iqr'][1]:.4g}"
        )
        print(
            f"  Previous   n={previous['n']:<4}  mean={previous['mean']:.4g}  sd={previous['sd']:.4g}  "
            f"median={previous['median']:.4g}  IQR={previous['iqr'][0]:.4g}–{previous['iqr'][1]:.4g}"
        )
        print(
            f"  Mean difference {row['mean_diff']:.4g}  "
            f"(RAS is {direction}); 95% CI [{row['ci'][0]:.4g}, {row['ci'][1]:.4g}]"
        )
        print(
            f"  Median pairwise difference (Hodges–Lehmann) {row['shift']:.4g}"
        )
        print(
            f"  Hedges' g {row['g']:.3f} ({magnitude_g(row['g'])})   "
            f"Cliff's delta {row['delta']:.3f} ({magnitude_delta(row['delta'])})"
        )
        print(
            f"  Welch t  df={row['welch_df']:.1f}  p={row['welch_p']:.3e}  "
            f"Bonferroni p={welch_adj:.3e}  {star(welch_adj)}"
        )
        print(
            f"  Mann–Whitney U={row['mann_u']:.4g}  p={row['mann_p']:.3e}  "
            f"Bonferroni p={mann_adj:.3e}  {star(mann_adj)}"
        )
        print(
            f"  Kolmogorov–Smirnov D={row['ks_d']:.3f}  p={row['ks_p']:.3e}  "
            f"Bonferroni p={ks_adj:.3e}  {star(ks_adj)}"
        )
        print(f"  Levene (variance) p={row['levene_p']:.3e}")
        print(
            f"  Shapiro–Wilk normality  RAS p={row['shapiro_ras_p']:.3e}  "
            f"Previous p={row['shapiro_prev_p']:.3e}"
        )

    print("\nStars use the Bonferroni p: * < 0.05, ** < 0.01, *** < 0.001.")
    print(
        "Mann–Whitney is the main test: the histograms are skewed, "
        "and Shapiro–Wilk rejects a normal shape."
    )
    print(
        "Windows from the same channel are not independent, "
        "so these p-values overstate the evidence."
    )


def plot_four(ein, exp1):
    """Aqua-MAFRI, baseline, fall AUC, and rise AUC, each with both histograms overlaid."""
    # Same size as the tank comparison, which reads clearly on a 2x2 figure.
    panel_font = 18
    labels = {
        "Aqua-MAFRI": "Normalized Aqua-MAFRI",
        "Baseline": "Normalized Baseline",
        "Fall AUC": "Normalized Fall AUC",
        "Rise AUC": "Normalized Rise AUC",
    }
    plt.rcParams.update(
        {
            "font.size": panel_font,
            "axes.labelsize": panel_font,
            "xtick.labelsize": panel_font,
            "ytick.labelsize": panel_font,
            "legend.fontsize": panel_font,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
    for ax, (letter, name) in zip(axs.ravel(), PANEL_FEATURES):
        left = finite_values(ein[name])
        right = finite_values(exp1[name])
        if name == "Fall AUC":
            left = left[left <= AUC_XMAX]
            right = right[right <= AUC_XMAX]
        bins = shared_bins(left, right)
        for values, color, series_name in (
            (left, RAS_COLOR, "RAS"),
            (right, PREVIOUS_COLOR, "Previous experiment"),
        ):
            inside = values[(values >= bins[0]) & (values <= bins[-1])]
            ax.hist(
                inside,
                bins=bins,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                density=True,
                alpha=0.5,
                label=series_name,
            )
            add_kde(ax, inside, color)
        ax.set_xlabel(labels.get(name, name))
        ax.set_ylabel("Density")
        if name in ("Rise AUC", "Fall AUC"):
            ax.set_xlim(0, AUC_XMAX)
        ax.text(
            0.03,
            0.97,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=panel_font,
            fontweight="bold",
        )
    axs[0, 1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


TANK_CHANNELS = {
    "Tank 1": ("Ch1", "Ch2"),
    "Tank 2": ("Ch3", "Ch4"),
    "Tank 3": ("Ch5", "Ch6"),
}
# One sensor from each tank: the even channels.
CHANNEL_246 = {
    "Ch2": ("Ch2",),
    "Ch4": ("Ch4",),
    "Ch6": ("Ch6",),
}
TANK_COLORS = ["#54278F", "#E31A1C", "#1B9E77"]
TANK_PANELS = [
    ("(a)", "peak_height", "Normalized Aqua-MAFRI"),
    ("(b)", "baseline", "Normalized Baseline"),
    ("(c)", "rise_AUC", "Normalized Rise AUC"),
    ("(d)", "fall_AUC", "Normalized Fall AUC"),
]


def plot_tanks(raw, groups=None):
    """Same four features as the RAS comparison, one histogram per group."""
    if groups is None:
        groups = TANK_CHANNELS
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    rise_parts = []
    for channels in groups.values():
        rise = finite_values(
            pd.to_numeric(raw.loc[raw["channel"].isin(channels), "rise_AUC"], errors="coerce")
        )
        if rise.size:
            rise_parts.append(rise)
    rise_max = float(np.max(np.concatenate(rise_parts)))
    # Shared right edge for panels (c) and (d): far enough for every rise value.
    auc_xmax = float(np.ceil(rise_max / 5.0) * 5.0)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
    for ax, (letter, column, label) in zip(axs.ravel(), TANK_PANELS):
        series = []
        for name, channels in groups.items():
            values = pd.to_numeric(
                raw.loc[raw["channel"].isin(channels), column],
                errors="coerce",
            )
            values = finite_values(values)
            if column == "fall_AUC":
                values = values[values <= auc_xmax]
            series.append((name, values))
        bins = shared_bins(series[0][1], np.concatenate([values for _, values in series[1:]]))
        for (name, values), color in zip(series, TANK_COLORS):
            inside = values[(values >= bins[0]) & (values <= bins[-1])]
            ax.hist(
                inside,
                bins=bins,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                density=True,
                alpha=0.45,
                label=name,
            )
            add_kde(ax, inside, color)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        if column in ("rise_AUC", "fall_AUC"):
            ax.set_xlim(0, auc_xmax)
        ax.text(
            0.03,
            0.97,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )
    axs[0, 1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    ein, ein_times, ein_unknown = feature_frame(EIN_PATH)
    exp1, exp1_times, exp1_unknown = feature_frame(EXP1_PATH)
    ein_names = list(ein.columns)
    exp1_names = list(exp1.columns)
    shared = [name for name in FEATURE_ORDER if name in ein.columns and name in exp1.columns]
    only_ein = [name for name in ein_names if name not in exp1.columns]
    only_exp1 = [name for name in exp1_names if name not in ein.columns]

    print(f"Ein rows:  {len(ein)}  features: {ein_names}")
    print(f"Exp1 rows: {len(exp1)}  features: {exp1_names}")
    if ein_unknown or exp1_unknown:
        print(f"Unmapped columns — Ein: {ein_unknown}  Exp1: {exp1_unknown}")
    if ein_names != exp1_names:
        print("Feature columns do not match.")
        print(f"  Only in Ein:  {only_ein or '(none)'}")
        print(f"  Only in Exp1: {only_exp1 or '(none)'}")
        raise SystemExit("Both tables must contain the same features.")

    print("Feature columns match.")

    missing = [name for _, name in PANEL_FEATURES if name not in shared]
    if missing:
        raise SystemExit(f"Missing panel features: {missing}")

    label = "all feedings"
    ein_days = int(ein_times.dt.normalize().nunique())
    exp_days = int(exp1_times.dt.normalize().nunique())
    print(
        f"\n{label}: RAS rows {len(ein)} ({ein_days} days), "
        f"previous rows {len(exp1)} ({exp_days} days)"
    )
    report_panel_tests(ein, exp1, label)
    fig = plot_four(ein, exp1)
    path = os.path.join(OUT_DIR, "compare_four.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show()


def save_tank_comparison():
    raw = pd.read_parquet(EIN_PATH, columns=["channel", *{col for _, col, _ in TANK_PANELS}])
    fig = plot_tanks(raw)
    path = os.path.join(OUT_DIR, "compare_tanks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    return path


def save_channel_246():
    """Four-feature histograms using only Ch2, Ch4, and Ch6."""
    columns = ["channel", *{col for _, col, _ in TANK_PANELS}]
    raw = pd.read_parquet(EIN_PATH, columns=columns)
    raw = raw.loc[raw["channel"].isin(["Ch2", "Ch4", "Ch6"])].copy()
    for name, channels in CHANNEL_246.items():
        n = int(raw["channel"].isin(channels).sum())
        print(f"{name}: {n} feedings")
    fig = plot_tanks(raw, CHANNEL_246)
    path = os.path.join(OUT_DIR, "compare_ch246.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    main()
