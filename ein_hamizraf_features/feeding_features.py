import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

BASE = r"H:\Exp_Ein_Hamifratz\output_111026"
SIGNAL_PATH = BASE + r"\combined_0.001.parquet"
FEED_PATH = BASE + r"\feeding_times_ch5.parquet"
OUT_DIR = r"H:\Exp_Ein_Hamifratz\feeding_features"
CHANNELS = [f"Ch{i}" for i in range(1, 7)]
N_PANELS = 30
NCOLS = 6
N_CURVE_PNGS = 4

CORR_HIGHLIGHT = 0.80
ALPHA = 0.05
MINUTES_PER_DAY = 24 * 60

# Appendix remaining features only (paper name, group, code column).
APPENDIX_FEATURES = [
    ("Response onset", "Temporal", "mountain_start"),
    ("Response duration", "Temporal", "mountain_duration"),
    ("FWHM", "Temporal", "FWHM"),
    ("Plateau-rise duration", "Temporal", "plateau_rise_duration"),
    ("Baseline", "Magnitude", "baseline"),
    ("Peak height", "Magnitude", "peak_height"),
    ("Plateau-rise change", "Magnitude", "plateau_rise_change"),
    ("Plateau-fall change", "Magnitude", "plateau_fall_change"),
    ("Plateau gradient", "Gradient", "plateau_gradient"),
    ("Plateau-gradient R2", "Gradient", "plateau_gradient_R2"),
    ("Plateau-rise gradient", "Gradient", "plateau_rise_gradient"),
    ("Plateau-rise-gradient R2", "Gradient", "plateau_rise_gradient_R2"),
    ("Maximum plateau-rise gradient", "Gradient", "max_plateau_rise_gradient"),
    ("Plateau-fall-gradient R2", "Gradient", "plateau_fall_gradient_R2"),
    ("Rise AUC", "Area", "rise_AUC"),
    ("Plateau-rise AUC", "Area", "plateau_rise_AUC"),
    ("Fall AUC fraction", "Area", "fall_AUC_fraction"),
    ("Plateau standard deviation", "Variability", "plateau_std"),
    ("Symmetry ratio", "Variability", "symmetry_ratio"),
    ("Rise-to-fall gradient ratio", "Variability", "rise_fall_gradient_ratio"),
    ("Feeding sine", "Feeding time", "feeding_sin"),
    ("Feeding cosine", "Feeding time", "feeding_cos"),
]


def to_datetime(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S")


def minmax_channel(values):
    """Scale one channel to 0-1 using its own full-series min and max."""
    values = np.asarray(values, dtype=float)
    ymin = np.nanmin(values)
    ymax = np.nanmax(values)
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return np.zeros_like(values, dtype=float)
    return (values - ymin) / (ymax - ymin)


def preview_positions(n, n_panels=N_PANELS):
    return np.unique(np.round(np.linspace(0, n - 1, n_panels)).astype(int))


def slice_window(times, values, ft):
    ft64 = np.datetime64(pd.Timestamp(ft), "ns")
    start = ft64 - np.timedelta64(30, "m")
    end = ft64 + np.timedelta64(30, "m")
    lo = np.searchsorted(times, start, side="left")
    hi = np.searchsorted(times, end, side="right")
    return times[lo:hi], values[lo:hi]


def window_frame(times, values, ft):
    window_t, window_y = slice_window(times, values, ft)
    return pd.DataFrame(
        {"value": window_y, "time": pd.to_datetime(window_t)}
    )


def extract_mountain_features(
    df_window,
    value_col="Ch1",
    time_col="time",
    plateau_threshold=0.90,
    slope_fraction=0.0,
    toe_fraction=0.05,
    ax=None,
    show_stage_markers=True,
    show_guide_lines=True,
):
    y = df_window[value_col].to_numpy(dtype=float)
    y = pd.Series(y).interpolate().bfill().ffill().to_numpy()
    idx = np.arange(len(y))
    x = pd.to_datetime(df_window[time_col]).to_numpy()

    if len(y) < 5:
        return {}

    y_smooth = gaussian_filter1d(y, sigma=2)

    # Baseline
    n_base = max(3, int(0.05 * len(y)))
    baseline = min(
        np.mean(y_smooth[:n_base]),
        np.mean(y_smooth[-n_base:])
    )

    # Peak
    peak_index = int(np.argmax(y_smooth))
    peak_height = float(y[peak_index])
    amplitude = peak_height - baseline

    if amplitude <= 0:
        return {}

    half_level = baseline + 0.50 * amplitude
    plateau_level = baseline + plateau_threshold * amplitude
    toe_level = baseline + toe_fraction * amplitude

    # Plateau first (anchors the mountain core)
    rise_90 = np.where(y_smooth[: peak_index + 1] >= plateau_level)[0]
    plateau_start = int(rise_90[0]) if len(rise_90) else peak_index

    fall_90 = np.where(y_smooth[peak_index:] < plateau_level)[0]
    plateau_end = (
        peak_index + int(fall_90[0])
        if len(fall_90)
        else len(y) - 1
    )

    # Flanks: expand from dominant rise/fall slopes while slope keeps
    # the same sign, but stop at the mountain toe so baseline tails
    # are not included.
    dy = gaussian_filter1d(np.gradient(y_smooth), sigma=1.5)

    i_rise = int(np.argmax(dy[: peak_index + 1]))
    i_fall = peak_index + int(np.argmin(dy[peak_index:]))

    rise_eps = slope_fraction * dy[i_rise] if dy[i_rise] > 0 else 0.0
    fall_eps = slope_fraction * dy[i_fall] if dy[i_fall] < 0 else 0.0

    mountain_start = i_rise
    while mountain_start > 0 and dy[mountain_start - 1] > rise_eps:
        mountain_start -= 1
        if y_smooth[mountain_start] <= toe_level:
            break

    mountain_end = i_fall
    while mountain_end < len(y) - 1 and dy[mountain_end] < fall_eps:
        mountain_end += 1
        if y_smooth[mountain_end] <= toe_level:
            break

    mountain_start = min(mountain_start, plateau_start)
    mountain_end = max(mountain_end, plateau_end)

    # FWHM
    rise_50 = np.where(
        y_smooth[mountain_start:peak_index + 1] >= half_level
    )[0]

    fall_50 = np.where(
        y_smooth[peak_index:mountain_end + 1] < half_level
    )[0]

    half_start = (
        mountain_start + int(rise_50[0])
        if len(rise_50)
        else np.nan
    )

    half_end = (
        peak_index + int(fall_50[0])
        if len(fall_50)
        else np.nan
    )

    fwhm = (
        half_end - half_start
        if np.isfinite(half_start) and np.isfinite(half_end)
        else np.nan
    )

    def fit_gradient(start, end):
        if end <= start:
            return np.nan, np.nan

        section_x = idx[start:end + 1]
        section_y = y[start:end + 1]

        slope, intercept = np.polyfit(section_x, section_y, 1)
        fitted = slope * section_x + intercept

        ss_res = np.sum((section_y - fitted) ** 2)
        ss_tot = np.sum((section_y - np.mean(section_y)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return slope, r2

    # Gradients
    rise_gradient, rise_r2 = fit_gradient(
        mountain_start, plateau_start
    )

    plateau_gradient, plateau_r2 = fit_gradient(
        plateau_start, plateau_end
    )

    plateau_rise_gradient, plateau_rise_r2 = fit_gradient(
        plateau_start, peak_index
    )

    plateau_fall_gradient, plateau_fall_r2 = fit_gradient(
        peak_index, plateau_end
    )

    fall_gradient, fall_r2 = fit_gradient(
        plateau_end, mountain_end
    )

    # Sections
    mountain_y = y[mountain_start:mountain_end + 1]
    rise_y = y[mountain_start:plateau_start + 1]
    plateau_y = y[plateau_start:plateau_end + 1]
    plateau_rise_y = y[plateau_start:peak_index + 1]
    plateau_fall_y = y[peak_index:plateau_end + 1]
    fall_y = y[plateau_end:mountain_end + 1]

    def corrected_auc(values):
        return np.trapezoid(np.maximum(values - baseline, 0))

    # AUC
    mountain_auc = corrected_auc(mountain_y)
    rise_auc = corrected_auc(rise_y)
    plateau_auc = corrected_auc(plateau_y)
    plateau_rise_auc = corrected_auc(plateau_rise_y)
    plateau_fall_auc = corrected_auc(plateau_fall_y)
    fall_auc = corrected_auc(fall_y)

    # Durations
    mountain_duration = mountain_end - mountain_start
    rise_duration = plateau_start - mountain_start
    plateau_duration = plateau_end - plateau_start
    plateau_rise_duration = peak_index - plateau_start
    plateau_fall_duration = plateau_end - peak_index
    fall_duration = mountain_end - plateau_end

    # Instantaneous gradients
    dy = np.diff(y)

    rise_dy = dy[mountain_start:plateau_start]
    plateau_dy = dy[plateau_start:plateau_end]
    plateau_rise_dy = dy[plateau_start:peak_index]
    plateau_fall_dy = dy[peak_index:plateau_end]
    fall_dy = dy[plateau_end:mountain_end]

    max_rise_gradient = (
        np.max(rise_dy) if len(rise_dy) else np.nan
    )

    max_fall_gradient = (
        np.min(fall_dy) if len(fall_dy) else np.nan
    )

    max_plateau_rise_gradient = (
        np.max(plateau_rise_dy)
        if len(plateau_rise_dy)
        else np.nan
    )

    max_plateau_fall_gradient = (
        np.min(plateau_fall_dy)
        if len(plateau_fall_dy)
        else np.nan
    )

    mean_absolute_plateau_gradient = (
        np.mean(np.abs(plateau_dy))
        if len(plateau_dy)
        else np.nan
    )

    # Plateau statistics
    plateau_mean = np.mean(plateau_y)
    plateau_std = np.std(plateau_y)
    plateau_range = np.ptp(plateau_y)

    plateau_cv = (
        plateau_std / plateau_mean
        if plateau_mean != 0
        else np.nan
    )

    plateau_rise_change = peak_height - y[plateau_start]
    plateau_fall_change = y[plateau_end] - peak_height
    plateau_total_change = y[plateau_end] - y[plateau_start]

    # Ratios
    symmetry_ratio = (
        rise_duration / fall_duration
        if fall_duration > 0
        else np.nan
    )

    rise_fall_gradient_ratio = (
        abs(rise_gradient / fall_gradient)
        if np.isfinite(fall_gradient) and fall_gradient != 0
        else np.nan
    )

    plateau_fraction = (
        plateau_duration / mountain_duration
        if mountain_duration > 0
        else np.nan
    )

    def auc_fraction(value):
        return value / mountain_auc if mountain_auc > 0 else np.nan

    features = {
        # Mountain
        "mountain_start": mountain_start,
        "mountain_end": mountain_end,
        "mountain_duration": mountain_duration,
        "baseline": baseline,
        "peak_index": peak_index,
        "peak_height": peak_height,
        "response_amplitude": amplitude,
        "FWHM": fwhm,
        "mountain_AUC": mountain_auc,
        "mean_mountain_height": (
            mountain_auc / mountain_duration
            if mountain_duration > 0
            else np.nan
        ),

        # Rise
        "rise_duration": rise_duration,
        "rise_gradient": rise_gradient,
        "rise_gradient_R2": rise_r2,
        "max_rise_gradient": max_rise_gradient,
        "rise_AUC": rise_auc,
        "rise_AUC_fraction": auc_fraction(rise_auc),

        # Plateau overview
        "plateau_start": plateau_start,
        "plateau_end": plateau_end,
        "plateau_duration": plateau_duration,
        "plateau_gradient": plateau_gradient,
        "plateau_gradient_R2": plateau_r2,
        "plateau_mean": plateau_mean,
        "plateau_std": plateau_std,
        "plateau_range": plateau_range,
        "plateau_cv": plateau_cv,
        "plateau_total_change": plateau_total_change,
        "mean_absolute_plateau_gradient":
            mean_absolute_plateau_gradient,
        "plateau_AUC": plateau_auc,
        "plateau_AUC_fraction": auc_fraction(plateau_auc),
        "plateau_fraction": plateau_fraction,

        # Plateau rise
        "plateau_rise_duration": plateau_rise_duration,
        "plateau_rise_change": plateau_rise_change,
        "plateau_rise_gradient": plateau_rise_gradient,
        "plateau_rise_gradient_R2": plateau_rise_r2,
        "max_plateau_rise_gradient":
            max_plateau_rise_gradient,
        "plateau_rise_AUC": plateau_rise_auc,

        # Plateau fall
        "plateau_fall_duration": plateau_fall_duration,
        "plateau_fall_change": plateau_fall_change,
        "plateau_fall_gradient": plateau_fall_gradient,
        "plateau_fall_gradient_R2": plateau_fall_r2,
        "max_plateau_fall_gradient":
            max_plateau_fall_gradient,
        "plateau_fall_AUC": plateau_fall_auc,

        # Fall
        "fall_duration": fall_duration,
        "fall_gradient": fall_gradient,
        "fall_gradient_R2": fall_r2,
        "max_fall_gradient": max_fall_gradient,
        "fall_AUC": fall_auc,
        "fall_AUC_fraction": auc_fraction(fall_auc),

        # Shape ratios
        "symmetry_ratio": symmetry_ratio,
        "rise_fall_gradient_ratio":
            rise_fall_gradient_ratio
    }

    # Plot
    if ax is not None:
        minutes = (
            pd.to_datetime(x) - pd.to_datetime(x[0])
        ).total_seconds() / 60

        ax.plot(
            minutes,
            y,
            alpha=0.25,
            label="Full signal"
        )

        ax.plot(
            minutes[mountain_start:plateau_start + 1],
            rise_y,
            linewidth=2.5,
            label="Rise"
        )

        ax.plot(
            minutes[plateau_start:peak_index + 1],
            plateau_rise_y,
            linewidth=2.5,
            label="Plateau rise"
        )

        ax.plot(
            minutes[peak_index:plateau_end + 1],
            plateau_fall_y,
            linewidth=2.5,
            label="Plateau fall"
        )

        ax.plot(
            minutes[plateau_end:mountain_end + 1],
            fall_y,
            linewidth=2.5,
            label="Fall"
        )

        ax.fill_between(
            minutes[mountain_start:mountain_end + 1],
            baseline,
            mountain_y,
            alpha=0.18,
            label="AUC"
        )

        ax.axhline(
            baseline,
            linestyle="--",
            alpha=0.7,
            label="Baseline"
        )

        if show_guide_lines:
            ax.axhline(
                plateau_level,
                linestyle=":",
                alpha=0.5,
                label="90% level"
            )

        if show_stage_markers:
            stage_points = [
                ("Mountain start", mountain_start, "v"),
                ("Plateau start", plateau_start, "o"),
                ("Peak", peak_index, "^"),
                ("Plateau end", plateau_end, "s"),
                ("Mountain end", mountain_end, "X")
            ]

            for label, position, marker in stage_points:
                ax.scatter(
                    minutes[position],
                    y[position],
                    marker=marker,
                    s=45,
                    zorder=6,
                    label=label
                )

        if show_guide_lines:
            for position in [
                mountain_start,
                plateau_start,
                peak_index,
                plateau_end,
                mountain_end
            ]:
                ax.axvline(
                    minutes[position],
                    linestyle=":",
                    alpha=0.45
                )

    return features


def remaining_feature_frame(features_df):
    """Appendix remaining features only, using paper names as columns."""
    out = features_df.copy()
    if "feeding_datetime" in out.columns:
        ts = pd.to_datetime(out["feeding_datetime"])
        minutes = ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60.0
        angle = 2.0 * np.pi * minutes / MINUTES_PER_DAY
        out["feeding_sin"] = np.sin(angle)
        out["feeding_cos"] = np.cos(angle)

    wanted = [col for _, _, col in APPENDIX_FEATURES]
    missing_cols = [c for c in wanted if c not in out.columns]
    present = [c for c in wanted if c in out.columns]
    remaining = out[present].apply(pd.to_numeric, errors="coerce")
    rename = {
        col: name
        for name, _, col in APPENDIX_FEATURES
        if col in remaining.columns
    }
    remaining = remaining.rename(columns=rename)
    ordered = [name for name, _, col in APPENDIX_FEATURES if col in present]
    return remaining[ordered], missing_cols


def report_remaining_features(
    features_df,
    n_no_data=0,
    n_extract_fail=0,
    n_expected=None,
):
    """Print paper-ready stats for appendix remaining features only."""
    remaining, missing_cols = remaining_feature_frame(features_df)
    n_rows = len(features_df)
    n_feat = remaining.shape[1]
    n_expected = n_rows if n_expected is None else n_expected
    name_to_group = {name: group for name, group, _ in APPENDIX_FEATURES}
    name_to_code = {name: col for name, _, col in APPENDIX_FEATURES}

    print("\n" + "=" * 72)
    print("PAPER SUMMARY — remaining features only (appendix table)")
    print("=" * 72)

    print("\n[1] Dataset coverage")
    print(f"  Expected feeding windows: {n_expected}")
    print(f"  Extracted windows (rows): {n_rows}")
    print(f"  Empty / no-data windows:  {n_no_data}")
    print(f"  Failed mountain extract:  {n_extract_fail}")
    if n_expected:
        print(
            f"  Extraction success:       {n_rows}/{n_expected} "
            f"({100.0 * n_rows / n_expected:.1f}%)"
        )
    if "date" in features_df.columns and n_rows:
        dates = pd.to_datetime(features_df["date"])
        print(f"  Unique dates:             {dates.nunique()}")
        print(f"  Date range:               {dates.min().date()} to {dates.max().date()}")
    if "feeding_window" in features_df.columns and n_rows:
        print("  Rows per feeding window:")
        counts = features_df["feeding_window"].value_counts().sort_index()
        for label, count in counts.items():
            print(f"    {label}: {count}")
    print(f"  Remaining features:       {n_feat}/{len(APPENDIX_FEATURES)}")
    if missing_cols:
        print(f"  Appendix columns not found: {missing_cols}")
    print("  Feature list (group | paper name | code):")
    for name, group, col in APPENDIX_FEATURES:
        status = "ok" if name in remaining.columns else "MISSING"
        print(f"    [{status}] {group:<14} {name:<32} {col}")

    if n_rows == 0 or n_feat == 0:
        print("\nNo remaining features to summarize.")
        return

    remaining_finite = remaining.replace([np.inf, -np.inf], np.nan)

    print("\n[2] Missing values (remaining features only)")
    n_cells = n_rows * n_feat
    na_count = remaining.isna().sum()
    inf_count = remaining.apply(
        lambda s: int(np.isinf(pd.to_numeric(s, errors="coerce")).sum())
    )
    n_valid = remaining_finite.notna().sum()
    missing_tbl = pd.DataFrame(
        {
            "group": [name_to_group.get(c, "") for c in remaining.columns],
            "code": [name_to_code.get(c, "") for c in remaining.columns],
            "n_missing": na_count,
            "pct_missing": 100.0 * na_count / n_rows,
            "n_inf": inf_count,
            "n_valid": n_valid,
            "pct_valid": 100.0 * n_valid / n_rows,
        },
        index=remaining.columns,
    ).sort_values(["n_missing", "n_inf", "n_valid"], ascending=[False, False, True])

    n_missing_cells = int(na_count.sum())
    n_inf_cells = int(inf_count.sum())
    n_feat_any_missing = int((na_count > 0).sum())
    n_rows_any_missing = int(remaining.isna().any(axis=1).sum())
    n_complete = int((~remaining.isna().any(axis=1)).sum())
    n_complete_finite = int(remaining_finite.notna().all(axis=1).sum())

    print(f"  Cells:                    {n_cells}  ({n_rows} rows x {n_feat} features)")
    print(
        f"  Missing cells:            {n_missing_cells} "
        f"({100.0 * n_missing_cells / n_cells:.2f}% of remaining-feature cells)"
    )
    print(
        f"  Non-finite (inf) cells:   {n_inf_cells} "
        f"({100.0 * n_inf_cells / n_cells:.2f}%)"
    )
    print(
        f"  Features with any NA:     {n_feat_any_missing}/{n_feat} "
        f"({100.0 * n_feat_any_missing / n_feat:.1f}%)"
    )
    print(
        f"  Rows with any NA:         {n_rows_any_missing}/{n_rows} "
        f"({100.0 * n_rows_any_missing / n_rows:.1f}%)"
    )
    print(
        f"  Complete-case rows:       {n_complete}/{n_rows} "
        f"({100.0 * n_complete / n_rows:.1f}%)"
    )
    print(
        f"  Finite complete-case rows: {n_complete_finite}/{n_rows} "
        f"({100.0 * n_complete_finite / n_rows:.1f}%)"
    )

    with pd.option_context(
        "display.max_rows", None,
        "display.width", 140,
        "display.float_format", "{:.2f}".format,
    ):
        print("\n  Per-feature missingness:")
        print(missing_tbl.to_string())

    missing_only = missing_tbl.loc[missing_tbl["n_missing"] + missing_tbl["n_inf"] > 0]
    if missing_only.empty:
        print("\n  No missing or non-finite values in remaining features.")
    else:
        print("\n  Features with missing or inf values:")
        print(missing_only.to_string())

    print("\n[3] Descriptive statistics (pairwise-available values; NaN/inf ignored)")
    nunique = remaining_finite.nunique(dropna=True)
    desc = remaining_finite.agg(
        ["count", "mean", "std", "min", "median", "max", "skew"]
    ).T
    q25 = remaining_finite.quantile(0.25)
    q75 = remaining_finite.quantile(0.75)
    desc["q25"] = q25
    desc["q75"] = q75
    desc["iqr"] = q75 - q25
    desc["cv"] = desc["std"] / desc["mean"].replace(0, np.nan)
    desc["n_unique"] = nunique
    desc.insert(0, "group", [name_to_group.get(c, "") for c in desc.index])
    desc.insert(1, "code", [name_to_code.get(c, "") for c in desc.index])
    desc = desc[
        [
            "group",
            "code",
            "count",
            "n_unique",
            "mean",
            "std",
            "cv",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "iqr",
            "skew",
        ]
    ]
    n_constant = int((nunique <= 1).sum())
    print(f"  Near-constant features (n_unique <= 1): {n_constant}")
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 180,
        "display.float_format", "{:.4g}".format,
    ):
        print(desc.to_string())

    print("\n[4] Pearson correlation (remaining features only, pairwise complete)")
    corr = remaining_finite.corr(method="pearson", min_periods=3)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", "{:.3f}".format,
    ):
        print(corr.to_string())

    pair_rows = []
    cols = list(remaining_finite.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pair = remaining_finite[[a, b]].dropna()
            n_pair = len(pair)
            if n_pair < 3 or pair[a].std(ddof=1) == 0 or pair[b].std(ddof=1) == 0:
                pair_rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "n": n_pair,
                        "r": np.nan,
                        "p": np.nan,
                        "abs_r": np.nan,
                    }
                )
                continue
            r, p = pearsonr(pair[a].to_numpy(), pair[b].to_numpy())
            pair_rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "n": n_pair,
                    "r": float(r),
                    "p": float(p),
                    "abs_r": abs(float(r)),
                }
            )

    pairs = pd.DataFrame(pair_rows)
    n_pairs = len(pairs)
    n_tested = int(pairs["r"].notna().sum())
    bonferroni = ALPHA / n_tested if n_tested else np.nan
    pairs["bonferroni_sig"] = pairs["p"] < bonferroni if n_tested else False

    print(f"\n  Pairwise tests:           {n_tested}/{n_pairs} (pairs with n>=3 and non-zero variance)")
    print(f"  Alpha (uncorrected):      {ALPHA}")
    if n_tested:
        print(f"  Bonferroni alpha:         {bonferroni:.3e}  (0.05 / {n_tested})")

    high = pairs.loc[pairs["abs_r"] >= CORR_HIGHLIGHT].sort_values(
        "abs_r", ascending=False
    )
    print(
        f"\n  Highly correlated pairs (|r| >= {CORR_HIGHLIGHT:.2f}): "
        f"{len(high)}"
    )
    if high.empty:
        print("    none")
    else:
        with pd.option_context(
            "display.max_rows", None,
            "display.width", 160,
            "display.float_format", "{:.4g}".format,
        ):
            print(
                high[["feature_a", "feature_b", "n", "r", "p", "bonferroni_sig"]]
                .to_string(index=False)
            )

    sig = pairs.loc[pairs["bonferroni_sig"]].sort_values("abs_r", ascending=False)
    print(f"\n  Bonferroni-significant Pearson pairs (alpha={ALPHA}): {len(sig)}")
    if not sig.empty:
        print(
            f"    of which |r| >= {CORR_HIGHLIGHT:.2f}: "
            f"{int((sig['abs_r'] >= CORR_HIGHLIGHT).sum())}"
        )

    abs_r = pairs["abs_r"].dropna()
    if len(abs_r):
        print("\n  |r| distribution (tested pairs):")
        print(f"    mean={abs_r.mean():.3f}  median={abs_r.median():.3f}  "
              f"min={abs_r.min():.3f}  max={abs_r.max():.3f}")
        for thr in (0.50, 0.70, 0.80, 0.90, 0.95):
            n_thr = int((abs_r >= thr).sum())
            print(
                f"    |r| >= {thr:.2f}: {n_thr}/{n_tested} "
                f"({100.0 * n_thr / n_tested:.1f}%)"
            )

    print("\n[5] Copy-ready paper sentence")
    print(
        f"  Remaining-feature table (appendix): n={n_rows} feeding windows, "
        f"{n_feat} features. "
        f"Missing cells: {n_missing_cells}/{n_cells} "
        f"({100.0 * n_missing_cells / n_cells:.2f}%); "
        f"{n_feat_any_missing} features and {n_rows_any_missing} windows "
        f"had at least one missing value; complete-case n={n_complete}. "
        f"Pairwise Pearson correlation identified {len(high)} pairs with "
        f"|r| >= {CORR_HIGHLIGHT:.2f} (Bonferroni-adjusted alpha={bonferroni:.2e}, "
        f"{n_tested} tests)."
    )

    fig_w = max(11, 0.42 * n_feat)
    fig_h = max(9, 0.42 * n_feat)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        square=True,
        cbar_kws={"label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
    )

    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    plt.show()

    n_hist_cols = 4
    n_hist_rows = int(np.ceil(n_feat / n_hist_cols))
    fig, axs = plt.subplots(
        n_hist_rows,
        n_hist_cols,
        figsize=(n_hist_cols * 4.4, n_hist_rows * 3.1),
        squeeze=False,
    )
    for i, col in enumerate(remaining_finite.columns):
        ax = axs[i // n_hist_cols, i % n_hist_cols]
        values = remaining_finite[col].dropna()
        group = name_to_group.get(col, "")
        n_unique = int(values.nunique())
        if len(values) == 0:
            ax.set_title(col, fontsize=10)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        n_bins = (
            1
            if n_unique <= 1
            else min(
                20,
                max(8, n_unique if n_unique < 8 else int(np.sqrt(len(values))) + 5),
            )
        )
        ax.hist(
            values,
            bins=n_bins,
            color="steelblue",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.axvline(
            float(values.mean()),
            color="black",
            linestyle="--",
            linewidth=1.1,
            label="Mean",
        )
        ax.axvline(
            float(values.median()),
            color="dimgray",
            linestyle=":",
            linewidth=1.1,
            label="Median",
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(group, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=8)
        n_miss = int(remaining[col].isna().sum())
        note = f"n={len(values)}"
        if n_miss:
            note += f", NA={n_miss}"
        ax.text(
            0.98,
            0.96,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    for j in range(n_feat, n_hist_rows * n_hist_cols):
        axs[j // n_hist_cols, j % n_hist_cols].axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("Histograms — appendix remaining features", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()


def plot_normalized_windows(times, values, feed_times, channel):
    n = len(feed_times)
    nrows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(3 * NCOLS, 2.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, ft in enumerate(feed_times):
        ax = axes[i // NCOLS, i % NCOLS]
        window_t, window_y = slice_window(times, values, ft)
        ft64 = np.datetime64(pd.Timestamp(ft), "ns")
        if len(window_t):
            minutes = (window_t - ft64) / np.timedelta64(1, "m")
            ax.plot(minutes, window_y, linewidth=0.8)
        ax.axvline(0, color="red", linewidth=0.8)
        ax.set_xlim(-30, 30)
        ax.set_xticks([-30, 0, 30])
        ax.set_title(pd.Timestamp(ft).strftime("%m-%d %H:%M"), fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * NCOLS):
        axes[j // NCOLS, j % NCOLS].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel(channel)
    for ax in axes[-1, :]:
        ax.set_xlabel("minutes from feeding")

    fig.suptitle(
        f"{channel} normalized 0-1, 30 min before and after "
        f"{n} evenly spaced feedings"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_feature_overlays(times, values, feed_times, channel, title=None):
    n = len(feed_times)
    nrows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(3.4 * NCOLS, 2.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    n_fail = 0

    for i, ft in enumerate(feed_times):
        ax = axes[i // NCOLS, i % NCOLS]
        df_window = window_frame(times, values, ft)
        ax.set_title(pd.Timestamp(ft).strftime("%m-%d %H:%M"), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)

        if df_window.empty:
            n_fail += 1
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            continue

        features = extract_mountain_features(
            df_window,
            value_col="value",
            time_col="time",
            plateau_threshold=0.90,
            ax=ax,
        )
        if not features:
            n_fail += 1
            ax.text(
                0.5,
                0.5,
                "No mountain",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    for j in range(n, nrows * NCOLS):
        axes[j // NCOLS, j % NCOLS].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel(channel)
    for ax in axes[-1, :]:
        ax.set_xlabel("minutes in window")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=3)

    fig.suptitle(title or f"{channel} feeding curves")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, n_fail


APPENDIX_COLS = [col for _, _, col in APPENDIX_FEATURES]


def extract_channel_rows(channel, times, values, feed_times):
    """One row per feeding. Failed windows keep the row and leave features empty."""
    rows = []
    for ft in feed_times:
        df_window = window_frame(times, values, ft)
        features = {}
        if not df_window.empty:
            features = extract_mountain_features(
                df_window,
                value_col="value",
                time_col="time",
                plateau_threshold=0.90,
            )
        row = {
            "channel": channel,
            "feeding_datetime": pd.Timestamp(ft),
        }
        for col in APPENDIX_COLS:
            if col in ("feeding_sin", "feeding_cos"):
                continue
            row[col] = features.get(col, np.nan) if features else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_feeding_time_features(features_df):
    ts = pd.to_datetime(features_df["feeding_datetime"])
    minutes = ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60.0
    angle = 2.0 * np.pi * minutes / MINUTES_PER_DAY
    out = features_df.copy()
    out["feeding_sin"] = np.sin(angle)
    out["feeding_cos"] = np.cos(angle)
    return out[["channel", "feeding_datetime", *APPENDIX_COLS]]


def save_curve_pngs(channel, times, values, feed_times, out_dir):
    parts = np.array_split(np.arange(len(feed_times)), N_CURVE_PNGS)
    for part_i, idx in enumerate(parts, start=1):
        subset = feed_times.iloc[idx]
        first = int(idx[0]) + 1
        last = int(idx[-1]) + 1
        fig, _ = plot_feature_overlays(
            times,
            values,
            subset,
            channel,
            title=f"{channel} feeding curves {first}-{last}",
        )
        path = os.path.join(out_dir, f"{channel}_curves_{part_i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    signal = pd.read_parquet(SIGNAL_PATH, columns=[*CHANNELS, "real time"])
    signal["t"] = to_datetime(signal["real time"])
    signal = signal.sort_values("t")
    times = signal["t"].to_numpy(dtype="datetime64[ns]")

    feed = pd.read_parquet(FEED_PATH, columns=["real time"])
    feed_times = (
        to_datetime(feed["real time"]).sort_values().reset_index(drop=True)
    )

    frames = []
    for channel in CHANNELS:
        values = minmax_channel(signal[channel].to_numpy())
        frames.append(extract_channel_rows(channel, times, values, feed_times))
        save_curve_pngs(channel, times, values, feed_times, OUT_DIR)
        print(channel, "features", len(frames[-1]))

    features = pd.concat(frames, ignore_index=True)
    features = add_feeding_time_features(features)
    features = features.sort_values(
        ["feeding_datetime", "channel"]
    ).reset_index(drop=True)
    out = os.path.join(OUT_DIR, "features.parquet")
    features.to_parquet(out, index=False)
    print(features.shape)
    print(out)


if __name__ == "__main__":
    main()
