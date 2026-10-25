"""Pairwise correlation of 3-day rolling *normalized* feeding, using both values and slopes.

Working series
--------------
Daily mean of Ch2 / Ch4 / Ch6 (tanks 1 / 2 / 3). Each channel is z-scored,
then a 3-day rolling mean is taken. That is the *value* series. The *slope*
is the day-to-day change, kept only when the two days are calendar neighbours
(a gap is not treated as a one-day step).

Correlation
-----------
Values  : Pearson / Spearman of the rolling z-scores (do heights move together).
Slopes  : Pearson / Spearman of those day-to-day changes (do they rise/fall together).
Joint   : Pearson after stacking [z(value), z(slope)] — equal weight on height and change.
Combined: mean of Pearson(values) and Pearson(slopes).

Where
-----
Global  : whole experiment.
Periods : named calendar intervals (usual vs after each weighing stress).
Local   : rolling windows of several lengths (5 / 7 / 14 days).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "feeding_values.csv"

CHANNEL_LABELS = {"Ch2": "Tank 1", "Ch4": "Tank 2", "Ch6": "Tank 3"}
CHANNEL_COLS = ["Ch2", "Ch4", "Ch6"]
PAIRS = [("Ch2", "Ch4"), ("Ch2", "Ch6"), ("Ch4", "Ch6")]
ROLLING_DAYS = 3
LOCAL_WINDOWS = (5, 7, 14)
STRESS_DATES = [pd.Timestamp("2025-09-16"), pd.Timestamp("2025-09-30")]
EVENT_RADIUS_DAYS = 3
CH6_LAST_DAY = pd.Timestamp("2025-10-14")
PAIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
CH_COLORS = {"Ch2": "#1f77b4", "Ch4": "#ff7f0e", "Ch6": "#2ca02c"}


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(skipna=True)
    if sd == 0 or not np.isfinite(sd):
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd


def load_rolling_normalized(
    path: Path = CSV_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Return (values, slopes, gap dates).

    values : z-scored daily means, then 3-day rolling mean
    slopes : consecutive-calendar-day diff of those values
    """
    df = pd.read_csv(path, parse_dates=["max_time"]).set_index("max_time")
    daily = df[CHANNEL_COLS].resample("D").mean()
    measured = daily.dropna(how="all").index
    gaps = daily.index.difference(measured)

    plot = daily.copy()
    plot.loc[plot.index > CH6_LAST_DAY, "Ch6"] = np.nan

    values = pd.DataFrame(index=plot.index)
    for col in CHANNEL_COLS:
        values[col] = (
            zscore(plot[col]).rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean()
        )

    slopes = values.diff()
    day_step = pd.Series(values.index, index=values.index).diff().dt.days
    slopes.loc[day_step.ne(1)] = np.nan
    slopes = slopes.where(values.shift(1).notna())
    return values, slopes, gaps


def pearson(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="pearson"))


def spearman(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def sign_agree(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return np.nan
    sa = np.sign(aligned.iloc[:, 0].to_numpy())
    sb = np.sign(aligned.iloc[:, 1].to_numpy())
    nz = (sa != 0) & (sb != 0)
    if nz.sum() == 0:
        return np.nan
    return float((sa[nz] == sb[nz]).mean())


def mean_of(a: float, b: float) -> float:
    arr = np.array([a, b], dtype=float)
    if not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmean(arr))


def joint_pearson(level_a, slope_a, level_b, slope_b) -> float:
    """Pearson of concatenated [z(value), z(slope)] — values and slopes together."""
    la, lb = zscore(level_a), zscore(level_b)
    sa, sb = zscore(slope_a), zscore(slope_b)
    va = pd.concat([la, sa]).reset_index(drop=True)
    vb = pd.concat([lb, sb]).reset_index(drop=True)
    return pearson(va, vb)


def pair_metrics(
    values: pd.DataFrame,
    slopes: pd.DataFrame,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    if mask is not None:
        values = values.loc[mask]
        slopes = slopes.loc[mask]
    rows = []
    for a, b in PAIRS:
        r_val = pearson(values[a], values[b])
        r_slp = pearson(slopes[a], slopes[b])
        rows.append(
            {
                "pair": f"{CHANNEL_LABELS[a]}–{CHANNEL_LABELS[b]}",
                "Pearson values": r_val,
                "Pearson slopes": r_slp,
                "Spearman values": spearman(values[a], values[b]),
                "Spearman slopes": spearman(slopes[a], slopes[b]),
                "Same-direction slopes": sign_agree(slopes[a], slopes[b]),
                "Joint value+slope": joint_pearson(
                    values[a], slopes[a], values[b], slopes[b]
                ),
                "Combined (mean r)": mean_of(r_val, r_slp),
                "n value days": int(
                    pd.concat([values[a], values[b]], axis=1).dropna().shape[0]
                ),
                "n slope days": int(
                    pd.concat([slopes[a], slopes[b]], axis=1).dropna().shape[0]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("pair")


def period_masks(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    d = pd.Series(index, index=index)
    first, second = STRESS_DATES
    masks = {
        "Global (all days)": pd.Series(True, index=index),
        "Usual (before 16 Sep)": d < first,
        "After 1st stress (16–29 Sep)": (d >= first) & (d < second),
        "After 2nd stress (from 30 Sep)": d >= second,
        "All days after 16 Sep": d >= first,
    }
    for event in STRESS_DATES:
        label = f"Local ±{EVENT_RADIUS_DAYS}d around {event.strftime('%d %b').lstrip('0')}"
        masks[label] = (d >= event - pd.Timedelta(days=EVENT_RADIUS_DAYS)) & (
            d <= event + pd.Timedelta(days=EVENT_RADIUS_DAYS)
        )
    weeks = d.dt.isocalendar().week.astype(int)
    for week in sorted(weeks.unique()):
        week_mask = weeks == week
        if int(week_mask.sum()) < 3:
            continue
        start = d[week_mask].min()
        end = d[week_mask].max()
        label = (
            f"Week {int(week)} "
            f"({start.strftime('%d %b').lstrip('0')}–{end.strftime('%d %b').lstrip('0')})"
        )
        masks[label] = week_mask
    return masks


def rolling_pair_corr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for a, b in PAIRS:
        out[f"{CHANNEL_LABELS[a]}–{CHANNEL_LABELS[b]}"] = df[a].rolling(
            window, min_periods=window
        ).corr(df[b])
    return out


def rolling_combined_corr(
    values: pd.DataFrame, slopes: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Local combined score: mean of rolling Pearson(values) and Pearson(slopes)."""
    r_val = rolling_pair_corr(values, window)
    r_slp = rolling_pair_corr(slopes, window)
    return (r_val + r_slp) / 2.0


def rolling_joint_corr(
    values: pd.DataFrame, slopes: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Rolling Pearson of stacked [z(value), z(slope)] in each window."""
    out = pd.DataFrame(index=values.index, columns=[
        f"{CHANNEL_LABELS[a]}–{CHANNEL_LABELS[b]}" for a, b in PAIRS
    ], dtype=float)
    n = len(values)
    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        for a, b in PAIRS:
            name = f"{CHANNEL_LABELS[a]}–{CHANNEL_LABELS[b]}"
            out.iloc[i, out.columns.get_loc(name)] = joint_pearson(
                values[a].iloc[sl],
                slopes[a].iloc[sl],
                values[b].iloc[sl],
                slopes[b].iloc[sl],
            )
    return out


def print_table(title: str, tbl: pd.DataFrame) -> None:
    print(title)
    print(tbl.round(3).to_string())
    print()


def plot_series(values: pd.DataFrame, slopes: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax = axes[0]
    for col in CHANNEL_COLS:
        ax.plot(
            values.index,
            values[col],
            color=CH_COLORS[col],
            lw=1.8,
            marker="o",
            ms=3.5,
            label=CHANNEL_LABELS[col],
        )
    for i, dt in enumerate(STRESS_DATES):
        ax.axvline(
            dt,
            color="red",
            ls="--",
            lw=1.4,
            label="Weighing stress" if i == 0 else None,
        )
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("z-score")
    ax.set_title(f"{ROLLING_DAYS}-day rolling mean of z-scored daily feeding")
    ax.legend(loc="upper left", frameon=False, ncol=4)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col in CHANNEL_COLS:
        ax.plot(
            slopes.index,
            slopes[col],
            color=CH_COLORS[col],
            lw=1.8,
            marker="o",
            ms=3.5,
            label=CHANNEL_LABELS[col],
        )
    for dt in STRESS_DATES:
        ax.axvline(dt, color="red", ls="--", lw=1.4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Δ z-score / day")
    ax.set_xlabel("Date")
    ax.set_title("Slope: consecutive-day change of the rolling z-score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_local(
    values: pd.DataFrame,
    slopes: pd.DataFrame,
    window: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r_val = rolling_pair_corr(values, window)
    r_slp = rolling_pair_corr(slopes, window)
    r_comb = rolling_combined_corr(values, slopes, window)
    r_joint = rolling_joint_corr(values, slopes, window)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    panels = [
        (r_val, "values (heights)"),
        (r_slp, "slopes (day-to-day change)"),
        (r_comb, "combined = mean(r_values, r_slopes)"),
        (r_joint, "joint Pearson of stacked [value, slope]"),
    ]
    for ax, (tbl, title) in zip(axes, panels):
        for col, color in zip(tbl.columns, PAIR_COLORS):
            ax.plot(tbl.index, tbl[col], color=color, lw=1.8, label=col)
        for dt in STRESS_DATES:
            ax.axvline(dt, color="red", ls="--", lw=1.3)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Local {window}-day correlation of {title}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower left", frameon=False)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return r_val, r_slp, r_comb, r_joint


def plot_period_bars(period_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(period_summary))
    w = 0.25
    ax.bar(x - w, period_summary["mean Pearson values"], w, label="Values", color="0.55")
    ax.bar(x, period_summary["mean Pearson slopes"], w, label="Slopes", color="tab:blue")
    ax.bar(
        x + w,
        period_summary["mean combined"],
        w,
        label="Combined (mean of both)",
        color="tab:green",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(period_summary.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("Mean r across tank pairs")
    ax.set_title("Values vs slopes vs combined, by time interval")
    ax.legend(frameon=False)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    values, slopes, gaps = load_rolling_normalized()

    print(f"Series: z-score each tank, then {ROLLING_DAYS}-day rolling mean.")
    print("Slopes: only this calendar day → next calendar day (gaps stay gaps).")
    print(
        "Calendar gaps: " + ", ".join(d.strftime("%Y-%m-%d") for d in gaps)
        if len(gaps)
        else "Calendar gaps: none"
    )
    print()
    print("What each number means")
    print("  Pearson values     : tanks sit high/low together.")
    print("  Pearson slopes     : tanks rise/fall on the same days.")
    print("  Joint value+slope  : one r from stacking height and change.")
    print("  Combined (mean r)  : 0.5 * r_values + 0.5 * r_slopes.")
    print("  Local W-day r      : the same, inside a sliding window of W days.")
    print()

    masks = period_masks(values.index)
    period_rows = []
    all_period_tables = []
    for name, mask in masks.items():
        tbl = pair_metrics(values, slopes, mask)
        if (tbl["n value days"] < 3).all():
            continue
        print_table(name, tbl)
        tagged = tbl.copy()
        tagged.insert(0, "interval", name)
        all_period_tables.append(tagged.reset_index())
        period_rows.append(
            {
                "interval": name,
                "mean Pearson values": tbl["Pearson values"].mean(),
                "mean Pearson slopes": tbl["Pearson slopes"].mean(),
                "mean joint": tbl["Joint value+slope"].mean(),
                "mean combined": tbl["Combined (mean r)"].mean(),
                "mean same-direction": tbl["Same-direction slopes"].mean(),
            }
        )
    period_summary = pd.DataFrame(period_rows).set_index("interval")
    print_table("Mean across the 3 tank pairs, by interval", period_summary)

    named_intervals = [
        "Usual (before 16 Sep)",
        "After 1st stress (16–29 Sep)",
        "After 2nd stress (from 30 Sep)",
        "All days after 16 Sep",
    ]
    local_event_intervals = [
        k for k in period_summary.index if k.startswith("Local")
    ]
    bar_rows = named_intervals + local_event_intervals
    
    plot_period_bars(period_summary.loc[bar_rows], DATA_DIR / "rolling3d_corr_periods.png")

    plot_series(values, slopes, DATA_DIR / "rolling3d_corr_series.png")

    local_frames = []
    for window in LOCAL_WINDOWS:
        r_val, r_slp, r_comb, r_joint = plot_local(
            values,
            slopes,
            window,
            DATA_DIR / f"rolling3d_corr_local_{window}d.png",
        )
        print(
            f"Local {window}-day mean combined r (across pairs, over time): "
            f"{r_comb.mean().mean():.3f}"
        )
        for kind, tbl in (
            ("values", r_val),
            ("slopes", r_slp),
            ("combined", r_comb),
            ("joint", r_joint),
        ):
            long = tbl.reset_index().melt(
                id_vars=tbl.index.name or "index",
                var_name="pair",
                value_name="r",
            )
            if tbl.index.name is None:
                long = long.rename(columns={"index": "max_time"})
            else:
                long = long.rename(columns={tbl.index.name: "max_time"})
            long.insert(0, "kind", kind)
            long.insert(0, "window_days", window)
            local_frames.append(long)

    period_tbl = pd.concat(all_period_tables, ignore_index=True)
    period_tbl.to_csv(DATA_DIR / "rolling3d_corr_intervals.csv", index=False)
    period_summary.to_csv(DATA_DIR / "rolling3d_corr_interval_means.csv")
    pd.concat(local_frames, ignore_index=True).to_csv(
        DATA_DIR / "rolling3d_corr_local.csv", index=False
    )

    print()
    print("Saved rolling3d_corr_series.png")
    print("Saved rolling3d_corr_periods.png")
    for window in LOCAL_WINDOWS:
        print(f"Saved rolling3d_corr_local_{window}d.png")
    print("Saved rolling3d_corr_intervals.csv")
    print("Saved rolling3d_corr_interval_means.csv")
    print("Saved rolling3d_corr_local.csv")


if __name__ == "__main__":
    main()
