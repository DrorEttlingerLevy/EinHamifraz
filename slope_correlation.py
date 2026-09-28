"""Correlation of z-scored 3-day rolling means and of their slopes.

Uses the same tanks as plot 2/3 in i.ipynb, but only real calendar neighbours.

Missing days are kept as gaps (not dropped). The 3-day rolling mean needs 3
measured calendar days in a row. A slope is kept only if it is from one
calendar day to the next (no jump over a hole). Gap-spanning slopes are
NaN so they cannot fake a shared rise.

Methods
-------
Pearson on levels : do the *heights* move together (global trend + shared level).
Pearson on slopes : do the *changes* move together (up/down on the same days).
Spearman          : same idea, but ranks (monotonic, less pulled by spikes).
Sign agreement    : fraction of days both tanks go the same way (up or down).
Joint             : Pearson after stacking [level, slope] for each tank
                    (equal weight on value and change).
Rolling Pearson   : 7-day window on levels and on slopes (local, varies in time).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "feeding_values.csv"

CHANNEL_LABELS = {"Ch2": "Tank 1", "Ch4": "Tank 2", "Ch6": "Tank 3"}
CHANNEL_COLS = ["Ch2", "Ch4", "Ch6"]
PAIRS = [("Ch2", "Ch4"), ("Ch2", "Ch6"), ("Ch4", "Ch6")]
ROLLING_WINDOW = 3
LOCAL_WINDOW = 7
STRESS_DATES = [pd.Timestamp("2025-09-16"), pd.Timestamp("2025-09-30")]
COLORS = ["tab:green", "tab:orange", "tab:blue"]
PAIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Return z-scored rolling levels, consecutive-day slopes, and gap dates."""
    df = pd.read_csv(CSV_PATH, parse_dates=["max_time"]).set_index("max_time")
    daily = df[CHANNEL_COLS].resample("D").mean()
    measured = daily.dropna(how="all").index
    gaps = daily.index.difference(measured)

    mask_ch6 = (daily.index.month < 10) | (
        (daily.index.month == 10) & (daily.index.day <= 14)
    )
    plot = daily.copy()
    plot.loc[~mask_ch6, "Ch6"] = np.nan

    rolling = plot.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    levels_z = pd.DataFrame(index=rolling.index)
    for col in CHANNEL_COLS:
        x = rolling[col]
        levels_z[col] = (x - x.mean(skipna=True)) / x.std(skipna=True)

    slopes_z = levels_z.diff()
    day_step = pd.Series(levels_z.index, index=levels_z.index).diff().dt.days
    slopes_z.loc[day_step.ne(1)] = np.nan
    slopes_z = slopes_z.where(levels_z.shift(1).notna())
    return levels_z, slopes_z, gaps


def pearson(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
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


def joint_pearson(level_a, slope_a, level_b, slope_b) -> float:
    """Pearson of concatenated [level, slope] after z-scoring each part."""
    la = (level_a - level_a.mean()) / level_a.std()
    lb = (level_b - level_b.mean()) / level_b.std()
    sa = (slope_a - slope_a.mean()) / slope_a.std()
    sb = (slope_b - slope_b.mean()) / slope_b.std()
    va = pd.concat([la, sa])
    vb = pd.concat([lb, sb])
    return pearson(va.reset_index(drop=True), vb.reset_index(drop=True))


def pair_metrics(levels: pd.DataFrame, slopes: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    if mask is not None:
        levels = levels.loc[mask]
        slopes = slopes.loc[mask]
    rows = []
    for a, b in PAIRS:
        rows.append(
            {
                "pair": f"{CHANNEL_LABELS[a]}–{CHANNEL_LABELS[b]}",
                "Pearson levels": pearson(levels[a], levels[b]),
                "Pearson slopes": pearson(slopes[a], slopes[b]),
                "Spearman levels": spearman(levels[a], levels[b]),
                "Spearman slopes": spearman(slopes[a], slopes[b]),
                "Same-direction slopes": sign_agree(slopes[a], slopes[b]),
                "Joint level+slope": joint_pearson(
                    levels[a], slopes[a], levels[b], slopes[b]
                ),
                "n level days": int(
                    pd.concat([levels[a], levels[b]], axis=1).dropna().shape[0]
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
    return {
        "Usual (before 16 Sep)": d < first,
        "After 1st stress (16 Sep–29 Sep)": (d >= first) & (d < second),
        "After 2nd stress (from 30 Sep)": d >= second,
        "All days after 16 Sep": d >= first,
    }


def rolling_pair_corr(df: pd.DataFrame, window: int = LOCAL_WINDOW) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for a, b in PAIRS:
        out[f"{CHANNEL_LABELS[a]}–{CHANNEL_LABELS[b]}"] = (
            df[a].rolling(window, min_periods=window).corr(df[b])
        )
    return out


def print_table(title: str, tbl: pd.DataFrame) -> None:
    print(title)
    print(tbl.round(3).to_string())
    print()


def main() -> None:
    levels, slopes, gaps = load_series()

    print("Series: 3-day rolling mean on the calendar (holes stay holes).")
    print("Slopes: only this day → next calendar day (no jump over a gap).")
    print(
        "Calendar gaps (excluded from slopes): "
        + ", ".join(d.strftime("%Y-%m-%d") for d in gaps)
    )
    print()
    print("What each number means")
    print("  Pearson levels     : tanks sit high/low together (global height).")
    print("  Pearson slopes     : tanks rise/fall on the same days (local change).")
    print("  Spearman           : same, but by rank (less sensitive to one spike).")
    print("  Same-direction     : share of days both slopes have the same sign.")
    print("  Joint level+slope  : one r from stacking height and change together.")
    print("  Rolling 7-day r    : local correlation; high in some weeks, low in others.")
    print()

    global_tbl = pair_metrics(levels, slopes)
    print_table("Global (whole experiment)", global_tbl)

    masks = period_masks(levels.index)
    period_rows = []
    for name, mask in masks.items():
        tbl = pair_metrics(levels, slopes, mask)
        print_table(name, tbl)
        mean_level = tbl["Pearson levels"].mean()
        mean_slope = tbl["Pearson slopes"].mean()
        mean_sign = tbl["Same-direction slopes"].mean()
        period_rows.append(
            {
                "period": name,
                "mean Pearson levels": mean_level,
                "mean Pearson slopes": mean_slope,
                "mean same-direction": mean_sign,
            }
        )
    period_summary = pd.DataFrame(period_rows).set_index("period")
    print_table("Usual vs after stress (mean across the 3 tank pairs)", period_summary)

    roll_level = rolling_pair_corr(levels)
    roll_slope = rolling_pair_corr(slopes)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax = axes[0]
    for col, color in zip(roll_level.columns, PAIR_COLORS):
        ax.plot(roll_level.index, roll_level[col], color=color, lw=1.8, label=col)
    for dt in STRESS_DATES:
        ax.axvline(dt, color="red", ls="--", lw=1.5)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("7-day Pearson r")
    ax.set_title("Local correlation of LEVELS (z-scored 3-day rolling mean)")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col, color in zip(roll_slope.columns, PAIR_COLORS):
        ax.plot(roll_slope.index, roll_slope[col], color=color, lw=1.8, label=col)
    for i, dt in enumerate(STRESS_DATES):
        ax.axvline(
            dt,
            color="red",
            ls="--",
            lw=1.5,
            label="Weighing stress" if i == 0 else None,
        )
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("7-day Pearson r")
    ax.set_xlabel("Date")
    ax.set_title("Local correlation of SLOPES (day-to-day change of that z-score)")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    level_path = DATA_DIR / "slope_corr_rolling.png"
    fig.savefig(level_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(period_summary))
    w = 0.35
    ax.bar(
        x - w / 2,
        period_summary["mean Pearson levels"],
        w,
        label="Pearson of levels",
        color="0.55",
    )
    ax.bar(
        x + w / 2,
        period_summary["mean Pearson slopes"],
        w,
        label="Pearson of slopes",
        color="tab:blue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Before 16 Sep\n(usual)", "16–29 Sep\n(after 1st)", "From 30 Sep\n(after 2nd)", "All after\n16 Sep"],
        fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean r across tank pairs")
    ax.set_title("Usual vs after stress")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = DATA_DIR / "slope_corr_periods.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    print("Saved", level_path.name)
    print("Saved", bar_path.name)
    global_tbl.to_csv(DATA_DIR / "slope_corr_global.csv")
    period_summary.to_csv(DATA_DIR / "slope_corr_periods.csv")


if __name__ == "__main__":
    main()
