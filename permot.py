"""Circular-shift permutation test for matching 3-day slopes (Ch2, Ch4, Ch6).

Working series
--------------
Each channel is z-scored on its own, then a 3-day rolling mean is taken.
That normalized rolling series is what is plotted, used for OLS slopes,
correlation, and the permutation test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "feeding_values.csv"

CHANNELS = ["Ch2", "Ch4", "Ch6"]
PAIRS = [("Ch2", "Ch4"), ("Ch2", "Ch6"), ("Ch4", "Ch6")]
PAIR_IDX = [(0, 1), (0, 2), (1, 2)]

WINDOW = 3
MIN_VALID = 3
N_PERM = 10_000
RNG_SEED = 42
AROUND = 3  # observation days before and after each query date
# Slope from this calendar day to the next observation on the shared timeline.
QUERY_STARTS = [pd.Timestamp("2025-09-16"), pd.Timestamp("2025-09-30")]
CH_COLORS = {"Ch2": "#1f77b4", "Ch4": "#ff7f0e", "Ch6": "#2ca02c"}
PAIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load_daily_channels(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["max_time"]).sort_values("max_time")
    values = df[["max_time", *CHANNELS]].copy()
    values["Ch6"] = values["Ch6"].mask(values["Ch6"] == 0)
    daily = (
        values.set_index("max_time")
        .resample("D")[CHANNELS]
        .mean()
        .dropna(how="all")
    )
    daily.index = daily.index.normalize()
    last_complete = daily.dropna(how="any").index.max()
    return daily.loc[:last_complete].dropna(how="any")


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        x = out[col].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Cannot z-score {col}: std={sd}")
        out[col] = (x - mu) / sd
    return out


def normalized_rolling_series(daily: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Per-channel z-score, then 3-day rolling mean. This is the working series."""
    return (
        zscore_columns(daily[CHANNELS])
        .rolling(window, min_periods=window)
        .mean()
        .dropna()
    )


def _window_slope(x: np.ndarray, y: np.ndarray, min_valid: int) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < min_valid:
        return np.nan
    xt = x[mask].astype(float) - x[mask].mean()
    denom = float(np.dot(xt, xt))
    if denom == 0:
        return np.nan
    yt = y[mask].astype(float) - y[mask].mean()
    return float(np.dot(xt, yt) / denom)


def rolling_slopes(
    dates: np.ndarray,
    Y: np.ndarray,
    window: int = WINDOW,
    min_valid: int = MIN_VALID,
) -> np.ndarray:
    n_ch, n = Y.shape
    n_win = n - window + 1
    if n_win < 1:
        raise ValueError("Series shorter than the moving window")
    x = dates.astype("datetime64[D]").astype(np.int64).astype(float)
    slopes = np.full((n_win, n_ch), np.nan, dtype=float)
    for i in range(n_win):
        for t in range(n_ch):
            slopes[i, t] = _window_slope(
                x[i : i + window], Y[t, i : i + window], min_valid
            )
    return slopes


def all_three_match(slopes: np.ndarray) -> np.ndarray:
    """True where all three finite slopes have the same sign."""
    valid = np.isfinite(slopes).all(axis=1)
    return valid & (((slopes > 0).all(axis=1)) | ((slopes < 0).all(axis=1)))


def alignment_R(slopes: np.ndarray) -> np.ndarray:
    valid = np.isfinite(slopes).all(axis=1)
    theta = np.arctan(slopes)
    r = np.sqrt(np.nanmean(np.cos(theta), axis=1) ** 2 + np.nanmean(np.sin(theta), axis=1) ** 2)
    r[~valid] = np.nan
    return r


def consecutive_day_slopes(dates: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Slope of each channel from day i to day i+1. Shape (n-1, n_channels)."""
    x = dates.astype("datetime64[D]").astype(np.int64).astype(float)
    dx = np.diff(x)
    dy = np.diff(Y, axis=1)
    slopes = np.full((Y.shape[1] - 1, Y.shape[0]), np.nan, dtype=float)
    ok = dx != 0
    slopes[ok] = (dy[:, ok] / dx[ok]).T
    return slopes


def match_strength(slopes_row: np.ndarray) -> float:
    """Larger when all three slopes share sign and similar steepness."""
    if not np.isfinite(slopes_row).all():
        return np.nan
    if not (((slopes_row > 0).all()) or ((slopes_row < 0).all())):
        return 0.0
    r = float(alignment_R(slopes_row.reshape(1, -1))[0])
    return r * float(np.abs(slopes_row).mean())


def print_consecutive_match_list(dates: np.ndarray, pair_slopes: np.ndarray) -> None:
    """Rank consecutive day-pairs (1→2, 2→3, ...) by 3-channel slope match."""
    starts = pd.to_datetime(dates[:-1])
    ends = pd.to_datetime(dates[1:])
    strength = np.array([match_strength(row) for row in pair_slopes])
    r_vals = alignment_R(pair_slopes)
    match = all_three_match(pair_slopes)
    order = np.argsort(-np.nan_to_num(strength, nan=-np.inf))

    print("Strongest consecutive-day slope matches (day i → i+1 only)")
    print("  rank  dates                    dir     R     strength  "
          f"{CHANNELS[0]:>10} {CHANNELS[1]:>10} {CHANNELS[2]:>10}")
    rank = 0
    for i in order:
        if not match[i] or not np.isfinite(strength[i]) or strength[i] <= 0:
            continue
        rank += 1
        direction = "up  " if pair_slopes[i].mean() > 0 else "down"
        print(
            f"  {rank:>4}  {starts[i].date()} → {ends[i].date()}  {direction}  "
            f"{r_vals[i]:.3f}  {strength[i]:8.4f}  "
            f"{pair_slopes[i, 0]:10.4f} {pair_slopes[i, 1]:10.4f} {pair_slopes[i, 2]:10.4f}"
        )
    if rank == 0:
        print("  (no consecutive day-pairs where all 3 slopes match)")
    print()


def pairwise_pearson(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for a, b in PAIRS:
        x = df[a].to_numpy(dtype=float)
        y = df[b].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
            out[(a, b)] = np.nan
            continue
        out[(a, b)] = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return out


def circular_shift_rows(Y: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    shifted = np.empty_like(Y)
    for t, k in enumerate(shifts):
        shifted[t] = np.roll(Y[t], int(k))
    return shifted


def pair_index_for_start(dates: np.ndarray, start: pd.Timestamp) -> int:
    """Index of the consecutive-day slope that begins on `start`."""
    starts = pd.to_datetime(dates[:-1]).normalize()
    hits = np.flatnonzero(starts == pd.Timestamp(start).normalize())
    if hits.size == 0:
        raise KeyError(f"No consecutive-day step starting {pd.Timestamp(start).date()}")
    return int(hits[0])


def permutation_null_same_times(
    dates: np.ndarray,
    Y: np.ndarray,
    query_idx: list[int],
    n_perm: int = N_PERM,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift channels independently; dates stay fixed so 'same time' is preserved.

    Returns
    -------
    null_rate : match rate over the whole timeline
    match_at_query : (n_perm, n_query) whether all 3 match at each query date
    n_query_match : how many of the query dates matched in that permutation
    """
    rng = np.random.default_rng(seed)
    n_ch, n = Y.shape
    n_query = len(query_idx)
    null_rate = np.empty(n_perm, dtype=float)
    match_at_query = np.zeros((n_perm, n_query), dtype=bool)
    n_query_match = np.empty(n_perm, dtype=float)
    shifts = rng.integers(0, n, size=(n_perm, n_ch), dtype=np.int64)
    for p in range(n_perm):
        Yp = circular_shift_rows(Y, shifts[p])
        match = all_three_match(consecutive_day_slopes(dates, Yp))
        null_rate[p] = float(match.mean()) if match.size else 0.0
        for j, idx in enumerate(query_idx):
            match_at_query[p, j] = bool(match[idx])
        n_query_match[p] = float(match_at_query[p].sum())
        if (p + 1) % 2000 == 0:
            print(f"  permutations {p + 1}/{n_perm}")
    return null_rate, match_at_query, n_query_match


def empirical_pvalue(observed: float, null: np.ndarray) -> float:
    return (int(np.sum(null >= observed)) + 1) / (len(null) + 1)


def before_after_indices(center: int, n_pairs: int, k: int = AROUND) -> tuple[list[int], list[int]]:
    """k consecutive-day steps before `center`, and k after (center itself excluded)."""
    before = [j for j in range(center - k, center) if 0 <= j < n_pairs]
    after = [j for j in range(center + 1, center + 1 + k) if 0 <= j < n_pairs]
    return before, after


def print_region(
    label: str,
    idxs: list[int],
    pair_match: np.ndarray,
    pair_slopes: np.ndarray,
    pair_starts: pd.DatetimeIndex,
    pair_ends: pd.DatetimeIndex,
) -> None:
    if not idxs:
        print(f"  {label}: (no days in range)")
        return
    n = len(idxs)
    n_match = int(pair_match[idxs].sum())
    print(f"  {label}: {n_match} / {n} days all 3 same direction ({100.0 * n_match / n:.1f}%)")
    for j in idxs:
        row = pair_slopes[j]
        if (row > 0).all():
            direction = "UP  "
        elif (row < 0).all():
            direction = "DOWN"
        else:
            direction = "MIXED"
        flag = "match" if pair_match[j] else "diff "
        print(
            f"    {pair_starts[j].date()} → {pair_ends[j].date()}  {flag} {direction}  "
            f"Ch2={row[0]:+.3f}  Ch4={row[1]:+.3f}  Ch6={row[2]:+.3f}"
        )
    series: pd.DataFrame,
    query_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col, color in CH_COLORS.items():
        ax.plot(
            series.index,
            series[col],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=color,
            label=col,
        )
    for i, (start, end) in enumerate(query_spans):
        ax.axvspan(
            start,
            end,
            color="crimson",
            alpha=0.18,
            label="Query dates (16-09 and 30-09)" if i == 0 else None,
        )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized 3-day rolling mean (z-score)")
    ax.set_title("Same timeline: do all 3 move the same way on 16 Sep and 30 Sep?")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_time(
    series: pd.DataFrame,
    pair_starts: pd.DatetimeIndex,
    pair_slopes: np.ndarray,
    pair_match: np.ndarray,
    pair_r: dict[tuple[str, str], float],
    query_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

    ax = axes[0]
    for col, color in CH_COLORS.items():
        ax.plot(
            series.index,
            series[col],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
            label=col,
        )
    for i, (start, end) in enumerate(query_spans):
        ax.axvspan(start, end, color="crimson", alpha=0.18)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_ylabel("z-score")
    r_txt = "  |  ".join(f"{a}–{b} r={pair_r[(a, b)]:.2f}" for a, b in PAIRS)
    ax.set_title("Normalized 3-day rolling data (shared timeline)\n" + r_txt)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for t, col in enumerate(CHANNELS):
        ax.plot(
            pair_starts,
            pair_slopes[:, t],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=CH_COLORS[col],
            label=col,
        )
    for start, end in query_spans:
        ax.axvspan(start, end, color="crimson", alpha=0.18)
    ax.scatter(
        pair_starts[pair_match],
        np.zeros(int(pair_match.sum())),
        c="green",
        s=28,
        zorder=5,
        label="All 3 same direction",
    )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Date (start of day → next day)")
    ax.set_ylabel("Day-to-day slope")
    ax.set_title("Same-time direction: slope of each channel from one day to the next")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_null_histogram(
    n_query_match: np.ndarray,
    observed_n: int,
    p_value: float,
    n_query: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(n_query + 2) - 0.5
    ax.hist(
        n_query_match,
        bins=bins,
        color="0.65",
        edgecolor="white",
        label="Shuffled timelines",
    )
    ax.axvline(
        observed_n,
        color="crimson",
        linewidth=2,
        label=f"Real data: {observed_n} / {n_query} dates match",
    )
    ax.set_xticks(range(n_query + 1))
    ax.set_xlabel("How many of the query dates (16 Sep, 30 Sep) still match after shuffling")
    ax.set_ylabel("Number of permutations")
    ax.set_title(
        f"Chance they move the same way on those same dates\n"
        f"empirical p = {p_value:.4f}"
    )
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    daily = load_daily_channels()
    series = normalized_rolling_series(daily)
    dates = series.index.to_numpy()
    Y = series.to_numpy(dtype=float).T
    pair_slopes = consecutive_day_slopes(dates, Y)
    pair_match = all_three_match(pair_slopes)
    pair_starts = pd.to_datetime(dates[:-1])
    pair_ends = pd.to_datetime(dates[1:])
    pair_r = pairwise_pearson(series)

    query_idx = [pair_index_for_start(dates, d) for d in QUERY_STARTS]
    query_spans = [
        (pair_starts[i], pair_ends[i]) for i in query_idx
    ]

    print("Working series: per-channel z-score, then 3-day rolling mean")
    print("Shared timeline:", series.index.min().date(), "→", series.index.max().date())
    print("Channels:", ", ".join(CHANNELS))
    print()

    print("Same-time direction (this day → next day on the shared timeline)")
    observed_n = 0
    for start, idx in zip(QUERY_STARTS, query_idx):
        row = pair_slopes[idx]
        matched = bool(pair_match[idx])
        observed_n += int(matched)
        if not np.isfinite(row).all():
            direction = "unknown"
        elif (row > 0).all():
            direction = "UP"
        elif (row < 0).all():
            direction = "DOWN"
        else:
            direction = "MIXED"
        print(
            f"  {pair_starts[idx].date()} → {pair_ends[idx].date()}: "
            f"{'MATCH' if matched else 'NO MATCH'} ({direction})"
        )
        for name, slope in zip(CHANNELS, row):
            way = "up" if slope > 0 else "down"
            print(f"    {name}: {slope:+.4f}  ({way})")
    print()

    print("Pairwise r on the normalized 3-day rolling series")
    print("  " + "  ".join(f"{a}–{b}={pair_r[(a, b)]:.3f}" for a, b in PAIRS))
    print()

    print_consecutive_match_list(dates, pair_slopes)

    print(f"Running {N_PERM:,} circular-shift permutations (seed={RNG_SEED})...")
    print("Dates stay fixed; only each channel's values are rotated.")
    _, match_at_query, n_query_match = permutation_null_same_times(
        dates, Y, query_idx
    )
    both = match_at_query.all(axis=1)
    p_both = empirical_pvalue(float(observed_n), n_query_match)
    chance_pct = 100.0 * p_both

    print()
    print("Chance they move the same way on those SAME dates")
    print("(shuffle each tank's series, keep 16 Sep and 30 Sep on the calendar)")
    for start, idx, j in zip(QUERY_STARTS, query_idx, range(len(query_idx))):
        n_ge = int(match_at_query[:, j].sum())
        p_date = (n_ge + 1) / (N_PERM + 1)
        print(
            f"  {pair_starts[idx].date()} → {pair_ends[idx].date()}: "
            f"all 3 match by chance in {100.0 * p_date:.2f}% of shuffles "
            f"(p = {p_date:.4f})"
        )
    n_both = int(both.sum())
    print(
        f"  both dates match together by chance in "
        f"{100.0 * (n_both + 1) / (N_PERM + 1):.2f}% of shuffles"
    )
    print()
    print(
        f"A same-direction match on these calendar dates would occur by chance "
        f"in approximately {chance_pct:.2f}% of experiments."
    )

    series_path = DATA_DIR / "permot_timeseries_window.png"
    corr_path = DATA_DIR / "permot_correlation_time.png"
    hist_path = DATA_DIR / "permot_null_histogram.png"
    plot_normalized_rolling(series, query_spans, series_path)
    plot_correlation_time(
        series,
        pair_starts,
        pair_slopes,
        pair_match,
        pair_r,
        query_spans,
        corr_path,
    )
    plot_null_histogram(
        n_query_match, observed_n, p_both, len(query_idx), hist_path
    )
    print()
    print("Saved", series_path.name)
    print("Saved", corr_path.name)
    print("Saved", hist_path.name)


if __name__ == "__main__":
    main()
