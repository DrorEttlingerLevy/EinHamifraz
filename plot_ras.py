"""Tank-averaged peak height as a centered 3-day rolling mean."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = Path("/Volumes/Expansion/features and curves 141026/features_kept.parquet")
FEED_TIMES = Path("/Volumes/Expansion/features and curves 141026/feeding_times_ein.parquet")
ROW2 = Path("/Users/dror/Downloads/row2_with_realtime.parquet")
ROW3 = Path("/Users/dror/Downloads/row3_with_realtime.parquet")
TANK_CHANNELS = {
    "Tank 1": ("Ch1", "Ch2"),
    "Tank 2": ("Ch3", "Ch4"),
    "Tank 3": ("Ch5", "Ch6"),
}
COLORS = ["#54278F", "#E31A1C", "#1B9E77"]
FONT_SIZE = 13
ROLLING_DAYS = 3
OUT = Path(__file__).resolve().parent / "plot_ras.png"
OUT_ONE = Path(__file__).resolve().parent / "plot_ras_one_channel.png"
OUT_STD = Path(__file__).resolve().parent / "plot_ras_std.png"
OUT_CH5 = Path(__file__).resolve().parent / "plot_ras_ch5.png"
OUT_FEEDING = Path(__file__).resolve().parent / "plot_ras_feeding.png"
OUT_PEAK = Path(__file__).resolve().parent / "plot_ras_peak.png"
OUT_SCALE = Path(__file__).resolve().parent / "plot_ras_scale.png"
FEEDING_VALUES = Path(__file__).resolve().parent / "feeding_values.csv"
FEEDING_CHANNELS = ["Ch2", "Ch4", "Ch6"]
FEEDING_COLORS = ["tab:green", "tab:orange", "tab:blue"]
# Row 1 is 2–9 Sep. Row 2 starts 14 Sep. Tank 3 (Ch6) ends 14 Oct.
ROW2_START = pd.Timestamp("2025-09-14")
TANK3_LAST = pd.Timestamp("2025-10-14")
REAL_MARKS = [pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-14")]
ROW_CLOCKS = {
    "row2": ROW2,
    "row3": ROW3,
}
FIRST_DAYS = 7


def tank_peaks(frame):
    pieces = []
    for name, channels in TANK_CHANNELS.items():
        pair = frame.loc[frame["channel"].isin(channels), ["feeding_datetime", "peak_height"]]
        tank = pair.groupby("feeding_datetime")["peak_height"].mean()
        tank.name = name
        pieces.append(tank)
    return pd.concat(pieces, axis=1).sort_index()


def single_channel_peaks(frame):
    """Peak height of Ch2, Ch4, and Ch6 only, one channel per tank."""
    pieces = []
    for name, column in zip(TANK_CHANNELS, FEEDING_CHANNELS):
        tank = frame.loc[frame["channel"] == column].set_index("feeding_datetime")["peak_height"]
        tank.name = name
        pieces.append(tank)
    return pd.concat(pieces, axis=1).sort_index()


def day_number(times, origin):
    elapsed = times - origin
    if isinstance(elapsed, pd.Series):
        elapsed = elapsed.dt
    return elapsed.total_seconds() / 86400.0 + 1.0


def plot_ch5_first_days():
    """Full Ch5 trace for the first week of row 2, with a red dot at each feeding."""
    frame = pd.read_parquet(ROW2, columns=["Ch5", "real time"])
    times = pd.to_datetime(frame["real time"].astype(str), format="%Y%m%d%H%M%S")
    order = times.argsort()
    times = times.iloc[order].reset_index(drop=True)
    values = frame["Ch5"].iloc[order].reset_index(drop=True)

    origin = times.iloc[0].normalize()
    end = origin + pd.Timedelta(days=FIRST_DAYS)
    keep = times < end
    day = day_number(times[keep], origin)

    feed = pd.read_parquet(FEED_TIMES, columns=["source", "index", "real_time"])
    feed = feed[feed["source"] == "row2"]
    feed_t = pd.to_datetime(feed["real_time"])
    feed = feed[(feed_t >= origin) & (feed_t < end)]
    feed_day = day_number(pd.to_datetime(feed["real_time"]), origin)
    feed_y = values.iloc[feed["index"].to_numpy()]

    signal_font = 22
    plt.rcParams.update(
        {
            "font.size": signal_font,
            "axes.labelsize": signal_font,
            "xtick.labelsize": signal_font,
            "ytick.labelsize": signal_font,
            "legend.fontsize": signal_font,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.plot(day, values[keep], color="#3949AB", linewidth=1.1, label="Aqua-MAFRI signal")
    ax.scatter(
        feed_day,
        feed_y,
        color="red",
        s=36,
        zorder=5,
        label="feeding events",
    )
    ax.set_xlim(1, FIRST_DAYS + 1)
    # Put each day number under that day's peaks, not at midnight.
    tick_at = [
        float(feed_day[feed_day.astype(int) == day_i].median())
        for day_i in range(1, FIRST_DAYS + 1)
    ]
    ax.set_xticks(tick_at)
    ax.set_xticklabels([str(day_i) for day_i in range(1, FIRST_DAYS + 1)])
    ax.set_xlabel("Day")
    ax.set_ylabel("Aqua-MAFRI")
    ax.grid(True, color="0.75", alpha=0.4)
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_CH5, dpi=150)
    print(f"Saved {OUT_CH5}")
    print(f"Ch5 samples {int(keep.sum())}  feedings {len(feed)}")


def daily_feeding():
    """Daily means of Ch2, Ch4, and Ch6, from row 2 through tank 3's last day."""
    frame = pd.read_csv(FEEDING_VALUES, parse_dates=["max_time"]).set_index("max_time")
    frame = frame.loc[ROW2_START:]
    daily = frame[FEEDING_CHANNELS].resample("D").mean().dropna(how="all")
    return daily.loc[:TANK3_LAST]


def plot_zscored_rolling(daily, ylabel, out_path):
    """Z-scored 3-day rolling mean, one line per tank, daily points behind it."""
    day = np.arange(len(daily)) + 1
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, column, color in zip(TANK_CHANNELS, daily.columns, FEEDING_COLORS):
        rolling = daily[column].rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean()
        center = rolling.mean(skipna=True)
        scale = rolling.std(skipna=True)
        ax.plot(day, (rolling - center) / scale, color=color, linewidth=3, label=name)
        ax.plot(
            day,
            (daily[column] - center) / scale,
            color=color,
            marker="o",
            linestyle="",
            alpha=0.2,
        )
    ax.set_xticks(day)
    ax.set_xticklabels([str(i) for i in day])
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"{ylabel} days {len(daily)}")


def plot_normalized_feeding():
    """Z-scored 3-day rolling feeding intensity from feeding_values.csv."""
    plot_zscored_rolling(daily_feeding(), "Normalized feeding intensity", OUT_FEEDING)


def daily_peak_height():
    """Daily mean peak height of Ch2, Ch4, and Ch6 from the features table."""
    frame = pd.read_parquet(FEATURES, columns=["channel", "feeding_datetime", "peak_height"])
    frame = frame[frame["channel"].isin(FEEDING_CHANNELS)]
    daily = frame.pivot_table(
        index="feeding_datetime",
        columns="channel",
        values="peak_height",
    )
    daily = daily[FEEDING_CHANNELS].resample("D").mean().dropna(how="all")
    # Features are row 2 and row 3 only. Stop where tank 3 (Ch6) stops.
    last = daily["Ch6"].last_valid_index()
    return daily.loc[:last]


def plot_normalized_peak():
    """Same rolling z-score, using peak height from the features table."""
    plot_zscored_rolling(daily_peak_height(), "Normalized peak height", OUT_PEAK)


def recording_scale():
    """0.5th and 99.9th percentile of each channel inside row 2 and inside row 3."""
    bounds = []
    for path in (ROW2, ROW3):
        frame = pd.read_parquet(path, columns=["real time", *FEEDING_CHANNELS])
        times = pd.to_datetime(frame["real time"].astype(str), format="%Y%m%d%H%M%S")
        limits = {}
        for column in FEEDING_CHANNELS:
            finite = frame[column].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            limits[column] = (
                float(np.percentile(finite, 0.5)),
                float(np.percentile(finite, 99.9)),
            )
        bounds.append((times.min(), times.max(), limits))
    return bounds


def scaled_feeding_daily():
    """Feeding counts clipped to 0–1 inside the recording they belong to."""
    frame = pd.read_csv(FEEDING_VALUES, parse_dates=["max_time"]).set_index("max_time")
    frame = frame.loc[ROW2_START:TANK3_LAST, FEEDING_CHANNELS]
    bounds = recording_scale()
    scaled = pd.DataFrame(index=frame.index, columns=FEEDING_CHANNELS, dtype=float)
    for column in FEEDING_CHANNELS:
        values = []
        for stamp, value in zip(frame.index, frame[column]):
            limits = next(item[column] for start, end, item in bounds if start <= stamp <= end)
            low, high = limits
            values.append(float(np.clip((value - low) / (high - low), 0.0, 1.0)))
        scaled[column] = values
    return scaled.resample("D").mean().dropna(how="all")


def draw_zscored(ax, daily):
    day = np.arange(len(daily)) + 1
    for name, column, color in zip(TANK_CHANNELS, daily.columns, FEEDING_COLORS):
        rolling = daily[column].rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean()
        center = rolling.mean(skipna=True)
        scale = rolling.std(skipna=True)
        ax.plot(day, (rolling - center) / scale, color=color, linewidth=3, label=name)
        ax.plot(
            day,
            (daily[column] - center) / scale,
            color=color,
            marker="o",
            linestyle="",
            alpha=0.35,
        )
    ax.set_xticks(day)
    ax.set_xticklabels([str(i) for i in day])
    ax.grid(True, alpha=0.3)
    return day


def plot_scale_push():
    """Raw feeding z-scores beside the same days after each row is scaled 0–1."""
    raw = daily_feeding()
    scaled = scaled_feeding_daily()
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
        }
    )
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    draw_zscored(axes[0], raw)
    axes[0].set_ylabel("Normalized feeding intensity")
    axes[0].set_title("Raw counts")
    axes[0].legend()
    draw_zscored(axes[1], scaled)
    axes[1].set_ylabel("Normalized feeding intensity")
    axes[1].set_xlabel("Day")
    axes[1].set_title("Same counts, scaled 0–1 inside each row")
    fig.tight_layout()
    fig.savefig(OUT_SCALE, dpi=150)
    print(f"Saved {OUT_SCALE}")


def feeding_real_times():
    """Map each shifted feeding label back to the real clock, before the day alignment."""
    feed = pd.read_parquet(FEED_TIMES, columns=["source", "index", "time"])
    feed = feed[feed["source"].isin(ROW_CLOCKS)].copy()
    clocks = {}
    for name, path in ROW_CLOCKS.items():
        raw = pd.read_parquet(path, columns=["real time"])
        times = pd.to_datetime(raw["real time"].astype(str), format="%Y%m%d%H%M%S")
        clocks[name] = times.sort_values().reset_index(drop=True)
    feed["real_t"] = [
        clocks[row.source].iloc[int(row.index)] for row in feed.itertuples(index=False)
    ]
    feed["label"] = pd.to_datetime(feed["time"])
    return feed.drop_duplicates("label").set_index("label")["real_t"]


def real_mark_days(origin):
    """Day numbers on the plot for feedings that happened on the real dates."""
    feed = feeding_real_times().reset_index()
    marks = []
    for real_day in REAL_MARKS:
        labels = feed.loc[feed["real_t"].dt.normalize() == real_day, "label"]
        if labels.empty:
            continue
        marks.append((real_day.strftime("%d %b"), float(day_number(labels, origin).median())))
    return marks


def plot_daily_peaks(peaks, out_path, statistic="mean", mark_real=False, show_rolling=True, on_real_time=False):
    """Daily peak statistic, with a centered 3-day rolling mean unless turned off."""
    if on_real_time:
        peaks = peaks.copy()
        peaks.index = pd.to_datetime(peaks.index).map(feeding_real_times())
        peaks = peaks.sort_index()
    origin = peaks.index.min().normalize()
    day = day_number(peaks.index, origin)
    daily = getattr(peaks.resample("D"), statistic)()
    rolling = daily.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS, center=True).mean()
    # Sit each day's value at midday, among that day's feedings.
    daily_day = day_number(daily.index + pd.Timedelta(hours=12), origin)
    roll_day = day_number(rolling.index + pd.Timedelta(hours=12), origin)
    # Real calendar span, so the empty day between row 2 and row 3 stays on the axis.
    if on_real_time:
        span = int((peaks.index.max().normalize() - origin).days) + 1
    else:
        span = int(np.ceil(day.max()))

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for name, color in zip(peaks.columns, COLORS):
        line_day = roll_day if show_rolling else daily_day
        line = rolling[name] if show_rolling else daily[name]
        ax.plot(line_day, line, color=color, linewidth=2.4, label=name, zorder=2)
        ax.scatter(daily_day, daily[name], color=color, s=28, zorder=3)
    right = span + 1 if on_real_time else span
    ax.set_xlim(0.5, right + 0.5)
    ax.set_xticks(range(1, right + 1))
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized Aqua-MAFRI" if statistic == "mean" else "Std of Aqua-MAFRI")
    if mark_real:
        for label, position in real_mark_days(origin):
            ax.axvline(
                position,
                color="red",
                linestyle="--",
                linewidth=1.6,
                label=label,
            )
    ax.grid(True, color="0.75", alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"Peaks {len(peaks)}  rolling days {int(rolling.dropna().shape[0])}")


def main():
    frame = pd.read_parquet(FEATURES, columns=["channel", "feeding_datetime", "peak_height"])
    plot_daily_peaks(tank_peaks(frame), OUT, show_rolling=False, on_real_time=True)
    plot_daily_peaks(tank_peaks(frame), OUT_STD, statistic="std")
    plot_daily_peaks(single_channel_peaks(frame), OUT_ONE)
    plot_ch5_first_days()
    plot_normalized_feeding()
    plot_normalized_peak()
    plot_scale_push()


if __name__ == "__main__":
    main()
