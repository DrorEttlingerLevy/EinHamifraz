import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SIGNAL_PATHS = {
    "row1": "/Users/dror/Downloads/row1_with_realtime.parquet",
    "row2": "/Users/dror/Downloads/row2_with_realtime.parquet",
    "row3": "/Users/dror/Downloads/row3_with_realtime.parquet",
}
FEED_PATH = "/Volumes/Expansion/features and curves 141026/feeding_times_ein.parquet"
CHANNELS = [f"Ch{i}" for i in range(1, 7)]
# Row 1 is stored as Ch0–Ch5. Those are the same sensors as Ch1–Ch6 in row 2 and row 3.
ROW1_CHANNEL_MAP = {f"Ch{i}": f"Ch{i + 1}" for i in range(6)}


def to_datetime(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S")


feed = pd.read_parquet(FEED_PATH, columns=["source", "index", "real_time", "time"])
feed["t"] = pd.to_datetime(feed["time"])

signals = {}
for name, path in SIGNAL_PATHS.items():
    signal = pd.read_parquet(path)
    if name == "row1":
        signal = signal.rename(columns=ROW1_CHANNEL_MAP)
    signal["t"] = to_datetime(signal["real time"])
    signals[name] = signal.sort_values("t").reset_index(drop=True)

peaks = []
for row in feed.itertuples(index=False):
    signal = signals[row.source]
    peaks.append(
        {
            "source": row.source,
            "index": row.index,
            "real_t": signal["t"].iloc[row.index],
            "label": row.t,
        }
    )
peaks = pd.DataFrame(peaks)

row3 = signals["row3"]
row3_mid = row3["t"].iloc[0] + (row3["t"].iloc[-1] - row3["t"].iloc[0]) / 2
panels = [
    ("row1", "row1", signals["row1"]),
    ("row2", "row2", signals["row2"]),
    ("row3, first half", "row3", row3[row3["t"] < row3_mid]),
    ("row3, second half", "row3", row3[row3["t"] >= row3_mid]),
]

fig, axes = plt.subplots(
    len(CHANNELS),
    len(panels),
    figsize=(16, 16),
    sharex="col",
    sharey="row",
)
for col, (title, source, signal) in enumerate(panels):
    start = signal["t"].iloc[0]
    end = signal["t"].iloc[-1]
    in_panel = peaks[
        (peaks["source"] == source)
        & (peaks["real_t"] >= start)
        & (peaks["real_t"] <= end)
    ]
    full = signals[source]
    for row_i, channel in enumerate(CHANNELS):
        ax = axes[row_i, col]
        if channel not in signal.columns:
            ax.set_visible(False)
            continue
        ax.plot(signal["t"], signal[channel], linewidth=0.6, color="C0")
        y = full[channel].iloc[in_panel["index"].to_numpy()]
        ax.scatter(in_panel["real_t"], y, color="red", s=12, zorder=5)
        if channel == "Ch5":
            for real_t, value, label in zip(in_panel["real_t"], y, in_panel["label"]):
                ax.annotate(
                    label.strftime("%d %H:%M"),
                    (real_t, value),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="left",
                    va="bottom",
                    fontsize=5,
                    rotation=90,
                    rotation_mode="anchor",
                    color="red",
                )
        ax.set_xlim(start, end)
        ax.grid(True, alpha=0.3)
        if row_i == 0:
            ax.set_title(
                f"{title}\n{start:%m-%d %H:%M} to {end:%m-%d %H:%M} ({len(in_panel)})",
                fontsize=9,
            )
        if col == 0:
            ax.set_ylabel(channel)
        if row_i == len(CHANNELS) - 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.set_xlabel("real time")

fig.suptitle(f"All channels with {len(feed)} feeding times", y=0.995)
fig.tight_layout()
plt.show()
