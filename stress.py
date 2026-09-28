from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

WINDOWS = [
    ("06:55", "07:50"),
    ("09:55", "10:40"),
    ("11:55", "12:50"),
    ("14:55", "15:50"),
    ("17:55", "18:50"),
]

df = pd.read_parquet(DATA_DIR / "combiend.parquet")
df["real_time_dt"] = pd.to_datetime(df["real_time_dt"])
df = df.sort_values("real_time_dt")
first_days = df["real_time_dt"].dt.normalize().drop_duplicates().iloc[:3]
df = df[df["real_time_dt"].dt.normalize().isin(first_days)]

plot_cols = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"]
tod = df["real_time_dt"].dt.hour * 60 + df["real_time_dt"].dt.minute
day = df["real_time_dt"].dt.normalize()

fig, axes = plt.subplots(len(plot_cols), 1, sharex=True, figsize=(14, 16))

for ax, col in zip(axes, plot_cols):
    for source, group in df.groupby("source", sort=False):
        ax.plot(group["real_time_dt"], group[col], label=source, linewidth=0.6)

    peak_times = []
    peak_vals = []
    for start_s, end_s in WINDOWS:
        start = datetime.strptime(start_s, "%H:%M")
        end = datetime.strptime(end_s, "%H:%M")
        start_m = start.hour * 60 + start.minute
        end_m = end.hour * 60 + end.minute
        mask = (tod >= start_m) & (tod <= end_m)
        window = df.loc[mask, [col]].assign(_day=day[mask], _t=df.loc[mask, "real_time_dt"])
        if window.empty:
            continue
        idx = window.groupby("_day")[col].idxmax()
        peaks = window.loc[idx]
        peak_times.append(peaks["_t"])
        peak_vals.append(peaks[col])

    if peak_times:
        ax.scatter(
            pd.concat(peak_times),
            pd.concat(peak_vals),
            c="red",
            s=18,
            zorder=5,
            label="window max",
        )

    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)

axes[0].legend(loc="upper right")
axes[-1].set_xlabel("real time")
fig.suptitle("First 3 days — window maxima (red)", y=0.995)
fig.tight_layout()
fig.savefig(DATA_DIR / "stress_window_maxima.png", dpi=150)
plt.show()
