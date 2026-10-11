import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = r"H:\Exp_Ein_Hamifratz\output_111026"

df1 = pd.read_parquet(base + r"\feeding_times_row1.parquet")
df2 = pd.read_parquet(base + r"\feeding_times_row2.parquet")
df3 = pd.read_parquet(base + r"\feeding_times_row3.parquet")

# Stack row 1, then row 2, then row 3, one above the other
combined_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

out = base + r"\combined_df.parquet"
combined_df.to_parquet(out)

print(combined_df.shape)
print(out)
print(combined_df.head())

half = pd.Timedelta(minutes=30)
ncols = 6
rows = [
    (1, df1, base + r"\row1_with_realtime_0.005.parquet"),
    (2, df2, base + r"\row2_with_realtime_0.005.parquet"),
    (3, df3, base + r"\row3_with_realtime_0.005.parquet"),
]


def to_datetime(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S")


for row_id, feed, signal_path in rows:
    signal = pd.read_parquet(signal_path, columns=["Ch5", "real time"])
    signal["t"] = to_datetime(signal["real time"])
    signal = signal.sort_values("t")
    times = signal["t"].to_numpy(dtype="datetime64[ns]")
    Ch5 = signal["Ch5"].to_numpy()

    feed_times = to_datetime(feed["real time"])
    n = len(feed_times)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 2.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, ft in enumerate(feed_times):
        ax = axes[i // ncols, i % ncols]
        ft64 = np.datetime64(pd.Timestamp(ft), "ns")
        start = ft64 - np.timedelta64(30, "m")
        end = ft64 + np.timedelta64(30, "m")
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, end, side="right")
        window_t = times[lo:hi]
        if len(window_t):
            minutes = (window_t - ft64) / np.timedelta64(1, "m")
            ax.plot(minutes, Ch5[lo:hi], linewidth=0.8)
        ax.axvline(0, color="red", linewidth=0.8)
        ax.set_xlim(-30, 30)
        ax.set_xticks([-30, 0, 30])
        ax.set_title(pd.Timestamp(ft).strftime("%m-%d %H:%M"), fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)


    for ax in axes[:, 0]:
        ax.set_ylabel("Ch5")
    for ax in axes[-1, :]:
        ax.set_xlabel("minutes from feeding")

    fig.suptitle(f"Row {row_id} Ch5, 30 min before and after each feeding")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = base + rf"\Ch5_feeding_pm30_row{row_id}.png"
    fig.savefig(png, dpi=150)
    print(png)

plt.show()
