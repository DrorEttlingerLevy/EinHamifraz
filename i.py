from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

row1 = pd.read_parquet(DATA_DIR / "row1_with_realtime.parquet")
row2 = pd.read_parquet(DATA_DIR / "row2_with_realtime.parquet")
row3 = pd.read_parquet(DATA_DIR / "row3_with_realtime.parquet")

df = pd.concat(
    [row1.assign(source="row1"), row2.assign(source="row2"), row3.assign(source="row3")],
    ignore_index=True,
    sort=False,
)

df["real_time_dt"] = pd.to_datetime(df["real time"].astype(str), format="%Y%m%d%H%M%S")
df = df.sort_values("real_time_dt")

plot_cols = [ "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "time"]
fig, axes = plt.subplots(len(plot_cols), 1, sharex=True, figsize=(14, 18))

for ax, col in zip(axes, plot_cols):
    for source, group in df.groupby("source", sort=False):
        ax.plot(group["real_time_dt"], group[col], label=source, linewidth=0.6)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)

axes[0].legend(loc="upper right")
axes[-1].set_xlabel("real time")
fig.suptitle("Combined rows — all columns", y=0.995)
fig.tight_layout()
fig.savefig(DATA_DIR / "combined_columns.png", dpi=150)
plt.show()

df.to_parquet(DATA_DIR / "combiend.parquet")