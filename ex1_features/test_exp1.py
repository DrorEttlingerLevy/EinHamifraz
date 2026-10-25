import matplotlib.pyplot as plt
import pandas as pd

BASE = "/Volumes/My Book/exp_1/sw"

signal = pd.read_parquet(f"{BASE}/thr_0.01.parquet", columns=["Ch5", "time"])
feed = pd.read_parquet(f"{BASE}/feeding_times_exp1.parquet", columns=["index", "time"])

signal["t"] = pd.to_datetime(signal["time"])
feed["t"] = pd.to_datetime(feed["time"])
peak_t = signal["t"].iloc[feed["index"].to_numpy()]
ch5_at_feed = signal["Ch5"].iloc[feed["index"].to_numpy()]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(signal["t"], signal["Ch5"], linewidth=0.4, color="C0", label="Ch5")
ax.scatter(peak_t, ch5_at_feed, color="red", s=18, zorder=5, label="feeding")
for t, x, y in zip(feed["t"], peak_t, ch5_at_feed):
    ax.annotate(
        t.strftime("%d %H:%M"),
        (x, y),
        textcoords="offset points",
        xytext=(0, 4),
        ha="left",
        va="bottom",
        fontsize=7,
        rotation=90,
        rotation_mode="anchor",
        color="red",
    )
ax.set_title(f"Ch5 with {len(feed)} feeding times")
ax.set_xlabel("real time")
ax.set_ylabel("Ch5")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()
