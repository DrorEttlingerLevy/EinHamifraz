import matplotlib.pyplot as plt
import pandas as pd

BASE = r"H:\Exp_Ein_Hamifratz\output_111026"
SIGNAL_PATH = BASE + r"\combined_0.001.parquet"
FEED_PATH = BASE + r"\feeding_times_ch5.parquet"


def main():
    signal = pd.read_parquet(SIGNAL_PATH, columns=["Ch5", "real time"])
    feed = pd.read_parquet(FEED_PATH, columns=["real time"])

    signal["t"] = pd.to_datetime(signal["real time"].astype(str), format="%Y%m%d%H%M%S")
    feed["t"] = pd.to_datetime(feed["real time"].astype(str), format="%Y%m%d%H%M%S")

    ch5_at_feed = signal.set_index("t")["Ch5"].loc[feed["t"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(signal["t"], signal["Ch5"], linewidth=0.4, color="C0", label="Ch5")
    ax.scatter(feed["t"], ch5_at_feed, color="red", s=18, zorder=5, label="feeding")
    ax.set_title(f"Ch5 with {len(feed)} feeding times")
    ax.set_xlabel("real time")
    ax.set_ylabel("Ch5")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
