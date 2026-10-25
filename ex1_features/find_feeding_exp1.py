import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIGNAL_PATH = "/Volumes/My Book/exp_1/sw/thr_0.01.parquet"
OUT_PATH = "/Volumes/My Book/exp_1/sw/feeding_times_exp1.parquet"


PAD = 100
HEAD = 10_000
FIRST_FEED = pd.Timedelta(hours=8)
DAY_GAP = pd.Timedelta(hours=10)


def ranges_above(above, pad=PAD):
    """Runs above threshold, each extended by `pad` samples on both sides."""
    n = len(above)
    spans = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i + 1
        while j < n and above[j]:
            j += 1
        spans.append((max(0, i - pad), min(n, j + pad)))
        i = j
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def align_first_to_8(times):
    """Each run of feedings is its own day. A gap longer than DAY_GAP starts the next day.

    The first feeding of that day is 08:00, and the later ones keep their spacing from it.
    """
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    if times.empty:
        return times
    day = (times.diff() > DAY_GAP).cumsum()
    start = times.iloc[0].normalize()
    aligned = times.copy()
    for d, idx in times.groupby(day).groups.items():
        group = times.loc[idx]
        anchor = start + pd.Timedelta(days=int(d)) + FIRST_FEED
        aligned.loc[idx] = anchor + (group - group.iloc[0])
    return aligned


def main():
    threshold = 190
    threshold_head = 160
    signal = pd.read_parquet(SIGNAL_PATH, columns=["Ch5", "time"])
    signal["t"] = pd.to_datetime(signal["time"])
    ch5 = signal["Ch5"].to_numpy()

    above = ch5 > threshold
    head_n = min(HEAD, len(ch5))
    above[:head_n] = ch5[:head_n] > threshold_head
    spans = ranges_above(above)
    peaks = []
    for start, end in spans:
        k = start + int(np.argmax(ch5[start:end]))
        peaks.append((k, signal["t"].iloc[k], float(ch5[k])))
    feed = pd.DataFrame(
        {
            "index": [k for k, _, _ in peaks],
            "time": align_first_to_8([t for _, t, _ in peaks]),
        }
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(signal["t"], signal["Ch5"], linewidth=0.4, color="C0", label="Ch5")
    thr_line = np.full(len(ch5), threshold, dtype=float)
    thr_line[:head_n] = threshold_head
    ax.plot(
        signal["t"],
        thr_line,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"threshold {threshold_head} / {threshold}",
    )
    if peaks:
        ax.scatter(
            [t for _, t, _ in peaks],
            [y for _, _, y in peaks],
            color="red",
            s=18,
            zorder=5,
            label="peak",
        )
    ax.set_title(
        f"Ch5 above {threshold_head} then {threshold}: "
        f"{len(spans)} ranges, {len(feed)} peaks (±{PAD})"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("Ch5")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

    answer = input(f"Save {len(feed)} peak indexes? [y/n]: ").strip().lower()
    if answer == "y":
        feed.to_parquet(OUT_PATH, index=False)
        print(f"{len(feed)} peak indexes written to {OUT_PATH}")
    else:
        print("Not saved.")


if __name__ == "__main__":
    main()
