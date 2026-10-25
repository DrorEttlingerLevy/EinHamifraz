import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same channels, three recordings. `real time` is the clock; row 2 and row 3
# also have a `time` column that restarts, so it is not used. The stretches
# are Sep 2–10, Sep 14–25, and Sep 26–Oct 22, with a gap between each pair.
SIGNAL_PATHS = [
    "/Users/dror/Downloads/row1_with_realtime.parquet",
    "/Users/dror/Downloads/row2_with_realtime.parquet",
    "/Users/dror/Downloads/row3_with_realtime.parquet",
]
OUT_PATH = "/Volumes/Expansion/features and curves 141026/feeding_times_ein.parquet"

PAD = 100
HEAD = 10_000
FIRST_FEED = pd.Timedelta(hours=8)
DAY_GAP = pd.Timedelta(hours=10)
# The baseline jumps after each gap, so one level does not fit all three.
# Each value sits where raising it further stops splitting one feeding.
THRESHOLDS = (300, 1000, 800)
THRESHOLD_HEAD = 300
# Row 1 is stored as Ch0–Ch5. Those are the same sensors as Ch1–Ch6 in row 2 and row 3.
ROW1_CHANNEL_MAP = {f"Ch{i}": f"Ch{i + 1}" for i in range(6)}


def to_datetime(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S")


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


def load_segments():
    """One frame per file, on real time. Detection never crosses a file gap."""
    segments = []
    for path, threshold in zip(SIGNAL_PATHS, THRESHOLDS):
        name = path.rsplit("/", 1)[-1].replace("_with_realtime.parquet", "")
        frame = pd.read_parquet(path)
        if name == "row1":
            frame = frame.rename(columns=ROW1_CHANNEL_MAP)
        frame = frame[["Ch5", "real time"]].copy()
        frame["t"] = to_datetime(frame["real time"])
        frame = frame.sort_values("t").reset_index(drop=True)
        segments.append(
            {
                "name": name,
                "t": frame["t"],
                "ch5": frame["Ch5"].to_numpy(),
                "threshold": threshold,
            }
        )
    return segments


def peaks_in_segment(segment, lower_head):
    ch5 = segment["ch5"]
    above = ch5 > segment["threshold"]
    head_n = 0
    if lower_head:
        head_n = min(HEAD, len(ch5))
        above[:head_n] = ch5[:head_n] > THRESHOLD_HEAD
    peaks = []
    for start, end in ranges_above(above):
        k = start + int(np.argmax(ch5[start:end]))
        peaks.append((k, segment["t"].iloc[k], float(ch5[k]), segment["name"]))
    return peaks, head_n


def main():
    segments = load_segments()
    peaks = []
    head_n = 0
    for i, segment in enumerate(segments):
        found, head_n_i = peaks_in_segment(segment, lower_head=(i == 0))
        if i == 0:
            head_n = head_n_i
        peaks.extend(found)
    peaks.sort(key=lambda item: item[1])

    real_times = [t for _, t, _, _ in peaks]
    feed = pd.DataFrame(
        {
            "source": [name for _, _, _, name in peaks],
            "index": [k for k, _, _, _ in peaks],
            "real_time": real_times,
            "time": align_first_to_8(real_times),
            # Raw Ch5 at the detected peak. The feature table uses a 0-1 curve.
            "real_peak": [y for _, _, y, _ in peaks],
        }
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    for segment in segments:
        ax.plot(
            segment["t"],
            segment["ch5"],
            linewidth=0.4,
            label=f"{segment['name']} Ch5",
        )
    for left, right in zip(segments, segments[1:]):
        gap_start = left["t"].iloc[-1]
        gap_end = right["t"].iloc[0]
        ax.axvspan(gap_start, gap_end, color="0.85", zorder=0)
    for segment, lower_head in zip(segments, (True, False, False)):
        thr_line = np.full(len(segment["ch5"]), segment["threshold"], dtype=float)
        if lower_head:
            thr_line[:head_n] = THRESHOLD_HEAD
        ax.plot(
            segment["t"],
            thr_line,
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )
    if peaks:
        ax.scatter(
            [t for _, t, _, _ in peaks],
            [y for _, _, y, _ in peaks],
            color="red",
            s=18,
            zorder=5,
            label="peak",
        )
    counts = feed["source"].value_counts() if len(feed) else {}
    count_text = ", ".join(f"{segment['name']} {int(counts.get(segment['name'], 0))}" for segment in segments)
    ax.set_title(f"Ch5 feedings across 3 recordings ({count_text}), gaps shaded")
    ax.set_xlabel("real time")
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
