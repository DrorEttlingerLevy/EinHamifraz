"""Per-channel threshold with the best recall on row 2 and row 3.

Only Ch2, Ch4, and Ch6 are scored, and only through each row's last feeding
time. A detection is the highest sample in a contiguous stretch above the
threshold. Recall is the share of feeding times that get such a peak. When
several levels share that recall, the higher precision, then the higher
level, is kept.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIGNAL_PATHS = [
    ("row2", "/Users/dror/Downloads/row2_with_realtime.parquet"),
    ("row3", "/Users/dror/Downloads/row3_with_realtime.parquet"),
]
FEED_PATH = "/Volumes/Expansion/features and curves 141026/feeding_times_ein.parquet"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "channel_thresholds_recall.csv"
CHANNELS = ["Ch2", "Ch4", "Ch6"]
MATCH_MINUTES = 30
N_CANDIDATES = 200


def to_datetime(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S")


def load_rows():
    rows = {}
    for name, path in SIGNAL_PATHS:
        frame = pd.read_parquet(path)
        frame["t"] = to_datetime(frame["real time"])
        rows[name] = frame.sort_values("t").reset_index(drop=True)
    return rows


def load_labels():
    feed = pd.read_parquet(FEED_PATH, columns=["source", "real_time"])
    feed = feed[feed["source"].isin([name for name, _ in SIGNAL_PATHS])].copy()
    feed["t"] = pd.to_datetime(feed["real_time"])
    return feed


def clip_to_last_feeding(rows, labels):
    """Keep each row only through its last labeled feeding."""
    clipped = {}
    for name, frame in rows.items():
        last = labels.loc[labels["source"] == name, "t"].max()
        clipped[name] = frame.loc[frame["t"] <= last].reset_index(drop=True)
    return clipped


def peak_times(values, times_ns, threshold):
    """Time and height of the highest sample in each stretch above `threshold`."""
    above = values > threshold
    if not above.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    index = np.empty(len(starts), dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        index[i] = start + int(np.argmax(values[start:end]))
    return times_ns[index], values[index]


def match_flags(peaks_ns, labels_ns, window_ns):
    """Which peaks and labels form a one-to-one match within `window_ns`."""
    peak_hit = np.zeros(len(peaks_ns), dtype=bool)
    label_hit = np.zeros(len(labels_ns), dtype=bool)
    if len(peaks_ns) == 0 or len(labels_ns) == 0:
        return peak_hit, label_hit
    order = np.argsort(peaks_ns)
    sorted_peaks = peaks_ns[order]
    pairs = []
    for label_i, label in enumerate(labels_ns):
        left = np.searchsorted(sorted_peaks, label - window_ns, side="left")
        right = np.searchsorted(sorted_peaks, label + window_ns, side="right")
        for sorted_i in range(left, right):
            peak_i = int(order[sorted_i])
            pairs.append((abs(int(peaks_ns[peak_i] - label)), label_i, peak_i))
    pairs.sort()
    for _, label_i, peak_i in pairs:
        if label_hit[label_i] or peak_hit[peak_i]:
            continue
        label_hit[label_i] = True
        peak_hit[peak_i] = True
    return peak_hit, label_hit


def match_counts(peaks_ns, labels_ns, window_ns):
    """True detections, peaks outside feeding times, and missed feedings."""
    peak_hit, label_hit = match_flags(peaks_ns, labels_ns, window_ns)
    true_pos = int(label_hit.sum())
    false_pos = int(len(peaks_ns) - peak_hit.sum())
    false_neg = int(len(labels_ns) - label_hit.sum())
    return true_pos, false_pos, false_neg


def candidate_levels(values):
    low, high = np.quantile(values, [0.10, 0.999])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.array([low], dtype=float)
    return np.unique(np.linspace(low, high, N_CANDIDATES))


def score_channel(rows, labels, channel, window_ns):
    series = []
    label_ns = {}
    for name, frame in rows.items():
        times_ns = frame["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        values = frame[channel].to_numpy(dtype=float)
        series.append((times_ns, values))
        in_row = labels.loc[labels["source"] == name, "t"]
        label_ns[name] = in_row.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    levels = candidate_levels(np.concatenate([values for _, values in series]))
    best = None
    for level in levels:
        true_pos = false_pos = false_neg = 0
        for (times_ns, values), name in zip(series, rows):
            peaks, _ = peak_times(values, times_ns, level)
            tp, fp, fn = match_counts(peaks, label_ns[name], window_ns)
            true_pos += tp
            false_pos += fp
            false_neg += fn
        n_feedings = true_pos + false_neg
        n_peaks = true_pos + false_pos
        recall = true_pos / n_feedings if n_feedings else 0.0
        precision = true_pos / n_peaks if n_peaks else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        # Highest recall, then higher precision, then the higher level.
        rank = (recall, precision, float(level))
        if best is None or rank > best[0]:
            best = (
                rank,
                {
                    "channel": channel,
                    "threshold": float(level),
                    "f1": f1,
                    "recall": recall,
                    "precision": precision,
                    "detected_feedings": true_pos,
                    "missed_feedings": false_neg,
                    "false_peaks": false_pos,
                    "n_feedings": n_feedings,
                    "n_peaks": n_peaks,
                },
            )
    return best[1]


def signal_at(times_ns, values, when_ns):
    """Sample nearest to each label time, for marking a miss on the trace."""
    if len(when_ns) == 0:
        return np.empty(0, dtype=float)
    pos = np.searchsorted(times_ns, when_ns)
    pos = np.clip(pos, 0, len(times_ns) - 1)
    prev = np.clip(pos - 1, 0, len(times_ns) - 1)
    take_prev = np.abs(times_ns[prev] - when_ns) < np.abs(times_ns[pos] - when_ns)
    pos = np.where(take_prev, prev, pos)
    return values[pos]


def plot_channel(rows, labels, channel, summary, window_ns):
    """Row 2, then row 3 split in half, with the chosen level and its peaks."""
    threshold = summary["threshold"]
    row3 = rows["row3"]
    mid = row3["t"].iloc[0] + (row3["t"].iloc[-1] - row3["t"].iloc[0]) / 2
    panels = [
        ("row2", "row2", rows["row2"]),
        ("row3, first half", "row3", row3[row3["t"] < mid]),
        ("row3, second half", "row3", row3[row3["t"] >= mid]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    for ax, (title, source, signal) in zip(axes, panels):
        times_ns = signal["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        values = signal[channel].to_numpy(dtype=float)
        full = rows[source]
        full_ns = full["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        full_values = full[channel].to_numpy(dtype=float)
        peaks_ns, peak_y = peak_times(full_values, full_ns, threshold)
        label_ns = (
            labels.loc[labels["source"] == source, "t"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        peak_hit, label_hit = match_flags(peaks_ns, label_ns, window_ns)
        start = times_ns[0]
        end = times_ns[-1]
        in_panel = (peaks_ns >= start) & (peaks_ns <= end)
        labels_in = (label_ns >= start) & (label_ns <= end)
        ax.plot(signal["t"], values, color="C0", linewidth=0.5)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
        matched = in_panel & peak_hit
        outside = in_panel & ~peak_hit
        if outside.any():
            ax.scatter(
                pd.to_datetime(peaks_ns[outside]),
                peak_y[outside],
                s=8,
                color="C3",
                zorder=4,
                label="peak outside feeding",
            )
        if matched.any():
            ax.scatter(
                pd.to_datetime(peaks_ns[matched]),
                peak_y[matched],
                s=16,
                color="C2",
                zorder=5,
                label="peak on feeding",
            )
        missed = labels_in & ~label_hit
        if missed.any():
            ax.scatter(
                pd.to_datetime(label_ns[missed]),
                signal_at(full_ns, full_values, label_ns[missed]),
                s=28,
                marker="x",
                color="black",
                zorder=6,
                label="feeding missed",
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel(channel)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{channel} threshold {threshold:.0f}  "
        f"recall {summary['recall']:.3f}  "
        f"precision {summary['precision']:.3f}  "
        f"F1 {summary['f1']:.3f}"
    )
    fig.tight_layout()
    path = OUT_DIR / f"threshold_recall_{channel}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    labels = load_labels()
    rows = clip_to_last_feeding(load_rows(), labels)
    for name, frame in rows.items():
        n_labels = int((labels["source"] == name).sum())
        print(f"{name}: {len(frame)} samples through {frame['t'].iloc[-1]} ({n_labels} feedings)")
    window_ns = int(pd.Timedelta(minutes=MATCH_MINUTES).value)
    results = [score_channel(rows, labels, channel, window_ns) for channel in CHANNELS]
    table = pd.DataFrame(results)
    table.to_csv(OUT_PATH, index=False)
    for row in results:
        print(
            f"{row['channel']}: threshold {row['threshold']:.1f}  "
            f"recall {row['recall']:.3f} ({row['detected_feedings']}/{row['n_feedings']}, "
            f"{row['missed_feedings']} missed)  "
            f"precision {row['precision']:.3f}  "
            f"F1 {row['f1']:.3f}  "
            f"({row['false_peaks']} peaks outside feeding times)"
        )
        path = plot_channel(rows, labels, row["channel"], row, window_ns)
        print(f"Wrote {path}")
    print(f"Wrote {OUT_PATH}")


F1_PATH = OUT_DIR / "channel_thresholds.csv"


def window_snr(times_ns, values, when_ns, window_ns):
    """Peak SNR in the same +/- 30 min window used for a match.

    Baseline is the lower of the mean of the first and last 5% of the
    window. Sigma is the standard deviation of those edge samples.
    SNR is (window maximum - baseline) / sigma.
    """
    snr = np.full(len(when_ns), np.nan)
    for i, when in enumerate(when_ns):
        lo = np.searchsorted(times_ns, when - window_ns, side="left")
        hi = np.searchsorted(times_ns, when + window_ns, side="right")
        y = values[lo:hi]
        y = y[np.isfinite(y)]
        if y.size < 10:
            continue
        n_base = max(1, int(0.05 * y.size))
        edges = np.concatenate([y[:n_base], y[-n_base:]])
        baseline = min(float(np.mean(y[:n_base])), float(np.mean(y[-n_base:])))
        sigma = float(np.std(edges, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        snr[i] = (float(np.max(y)) - baseline) / sigma
    return snr


def f1_snr_frame(rows, labels, window_ns):
    """SNR of every feeding at the saved F1-maximizing threshold."""
    thresholds = pd.read_csv(F1_PATH).set_index("channel")["threshold"]
    records = []
    for channel, level in thresholds.items():
        level = float(level)
        for name, frame in rows.items():
            times_ns = frame["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            values = frame[channel].to_numpy(dtype=float)
            label_ns = (
                labels.loc[labels["source"] == name, "t"]
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            peaks_ns, _ = peak_times(values, times_ns, level)
            _, label_hit = match_flags(peaks_ns, label_ns, window_ns)
            snr = window_snr(times_ns, values, label_ns, window_ns)
            for hit, value in zip(label_hit, snr):
                records.append(
                    {
                        "channel": channel,
                        "source": name,
                        "snr": value,
                        "missed": not bool(hit),
                    }
                )
    return pd.DataFrame(records)


def plot_snr_histograms(frame, path, missed_only):
    """One SNR histogram per channel. `frame` is already filtered."""
    channels = [f"Ch{i}" for i in range(1, 7) if f"Ch{i}" in set(frame["channel"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False)
    for ax, channel in zip(axes.ravel(), channels):
        values = frame.loc[frame["channel"] == channel, "snr"].dropna().to_numpy()
        n_bins = 1 if values.size < 2 else int(np.clip(np.sqrt(values.size), 8, 20))
        ax.hist(values, bins=n_bins, color="C3" if missed_only else "C0", edgecolor="white")
        if values.size:
            median = float(np.median(values))
            ax.axvline(median, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{channel}  n={values.size}  median {median:.1f}")
        else:
            ax.set_title(f"{channel}  n=0")
        ax.set_xlabel("SNR")
        ax.set_ylabel("Feedings")
        ax.grid(True, alpha=0.3)
    kind = "missed feedings" if missed_only else "all feedings"
    fig.suptitle(f"SNR at the F1 threshold — {kind}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def background_z(times_ns, values, label_ns, window_ns):
    """Signal-to-background Z-score of each feeding.

    The signal is the highest sample within +/- 30 minutes. The background
    is every sample on this channel outside all of those feeding windows.
    Z is (signal - background mean) / background standard deviation.
    """
    near = np.zeros(len(times_ns), dtype=bool)
    for when in label_ns:
        lo = np.searchsorted(times_ns, when - window_ns, side="left")
        hi = np.searchsorted(times_ns, when + window_ns, side="right")
        near[lo:hi] = True
    background = values[~near & np.isfinite(values)]
    mu = float(np.mean(background))
    sigma = float(np.std(background, ddof=1))
    z = np.full(len(label_ns), np.nan)
    if not np.isfinite(sigma) or sigma <= 0:
        return z
    for i, when in enumerate(label_ns):
        lo = np.searchsorted(times_ns, when - window_ns, side="left")
        hi = np.searchsorted(times_ns, when + window_ns, side="right")
        window = values[lo:hi]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        z[i] = (float(np.max(window)) - mu) / sigma
    return z


def f1_z_frame(rows, labels, window_ns):
    """Z-score of every feeding at the saved F1-maximizing threshold."""
    thresholds = pd.read_csv(F1_PATH).set_index("channel")["threshold"]
    records = []
    for channel, level in thresholds.items():
        level = float(level)
        for name, frame in rows.items():
            times_ns = frame["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            values = frame[channel].to_numpy(dtype=float)
            label_ns = (
                labels.loc[labels["source"] == name, "t"]
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            peaks_ns, _ = peak_times(values, times_ns, level)
            _, label_hit = match_flags(peaks_ns, label_ns, window_ns)
            z = background_z(times_ns, values, label_ns, window_ns)
            for hit, value in zip(label_hit, z):
                records.append(
                    {
                        "channel": channel,
                        "z": value,
                        "missed": not bool(hit),
                    }
                )
    return pd.DataFrame(records)


def plot_missed_z():
    """Histogram of the signal-to-background Z-score for missed feedings."""
    labels = load_labels()
    rows = clip_to_last_feeding(load_rows(), labels)
    window_ns = int(pd.Timedelta(minutes=MATCH_MINUTES).value)
    frame = f1_z_frame(rows, labels, window_ns)
    missed = frame.loc[frame["missed"]]
    channels = [f"Ch{i}" for i in range(1, 7)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, channel in zip(axes.ravel(), channels):
        values = missed.loc[missed["channel"] == channel, "z"].dropna().to_numpy()
        n_bins = 1 if values.size < 2 else int(np.clip(np.sqrt(values.size), 8, 20))
        ax.hist(values, bins=n_bins, color="C3", edgecolor="white")
        if values.size:
            median = float(np.median(values))
            ax.axvline(median, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{channel}  n={values.size}  median {median:.1f}")
            print(f"{channel}: n={values.size} median {median:.2f} min {values.min():.2f} max {values.max():.2f}")
        else:
            ax.set_title(f"{channel}  n=0")
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Missed feedings")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Signal-to-background Z-score — missed feedings")
    fig.tight_layout()
    path = OUT_DIR / "z_hist_missed.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def plot_f1_snr():
    labels = load_labels()
    rows = clip_to_last_feeding(load_rows(), labels)
    window_ns = int(pd.Timedelta(minutes=MATCH_MINUTES).value)
    frame = f1_snr_frame(rows, labels, window_ns)
    all_path = plot_snr_histograms(frame, OUT_DIR / "snr_hist.png", missed_only=False)
    missed = frame.loc[frame["missed"]]
    missed_path = plot_snr_histograms(missed, OUT_DIR / "snr_hist_missed.png", missed_only=True)
    print(f"Wrote {all_path}")
    print(f"Wrote {missed_path}")
    return all_path, missed_path


if __name__ == "__main__":
    main()
