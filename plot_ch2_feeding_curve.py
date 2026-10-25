"""One Ch2 feeding curve: peak, baseline, rise AUC, fall AUC, and the plateau.

The plateau is the span where the smoothed curve stays at or above 90% of
its amplitude, measured from the smoothed baseline to the smoothed peak.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent / "ein_hamizraf_features"))
import feeding_features_ein as features

FEEDING = pd.Timestamp("2025-09-16 12:20:42")
CHANNEL = "Ch2"
OUT = Path(__file__).resolve().parent / "ch2_feeding_curve_0916_1220.png"


def main():
    signal, _clocks = features.load_signal()
    times = signal["t"].to_numpy(dtype="datetime64[ns]")
    values = signal[CHANNEL].to_numpy(dtype=float)
    window = features.window_frame(times, values, FEEDING)
    found = features.extract_mountain_features(
        window,
        value_col="value",
        time_col="time",
        plateau_threshold=0.90,
    )
    if not found:
        raise SystemExit(f"No feeding curve for {CHANNEL} at {FEEDING}")

    y = window["value"].to_numpy(dtype=float)
    y = pd.Series(y).interpolate().bfill().ffill().to_numpy()
    clock = pd.to_datetime(window["time"])
    minutes = (clock - clock.iloc[0]).dt.total_seconds().to_numpy() / 60

    start = int(found["mountain_start"])
    rise_end = int(found["plateau_start"])
    fall_start = int(found["plateau_end"])
    end = int(found["mountain_end"])
    peak = int(found["peak_index"])
    baseline = float(found["baseline"])

    clock_ns = pd.to_datetime(window["time"]).to_numpy()
    y_smooth = gaussian_filter1d(
        y, sigma=features.samples_per_minutes(clock_ns, features.DETECT_SMOOTH_MINUTES)
    )
    n_base = max(3, int(0.05 * len(y)))
    search_baseline = min(np.mean(y_smooth[:n_base]), np.mean(y_smooth[-n_base:]))
    search_peak = float(y_smooth[peak])
    plateau_level = search_baseline + 0.90 * (search_peak - search_baseline)

    signal_color = "#222222"
    smooth_color = "#2EAD4A"
    rise_color = "#6A4C93"
    fall_color = "#F0D040"
    plateau_color = "#2EAD4A"
    baseline_color = "#7A7A7A"
    peak_color = "#C23B22"

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axvspan(
        minutes[rise_end],
        minutes[fall_start],
        color=plateau_color,
        alpha=0.14,
        label="Plateau",
        zorder=1,
    )
    ax.plot(minutes, y, color=signal_color, linewidth=2.0, zorder=3)
    ax.plot(
        minutes,
        y_smooth,
        color=smooth_color,
        linewidth=1.25,
        label="Smoothed",
        zorder=4,
    )

    ax.fill_between(
        minutes[start : rise_end + 1],
        baseline,
        y[start : rise_end + 1],
        color=rise_color,
        alpha=0.42,
        label="Rise AUC",
        zorder=2,
    )
    ax.fill_between(
        minutes[fall_start : end + 1],
        baseline,
        y[fall_start : end + 1],
        color=fall_color,
        alpha=0.55,
        label="Fall AUC",
        zorder=2,
    )
    ax.hlines(
        baseline,
        minutes[0],
        minutes[-1],
        colors=baseline_color,
        linestyles="--",
        linewidth=1.2,
        label="Baseline",
        zorder=4,
    )
    ax.hlines(
        plateau_level,
        minutes[0],
        minutes[-1],
        colors=plateau_color,
        linestyles=":",
        linewidth=1.4,
        label="90% level",
        zorder=4,
    )
    ax.scatter(
        [minutes[rise_end], minutes[fall_start]],
        [y_smooth[rise_end], y_smooth[fall_start]],
        marker="o",
        s=42,
        facecolors="white",
        edgecolors=plateau_color,
        linewidths=1.6,
        zorder=6,
        label="Plateau start / end",
    )
    ax.scatter(
        minutes[peak],
        y[peak],
        marker="^",
        s=55,
        color=peak_color,
        zorder=6,
        label="Peak",
    )
    ax.annotate(
        "Plateau: smoothed curve\nat or above 90%",
        xy=(6, plateau_level),
        xytext=(1, 0.96),
        ha="left",
        va="top",
        fontsize=13,
        color="#1B7A32",
        arrowprops={"arrowstyle": "-", "color": "#1B7A32"},
    )

    ax.set_xlim(minutes[0], minutes[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minutes in window", fontsize=16)
    ax.set_ylabel("Normalized Aqua-MAFRI", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
