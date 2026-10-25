import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feeding_features import APPENDIX_FEATURES, CORR_HIGHLIGHT, remaining_feature_frame

FEATURES_PATH = "/Volumes/My Book/exp_1/sw/feeding_features/features.parquet"
KEPT_NAME = "features_kept.parquet"

# One line each, matching how extract_mountain_features computes the value.
FEATURE_HELP = {
    "Response onset": "Sample index where the rise leaves the baseline.",
    "Response duration": "Samples from curve start to curve end.",
    "FWHM": "Width at half the baseline-to-peak rise.",
    "Plateau-rise duration": "Samples from the 90% level to the peak.",
    "Baseline": "Lower of the mean of the first and last 5% of the window.",
    "Peak height": "Raw value at the smoothed peak.",
    "Plateau-rise change": "Peak minus the value at the 90% rise.",
    "Plateau-fall change": "Value at the 90% fall minus the peak.",
    "Plateau gradient": "Linear slope from the 90% rise to the 90% fall.",
    "Plateau-gradient R2": "How straight that plateau slope is.",
    "Plateau-rise gradient": "Linear slope from the 90% level to the peak.",
    "Plateau-rise-gradient R2": "How straight the rise-to-peak slope is.",
    "Maximum plateau-rise gradient": "Steepest one-sample step from the 90% level to the peak.",
    "Plateau-fall gradient": "Linear slope from the peak to the 90% fall.",
    "Plateau-fall-gradient R2": "How straight the peak-to-90% fall slope is.",
    "Rise AUC": "Area above baseline from curve start to the 90% level.",
    "Fall AUC": "Area above baseline from the 90% fall to the curve end.",
    "Plateau-rise AUC": "Area above baseline from the 90% level to the peak.",
    "Fall AUC fraction": "Fall area divided by the whole curve area.",
    "Plateau standard deviation": "Spread of values between the 90% rise and the 90% fall.",
    "Symmetry ratio": "Rise duration divided by fall duration.",
    "Rise-to-fall gradient ratio": "Absolute rise slope divided by the fall slope.",
    "Rise-to-fall AUC ratio": "Rise area above baseline divided by the fall area.",
    "Feeding sine": "Sine of time of day.",
    "Feeding cosine": "Cosine of time of day, so sine and cosine make a clock circle.",
}


def groups_in_order(names):
    ordered = []
    seen = set()
    for name, group, _ in APPENDIX_FEATURES:
        if name not in names or group in seen:
            continue
        seen.add(group)
        ordered.append(group)
    return ordered


def names_in_group(group, names):
    return [
        name
        for name, feature_group, _ in APPENDIX_FEATURES
        if feature_group == group and name in names
    ]


def plot_histograms(frame, group, names):
    n = len(names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.6, nrows * 3.4),
        squeeze=False,
    )
    for i, name in enumerate(names):
        ax = axs[i // ncols, i % ncols]
        values = frame[name].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            ax.set_title(name, fontsize=10)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        n_unique = int(values.nunique())
        n_bins = (
            1
            if n_unique <= 1
            else min(20, max(8, n_unique if n_unique < 8 else int(np.sqrt(len(values))) + 5))
        )
        ax.hist(
            values,
            bins=n_bins,
            color="steelblue",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.axvline(float(values.mean()), color="black", linestyle="--", linewidth=1.1, label="Mean")
        ax.axvline(float(values.median()), color="dimgray", linestyle=":", linewidth=1.1, label="Median")
        subtitle = textwrap.fill(FEATURE_HELP[name], width=42)
        ax.set_title(f"{name}\n{subtitle}", fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.text(
            0.98,
            0.96,
            f"n={len(values)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    for j in range(n, nrows * ncols):
        axs[j // ncols, j % ncols].axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"{group} — distributions", fontsize=13)
    fig.tight_layout()
    return fig


def plot_correlation(frame, group, names):
    finite = frame[names].replace([np.inf, -np.inf], np.nan)
    corr = finite.corr(method="pearson", min_periods=3)
    n = len(names)
    size = max(4.5, 0.7 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        square=True,
        cbar_kws={"label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(f"{group} — Pearson correlation")
    fig.tight_layout()
    return fig, corr


def print_high_pairs(corr, names):
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(float(r)) >= CORR_HIGHLIGHT:
                rows.append((a, b, float(r)))
    rows.sort(key=lambda item: abs(item[2]), reverse=True)
    print(f"  Pairs with |r| >= {CORR_HIGHLIGHT:.2f}: {len(rows)}")
    if not rows:
        print("    none")
        return
    for a, b, r in rows:
        print(f"    {a}  ~  {b}    r={r:.3f}")


def choose_features(names):
    by_lower = {name.lower(): name for name in names}
    code_to_name = {
        col.lower(): name
        for name, _, col in APPENDIX_FEATURES
        if name in by_lower.values()
    }
    listed = ", ".join(f"{i}. {name}" for i, name in enumerate(names, start=1))
    print(f"  {listed}")
    while True:
        answer = input(
            "Keep which features? numbers or names, comma-separated "
            "[Enter = all, none = drop]: "
        ).strip()
        if answer == "" or answer.lower() == "all":
            return list(names)
        if answer.lower() == "none":
            return []
        kept = []
        unknown = []
        for token in answer.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                index = int(token)
                if 1 <= index <= len(names):
                    kept.append(names[index - 1])
                else:
                    unknown.append(token)
                continue
            match = by_lower.get(token.lower()) or code_to_name.get(token.lower())
            if match is None:
                unknown.append(token)
            else:
                kept.append(match)
        if unknown:
            print(f"  Not in this group: {', '.join(unknown)}")
            continue
        if not kept:
            print("  No features given. Enter keeps all, none drops the group.")
            continue
        ordered = [name for name in names if name in set(kept)]
        return ordered


def chosen_frame(ids, remaining, kept):
    """Rows stay aligned with the source table. Columns are ids plus the kept features."""
    return pd.concat([ids, remaining.loc[:, kept]], axis=1)


def main():
    missing_help = [name for name, _, _ in APPENDIX_FEATURES if name not in FEATURE_HELP]
    if missing_help:
        raise RuntimeError(f"Missing explanations: {missing_help}")

    raw = pd.read_parquet(FEATURES_PATH)
    remaining, missing_cols = remaining_feature_frame(raw)
    id_cols = [col for col in ("channel", "feeding_datetime") if col in raw.columns]
    ids = raw.loc[remaining.index, id_cols].reset_index(drop=True)
    remaining = remaining.reset_index(drop=True)
    appendix_names = [name for name, _, _ in APPENDIX_FEATURES]
    for name in appendix_names:
        if name not in remaining.columns:
            remaining[name] = np.nan
    remaining = remaining[appendix_names]

    print(f"Rows: {len(remaining)}")
    print(f"Asking about all {len(appendix_names)} features from feeding_features.py")
    if "channel" in ids.columns:
        print(f"Channels: {', '.join(map(str, sorted(ids['channel'].unique())))}")
    if missing_cols:
        print(f"Appendix columns not found: {missing_cols}")

    kept = []
    for group in groups_in_order(appendix_names):
        names = names_in_group(group, appendix_names)
        print("\n" + "=" * 72)
        print(group)
        print("=" * 72)
        for i, name in enumerate(names, start=1):
            print(f"  {i}. {name}")
            print(f"     {FEATURE_HELP[name]}")

        plot_histograms(remaining, group, names)
        _, corr = plot_correlation(remaining, group, names)
        print_high_pairs(corr, names)
        print("  Close the figures to choose features for this group.")
        plt.show()

        chosen = choose_features(names)
        print(f"  Kept: {', '.join(chosen) if chosen else '(none)'}")
        kept.extend(chosen)

    print("\n" + "=" * 72)
    print(f"Kept {len(kept)} features")
    for name in kept:
        print(f"  {name}")
    if not kept:
        print("Nothing kept. No file written.")
        return None

    chosen_df = chosen_frame(ids, remaining, kept)
    path = os.path.join(os.path.dirname(FEATURES_PATH), KEPT_NAME)
    chosen_df.to_parquet(path, index=False)
    print(chosen_df.shape)
    print(list(chosen_df.columns))
    print(path)
    return chosen_df


if __name__ == "__main__":
    chosen_df = main()
