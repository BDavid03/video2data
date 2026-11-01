"""Dashboard figures regenerated after every iteration (matplotlib, Agg backend).

Palette and mark rules follow the project dataviz guidance: fixed categorical
hue order per technique, one y-axis per panel, thin marks, recessive grid.
"""

import math

from track3.paths import project_path
from ._common import ROOT
from .ledger import best, best_so_far, completed

SERIES = {"transformer": "#2a78d6", "cnn": "#eb6834", "scratch": "#1baf7a", "gradient_boosting": "#eda100",
          "ann": "#e87ba4", "ensemble": "#6250d6"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "legend.frameon": False})


def render_dashboard(entries, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _style(plt)
    done = completed(entries)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Auto-research loop: tuning log-depth MAE vs MoGe2 pseudo-targets (episodes 39/42)", color=INK)

    ax = axes[0][0]
    xs = [e["iteration"] for e in entries]
    curve = best_so_far(entries, key="composite")
    ax.plot([x for x, c in zip(xs, curve) if c is not None], [c for c in curve if c is not None],
            color=INK2, linewidth=2, label="best usable composite so far")
    for e in done:
        if isinstance(e.get("composite"), (int, float)):
            ax.scatter(e["iteration"], e["composite"], s=46, color=SERIES.get(e["technique"], INK2),
                       edgecolor=SURFACE if e.get("usable") else "#d03b3b", linewidth=1.2 if e.get("usable") else 2.0, zorder=3)
        ax.scatter(e["iteration"], e["tuning_log_mae"], s=16, color=SERIES.get(e["technique"], INK2), alpha=0.45, zorder=2)
    for e in entries:
        if e["status"] != "completed":
            ax.scatter(e["iteration"], ax.get_ylim()[1] if done else 0, marker="x", color="#d03b3b", s=36, zorder=3)
    ax.set_xlabel("iteration"), ax.set_ylabel("composite (large) and log-MAE (small)"), ax.set_title("Progress by iteration: usability composite", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=t) for t, c in SERIES.items()
               if any(e["technique"] == t for e in entries)]
    handles.append(Line2D([], [], color=INK2, linewidth=2, label="best usable composite"))
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#d03b3b", label="fails a gate"))
    handles.append(Line2D([], [], marker="x", linestyle="", color="#d03b3b", label="failed"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax = axes[0][1]
    for e in done:
        if e.get("train_wall_seconds"):
            ax.scatter(e["train_wall_seconds"], e["tuning_log_mae"], s=42, color=SERIES.get(e["technique"], INK2),
                       edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_xscale("log"), ax.set_xlabel("training wall seconds (log)"), ax.set_ylabel("tuning log-depth MAE")
    ax.set_title("Accuracy versus training cost", loc="left")

    ax = axes[1][0]
    top = sorted(done, key=lambda e: (e["composite"] if e.get("usable") and isinstance(e.get("composite"), (int, float)) else 9.0, e["tuning_log_mae"]))[:6]
    episodes = sorted({k for e in top for k in e.get("episode_log_mae", {})})
    width = 0.8 / max(len(episodes), 1)
    for j, episode in enumerate(episodes):
        values = [e["episode_log_mae"].get(episode, math.nan) for e in top]
        ax.bar([i + j * width for i in range(len(top))], values, width=width * 0.92,
               color=["#2a78d6", "#86b6ef", "#184f95"][j % 3], label=f"episode {episode}")
    ax.set_xticks([i + width * (len(episodes) - 1) / 2 for i in range(len(top))])
    ax.set_xticklabels([f"#{e['iteration']} {e['technique']}" for e in top], rotation=20, ha="right")
    ax.set_ylabel("log-depth MAE"), ax.set_title("Per-episode MAE of the top recipes", loc="left"), ax.legend(fontsize=8)

    ax = axes[1][1]
    names = ("abs_rel", "silog", "delta1", "edge_correlation")
    width = 0.8 / max(len(top), 1)
    for i, e in enumerate(top):
        values = [e.get(n) if isinstance(e.get(n), (int, float)) else math.nan for n in names]
        ax.bar([k + i * width for k in range(len(names))], values, width=width * 0.92,
               color=SERIES.get(e["technique"], INK2), label=f"#{e['iteration']} {e['technique']}")
    ax.set_xticks([k + width * (len(top) - 1) / 2 for k in range(len(names))]), ax.set_xticklabels(names)
    ax.set_title("Facets of the top recipes (delta1, edge higher is better)", loc="left"), ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    return destination


def render_samples(entry, destination, count=3):
    """Qualitative panel for one completed entry: RGB, teacher, prediction, |log error|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from track3.features.learning_experiments.data import PreparedPairs
    from track3.features.learning_experiments._common import episode_partition
    from ._common import HOLDOUT_EPISODES, TRAIN_EPISODES, TUNING_EPISODES, read_json
    _style(plt)
    outcome = read_json(entry["outcome_path"])
    dataset = outcome["inputs"]["dataset"]
    pairs = PreparedPairs(project_path(ROOT, dataset), episode_partition(TRAIN_EPISODES, TUNING_EPISODES, HOLDOUT_EPISODES),
                          "tuning", np)
    producer = read_json(project_path(ROOT, entry["prediction_run"]))
    frames = {(f["episode_index"], f["frame_index"]): f for f in producer["frames"]}
    base = project_path(ROOT, entry["prediction_run"]).parent
    picks = list(range(0, len(pairs), max(1, len(pairs) // count)))[:count]
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, index in enumerate(picks):
        sample = pairs[index]
        row = sample["row"]
        with np.load(base / frames[(row["episode_index"], row["frame_index"])]["artifact"], allow_pickle=False) as arrays:
            depth = arrays["depth_m"]
        target = sample["depth_m"]
        error = np.abs(np.log(depth) - np.log(target))
        finite = np.isfinite(target)
        vmin, vmax = (np.nanpercentile(target[finite], 2), np.nanpercentile(target[finite], 98)) if finite.any() else (0, 1)
        panels = ((sample["rgb"], None, f"RGB ep{row['episode_index']} f{row['frame_index']}"),
                  (target, "viridis", "MoGe2 teacher depth (m)"), (depth, "viridis", "prediction depth (m)"),
                  (error, "magma", "|log error|"))
        for c, (image, cmap, title) in enumerate(panels):
            ax = axes[r][c]
            if cmap is None:
                ax.imshow(image)
            else:
                im = ax.imshow(image, cmap=cmap, vmin=0 if c == 3 else vmin, vmax=1.0 if c == 3 else vmax)
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            ax.set_title(title, fontsize=8, loc="left"), ax.axis("off")
    fig.suptitle(f"Iteration {entry['iteration']} {entry['description']}  tuning MAE {entry['tuning_log_mae']:.4f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    return destination


def render_all(entries, directory):
    outputs = {"dashboard": render_dashboard(entries, directory / "dashboard.png")}
    top = best(entries)
    if top is not None:
        try:
            outputs["samples"] = render_samples(top, directory / "best-samples.png")
        except Exception as error:  # the qualitative panel is optional evidence
            outputs["samples_error"] = f"{type(error).__name__}: {error}"
    return outputs
