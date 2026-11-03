"""Per-object local CD-O bars for D1 and the pair-weighted means.

    .runtime/linux-cuda/bin/python scripts/track3/d1_figure.py <video_box.json> <out.png>

Reads local_cdo.py outputs: baseline-cube.json, analysis-oracle-cube.json (scan PCA extents,
analysis only) and the video-measured box file given. Pair weights are the scored (episode,
object) counts from the kit roster; wooden_spoon has no scan and is not in either mean.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
SERIES = [("outputs/mesh_eval/baseline-cube.json", "Cube (floor)", "#2a78d6"),
          ("outputs/mesh_eval/analysis-oracle-cube.json", "Box, scan PCA extents (analysis only)", "#eb6834"),
          (None, "Box, video-measured extents (submitted)", "#1baf7a")]


def scores(path):
    return {row["object"]: row["cd_o_cm"] for row in json.loads(Path(path).read_text())["objects"]
            if row["status"] == "scored"}


def main():
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = [scores(ROOT / path if path else video) for path, _, _ in SERIES]
    pairs = Counter(name for names in json.loads(ROSTER.read_text())["episodes"].values() for name in names)
    objects = sorted(data[2], key=lambda name: data[2][name])
    summary = {}
    for (_, label, _), values in zip(SERIES, data):
        weight = sum(pairs[o] for o in objects)
        summary[label] = {"per_object_mean_cm": round(sum(values[o] for o in objects) / len(objects), 3),
                          "pair_weighted_mean_cm": round(sum(values[o] * pairs[o] for o in objects) / weight, 3),
                          "pairs": weight}
    print(json.dumps(summary, indent=1))
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    height = 0.26
    for i, ((_, label, colour), values) in enumerate(zip(SERIES, data)):
        ys = [k + (1 - i) * (height + 0.02) for k in range(len(objects))]
        ax.barh(ys, [values[o] for o in objects], height=height, color=colour, label=label, edgecolor="white", linewidth=1)
        if i == 2:
            for y, o in zip(ys, objects):
                ax.text(values[o] + 0.05, y, f"{values[o]:.2f}", va="center", fontsize=7, color="#333333")
    ax.set_yticks(range(len(objects)), [f"{o} ({pairs[o]})" for o in objects], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Local CD-O (cm, kit scorer, 3 seeds; lower is better)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    means = [summary[label]["pair_weighted_mean_cm"] for _, label, _ in SERIES]
    ax.set_title("Track 3 meshes: object proportions measured from video vs a cube\n"
                 f"Pair-weighted mean, 26 scanned pairs: cube {means[0]:.2f} cm, scan-PCA box {means[1]:.2f} cm, "
                 f"video box {means[2]:.2f} cm\n(number of scored pairs in brackets; wooden_spoon has no scan)",
                 fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
