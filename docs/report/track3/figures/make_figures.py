"""Draw recorded mesh examples, initial pose offsets, or legacy mesh bars."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/tmp/d3fix/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SUMMARY = ROOT / "outputs/mesh_eval/rules/summary.json"
POSE_SHRINK = ROOT / "outputs/pose_insurance/shrink/summary.json"
BUILD = ROOT / "outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295"
SCORES = ROOT / "outputs/mesh_eval/pairs/mesh-round2-v2-v3/summary.json"
PREVIOUS_BUILD = ROOT / "outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f"
SCANS = ROOT / ("datasets--nvidia--video_to_data_challenge/snapshots/"
                "5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh")
TARGET = Path(__file__).with_name("d1-v3-qualitative.pdf")
# Fixed display-only rotations: upright, handle/long horizontal axis towards +x.
SCAN_ROTATIONS = {
    "white_pot": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    "blue_cup": np.diag([1, -1, -1]),
    "plastic_dish_rack": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
}
PANEL_OBJECTS = ("white_pot", "plastic_dish_rack", "blue_cup")
BLUE = "#0072B2"
LABELS = {
    "beige_cup": "Beige cup", "blue_cup": "Blue cup",
    "mini_dust_pan": "Dust pan", "mini_sweeper": "Sweeper",
    "plastic_dish_rack": "Dish rack", "water_pitcher": "Pitcher",
    "white_pot": "Pot", "white_pot_lid": "Lid",
    "wooden_piece_1": "Wood 1", "wooden_piece_2": "Wood 2",
}


def pose_offset_panel(episodes: list[int], offsets: dict[str, list[float]]) -> plt.Figure:
    """Compare recorded frame-zero errors of the second object's relative position."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.18, 2.10), layout="constrained")
    x = np.arange(len(episodes))
    for shift, name, label, colour, hatch in (
        (-0.25, "I", "Earlier rule", BLUE, ""),
        (0.0, "S1", "Stereo shrinkage", "#D55E00", "//"),
        (0.25, "G", "Uniform shrinkage", "#009E73", ".."),
    ):
        ax.bar(x + shift, offsets[name], width=0.24, label=label,
               color=colour, hatch=hatch, linewidth=0.4, edgecolor="white")
    labels = [f"{episode}*" if episode in (0, 26) else str(episode) for episode in episodes]
    ax.set(xticks=x, xticklabels=labels, xlabel="Train episode (* stacked)",
           ylabel="Frame-0 offset error of\nsecond object (cm)",
           ylim=(0, 100), yticks=[0, 20, 40, 60, 80])
    ax.yaxis.label.set_fontsize(8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", pad=2)
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=8,
              handlelength=1.1, columnspacing=0.8, handletextpad=0.4)
    return fig


def pose_offset_chart(preview: Path | None, output_dir: Path = TARGET.parent) -> None:
    """Read the same frame-zero metric for all three recorded shrink-control candidates."""
    episodes = [0, 11, 12, 18, 21, 23, 26, 31]
    summary = json.loads(POSE_SHRINK.read_text())
    values = {
        name: [summary["episodes"][name][str(e)]["frame0"]["offset_error_cm"] for e in episodes]
        for name in ("I", "S1", "G")
    }
    fig = pose_offset_panel(episodes, values)
    target = output_dir / "d2-slot1-offset.pdf"
    fig.savefig(target, metadata={"Title": "Initial placement errors: earlier rule, stereo and uniform shrinkage",
                                 "Author": "", "Creator": "", "CreationDate": None,
                                 "ModDate": None})
    if preview:
        fig.savefig(preview, dpi=300)
    plt.close(fig)
    print(json.dumps({"episodes": episodes, "offset_error_cm": values}, indent=2))
    print(target.relative_to(ROOT))


def score_chart(output_dir: Path = TARGET.parent) -> None:
    """Read recorded scores, save a vector PDF, and print the derived means for review."""
    summary = json.loads(SUMMARY.read_text())
    rule = next(row for row in summary["ranking"] if row["rule"] == "R2 vessel-handle")
    series = [
        ("Video box", summary["baseline_per_object_cm"], "#4477AA"),
        ("Cylinder-plus-handle rule (cross-fitted)", rule["per_object_cm"], "#228833"),
    ]
    objects = sorted(rule["per_object_cm"], key=rule["per_object_cm"].get)
    weights = summary["pair_counts"]
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.65), layout="constrained")
    for i, (label, values, colour) in enumerate(series):
        ax.barh([k + (i - 0.5) * 0.32 for k in range(len(objects))],
                [values[o] for o in objects], height=0.29, label=label, color=colour)
        print(f"{label}: object mean {sum(values[o] for o in objects) / len(objects):.6f} cm; "
              f"pair weighted {sum(values[o] * weights[o] for o in objects) / sum(weights[o] for o in objects):.6f} cm")
    ax.set_yticks(range(len(objects)), [LABELS[o] for o in objects])
    ax.invert_yaxis()
    ax.set_xlabel("Local CD-O (cm; lower is better)")
    ax.set_xlim(0, 2.4)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    target = output_dir / "d1-v2-cdo.pdf"
    fig.savefig(target, metadata={"Title": "Cross-fitted development CD-O on public scans", "Author": "", "Creator": ""})
    plt.close(fig)
    print(target.relative_to(ROOT))


def normalised_vertices(vertices, rotation):
    """Rigidly orient, centre the bounding box, and uniformly scale its longest side to one."""
    points = np.asarray(vertices) @ rotation.T
    return (points - (points.max(0) + points.min(0)) / 2) / np.ptp(points, axis=0).max()


def draw_mesh(ax, vertices, faces, colour):
    """Use one orthographic camera and one light for all normalised meshes."""
    surface = Poly3DCollection(vertices[faces], facecolors=colour, linewidths=0, shade=True)
    surface.set_rasterized(True)
    ax.add_collection3d(surface)
    ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), zlim=(-0.55, 0.55))
    ax.set_box_aspect((1, 1, 1), zoom=1.65)
    ax.set_proj_type("ortho")
    ax.view_init(elev=40, azim=-65)
    ax.set_axis_off()


def qualitative_panel(rows):
    """Compare previous/revised geometry and scans with matched views and readable scores."""
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.18, 3.10))
    grid = fig.add_gridspec(len(rows), 3, left=0.10, right=0.99, bottom=0.07, top=0.89,
                           wspace=0.02, hspace=0.54)
    for j, title in enumerate(("v2 mesh", "v3 mesh", "Public scan")):
        bounds = grid[0, j].get_position(fig)
        fig.text((bounds.x0 + bounds.x1) / 2, 0.98, title,
                 ha="center", va="top", fontsize=8.5, weight="bold")
    for i, row in enumerate(rows):
        bounds = grid[i, 0].get_position(fig)
        fig.text(0.035, (bounds.y0 + bounds.y1) / 2, LABELS[row["name"]],
                 rotation=90, ha="center", va="center", fontsize=8, weight="bold")
        for j, key, colour in ((0, "previous", "#669BBC"),
                               (1, "candidate", BLUE), (2, "scan", "#B5B5B5")):
            ax = fig.add_subplot(grid[i, j], projection="3d")
            draw_mesh(ax, *row[key], colour)
            if key != "scan":
                score = row["previous_score"] if key == "previous" else row["score"]
                ax.text2D(0.5, -0.02, f"{score:.3f} cm", ha="center", va="top",
                          transform=ax.transAxes, fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", action="store_true", help="regenerate the legacy score chart")
    parser.add_argument("--pose-offsets", action="store_true", help="draw recorded earlier-rule, stereo and uniform pose offsets")
    parser.add_argument("--preview", type=Path, help="also save the panel as a 300 dpi PNG")
    parser.add_argument("--output-dir", type=Path, default=TARGET.parent,
                        help="write figure PDFs here instead of beside this script")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pose_offsets:
        pose_offset_chart(args.preview, args.output_dir)
        return
    if args.bars:
        score_chart(args.output_dir)
        return
    objects = json.loads((BUILD / "objects.json").read_text())["objects"]
    previous = json.loads((PREVIOUS_BUILD / "objects.json").read_text())["objects"]
    scores = {r["object"]: r for r in json.loads(SCORES.read_text())["objects"]}
    rows = []
    for name in PANEL_OBJECTS:
        old = trimesh.load(PREVIOUS_BUILD / previous[name]["file"], force="mesh")
        candidate = trimesh.load(BUILD / objects[name]["file"], force="mesh")
        scan = trimesh.load(SCANS / name / f"{name}.glb", force="mesh")
        rows.append({"name": name,
                     "previous": (normalised_vertices(old.vertices, np.eye(3)), old.faces),
                     "candidate": (normalised_vertices(candidate.vertices, np.eye(3)), candidate.faces),
                     "scan": (normalised_vertices(scan.vertices, SCAN_ROTATIONS[name]), scan.faces),
                     "score": scores[name]["v3"], "previous_score": scores[name]["v2"]})
        print(f"{name}: v2 {scores[name]['v2']:.4f} cm; v3 {scores[name]['v3']:.4f} cm")
    fig = qualitative_panel(rows)
    target = args.output_dir / TARGET.name
    fig.savefig(target, dpi=300, metadata={"Title": "Previous and revised meshes beside public scans; exact packed means",
                                         "Author": "", "Creator": ""})
    if args.preview:
        fig.savefig(args.preview, dpi=300)
    plt.close(fig)
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
