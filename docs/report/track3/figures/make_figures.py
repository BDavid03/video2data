"""Draw the qualitative panel (default), or regenerate the legacy chart with --bars."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[4]
SUMMARY = ROOT / "outputs/mesh_eval/rules/summary.json"
BUILD = ROOT / "outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f"
SCORES = ROOT / "outputs/mesh_eval/d1-ctx03/v2-cdo.json"
BOX_SCORES = ROOT / "outputs/mesh_eval/d1-ctx02/v1-box.json"
ASSIGNMENTS = ROOT / "track3/features/mesh_extents/assignments.json"
SCANS = ROOT / ("datasets--nvidia--video_to_data_challenge/snapshots/"
                "5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh")
TARGET = Path(__file__).with_name("d1-v2-qualitative.pdf")
# Fixed display-only rotations: upright, handle/long horizontal axis towards +x.
SCAN_ROTATIONS = {
    "white_pot": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    "water_pitcher": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    "plastic_dish_rack": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
}
PANEL_ROWS = (("white_pot", 42), ("water_pitcher", 21), ("plastic_dish_rack", 12))
BLUE = "#0072B2"
LABELS = {
    "beige_cup": "Beige cup", "blue_cup": "Blue cup",
    "mini_dust_pan": "Dust pan", "mini_sweeper": "Sweeper",
    "plastic_dish_rack": "Dish rack", "water_pitcher": "Pitcher",
    "white_pot": "Pot", "white_pot_lid": "Lid",
    "wooden_piece_1": "Wood 1", "wooden_piece_2": "Wood 2",
}


def score_chart() -> None:
    """Read recorded scores, save a vector PDF, and print the derived means for review."""
    summary = json.loads(SUMMARY.read_text())
    rule = next(row for row in summary["ranking"] if row["rule"] == "R2 vessel-handle")
    series = [
        ("Video box", summary["baseline_per_object_cm"], "#4477AA"),
        ("R2 cross-fitted", rule["per_object_cm"], "#228833"),
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
    target = Path(__file__).with_name("d1-v2-cdo.pdf")
    fig.savefig(target, metadata={"Title": "Cross-fitted development CD-O on public scans", "Author": "", "Creator": ""})
    plt.close(fig)
    print(target.relative_to(ROOT))


def cluster_crop(rgb, labels, cluster_ids):
    """Crop around the union of assigned clusters, retaining a small RGB margin."""
    mask = np.isin(labels, np.asarray(cluster_ids) + 1)  # Stored labels are one-based.
    ys, xs = np.nonzero(mask)
    margin = int(0.10 * max(np.ptp(xs), np.ptp(ys)))
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


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
    ax.view_init(elev=25, azim=-65)
    ax.set_axis_off()


def qualitative_panel(rows):
    """Plot the supplied RGB masks, mesh arrays and saved scores without reading files."""
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.18, 3.05))
    grid = fig.add_gridspec(3, 3, left=0.01, right=0.99, bottom=0.12, top=0.82,
                           wspace=0.05, hspace=0.36)
    for x, title in zip((0.17, 0.50, 0.83),
                        ("(a) Public frame 0\nassigned cluster",
                         "(b) D1 v2 (R2)\nR2 / box CD-O (cm)",
                         "(c) Public scan\nscoring reference")):
        fig.text(x, 0.96, title, ha="center", va="top", fontsize=6.8, weight="bold")
    for i, row in enumerate(rows):
        ax = fig.add_subplot(grid[i, 0])
        rgb, mask = row["crop"]
        ax.imshow(rgb)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = (*matplotlib.colors.to_rgb(BLUE), 0.30)
        ax.imshow(overlay)
        ax.contour(mask, levels=[0.5], colors=[BLUE], linewidths=0.65)
        ax.set_axis_off()
        ax.set_title(f"{LABELS[row['name']]} · ep. {row['episode']}",
                     fontsize=6.8, pad=3, weight="bold")
        for j, key, colour in ((1, "candidate", BLUE), (2, "scan", "#B5B5B5")):
            mesh_ax = fig.add_subplot(grid[i, j], projection="3d")
            draw_mesh(mesh_ax, *row[key], colour)
            if key == "candidate":
                mesh_ax.text2D(0.5, -0.02, f"{row['score']:.3f} / {row['box_score']:.3f}",
                               ha="center", va="top", transform=mesh_ax.transAxes, fontsize=6.6)
    fig.text(0.5, 0.014, "Meshes shown at normalised scale · lower CD-O is better",
             ha="center", va="bottom", fontsize=6.2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", action="store_true", help="regenerate the legacy score chart")
    parser.add_argument("--preview", type=Path, help="also save the panel as a 300 dpi PNG")
    args = parser.parse_args()
    if args.bars:
        score_chart()
        return
    build = json.loads((BUILD / "run.json").read_text())
    assignments = json.loads(ASSIGNMENTS.read_text())["episodes"]
    scores = {r["object"]: r for r in json.loads(SCORES.read_text())["objects"]}
    boxes = {r["object"]: r["cd_o_cm"] for r in json.loads(BOX_SCORES.read_text())["objects"]}
    rows = []
    for name, episode in PANEL_ROWS:
        measure = next(ROOT / p for p in build["request"]["measure_runs"]
                       if f"/public/episode_{episode:06d}/" in p)
        record = json.loads(measure.read_text())
        ids = record["request"]["objects"][name]
        assert ids == assignments[f"public:{episode}"]["objects"][name]
        with np.load(measure.with_name("labels_000000.npz")) as saved:
            labels = saved["label"]
        # The recorded overlay's left half is the unmodified, remapped RGB frame.
        rgb = plt.imread(measure.with_name("overlay_000000.png"))[:, :labels.shape[1], :3]
        candidate = trimesh.load(ROOT / scores[name]["candidate"], force="mesh")
        scan = trimesh.load(SCANS / name / f"{name}.glb", force="mesh")
        rows.append({"name": name, "episode": episode, "crop": cluster_crop(rgb, labels, ids),
                     "candidate": (normalised_vertices(candidate.vertices, np.eye(3)), candidate.faces),
                     "scan": (normalised_vertices(scan.vertices, SCAN_ROTATIONS[name]), scan.faces),
                     "score": scores[name]["cd_o_cm"], "box_score": boxes[name]})
        print(f"{name}: R2 {scores[name]['cd_o_cm']:.4f} cm; box {boxes[name]:.4f} cm; "
              f"public ep. {episode}, clusters {ids}; {measure.relative_to(ROOT)}")
    fig = qualitative_panel(rows)
    fig.savefig(TARGET, dpi=300, metadata={"Title": "Video clusters, submitted R2 meshes and scoring references",
                                         "Author": "", "Creator": ""})
    if args.preview:
        fig.savefig(args.preview, dpi=300)
    plt.close(fig)
    print(TARGET.relative_to(ROOT))


if __name__ == "__main__":
    main()
