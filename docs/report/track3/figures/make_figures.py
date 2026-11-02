"""Draw recorded mesh scores, qualitative examples, or initial pose errors."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/tmp/report-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SUMMARY = ROOT / "outputs/mesh_eval/rules/summary.json"
POSE_SHRINK = ROOT / "outputs/pose_insurance/shrink/summary.json"
SCORES = ROOT / "outputs/mesh_eval/pairs/mesh-round3-veto/summary.json"
BUILD = ROOT / "outputs/mesh_extents/build/20261102T051416.149421Z-f377e0a1"
PREVIOUS_BUILD = ROOT / "outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295"
SCANS = ROOT / ("datasets--nvidia--video_to_data_challenge/snapshots/"
                "5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh")
PROXY_LEDGER = ROOT / "outputs/mesh_extents/spoon-proxy/ledger.jsonl"
PROXY_IDENTITY = "3c865ba6dd91e9436badacb858eabacb10e08abce1342ed92f52d12b859cb54b"
TARGET = Path(__file__).with_name("d1-v4-objects.pdf")
BLUE = "#0072B2"
ORANGE = "#D55E00"
LABELS = {
    "beige_cup": "Beige cup", "blue_cup": "Blue cup",
    "mini_dust_pan": "Dust pan", "mini_sweeper": "Sweeper",
    "plastic_dish_rack": "Dish rack", "water_pitcher": "Pitcher",
    "white_pot": "Pot", "white_pot_lid": "Lid", "wooden_spoon": "Spoon",
    "wooden_piece_1": "Wood 1", "wooden_piece_2": "Wood 2",
}


def pose_offset_panel(episodes: list[int], offsets: dict[str, list[float]]) -> plt.Figure:
    """Compare recorded frame-zero errors of the second object's relative position."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.18, 2.10), layout="constrained")
    x = np.arange(len(episodes))
    for shift, name, label, colour, hatch in (
        (-0.25, "I", "Centroids + stacking", BLUE, ""),
        (0.0, "S1", "Stereo shrinkage", "#D55E00", "//"),
        (0.25, "G", "Uniform shrinkage", "#009E73", ".."),
    ):
        ax.bar(x + shift, offsets[name], width=0.24, label=label,
               color=colour, hatch=hatch, linewidth=0.4, edgecolor="white")
    labels = [str(episode) for episode in episodes]
    ax.set(xticks=x, xticklabels=labels, xlabel="Training episode",
           ylabel="Initial second-object\nposition error (cm)",
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


def object_score_panel(rows: list[dict]) -> plt.Figure:
    """Join v3/v4 per-object means on one axis, with an unmeasured spoon row."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.18, 3.40))
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.27, top=0.86)
    y = np.arange(len(rows))
    before = [row["v3"] for row in rows]
    after = [row["v4"] for row in rows]
    ax.hlines(y, after, before, color="#777777", linewidth=1.3, zorder=2)
    ax.plot(before, y, linestyle="none", marker="o", markersize=6,
            markerfacecolor="none", markeredgewidth=1.2, color=BLUE,
            label="v3", zorder=3)
    ax.plot(after, y, linestyle="none", marker="x", markersize=4.5,
            markeredgewidth=1.3, color=ORANGE, label="v4", zorder=4)
    ax.set(yticks=y,
           yticklabels=[f"{LABELS[row['object']]} ({row['pairs']})" for row in rows],
           xlim=(0, 1.6), xticks=[0, 0.5, 1.0, 1.5], ylim=(len(rows)-0.3, -0.7),
           xlabel="CD-O (cm; lower is better)")
    ax.xaxis.label.set_fontsize(8.5)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.01, 0.97, "Object (pairs)", va="top", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=False, fontsize=9,
               handlelength=1, columnspacing=0.8, handletextpad=0.4)
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.145, 0.145],
                             transform=fig.transFigure, color="#bbbbbb", linewidth=0.6))
    fig.text(0.02, 0.12, "Spoon", va="top", fontsize=9, weight="bold")
    fig.text(0.02, 0.065, "unmeasured: 10/36 pair entries, no public scan",
             va="top", fontsize=8)
    return fig


def object_score_chart(preview: Path | None, output_dir: Path = TARGET.parent) -> None:
    """Aggregate recorded pair scores, assert the published means, and draw all objects."""
    summary = json.loads(SCORES.read_text())
    rows = []
    for saved in summary["objects"]:
        pairs = [row for row in summary["pairs"] if row["object"] == saved["object"]]
        before = float(np.mean([row["v3"] for row in pairs]))
        after = float(np.mean([row["candidate"] for row in pairs]))
        assert len(pairs) == saved["pairs"]
        assert np.isclose(before, saved["v3"], rtol=0, atol=1e-12)
        assert np.isclose(after, saved["candidate"], rtol=0, atol=1e-12)
        rows.append({"object": saved["object"], "pairs": len(pairs),
                     "v3": before, "v4": after})
    rows.sort(key=lambda row: (row["v4"]-row["v3"], LABELS[row["object"]]))
    means = {name: float(np.mean([row[name] for row in rows])) for name in ("v3", "v4")}
    assert len(rows) == 10 and sum(row["pairs"] for row in rows) == 26
    assert f"{means['v3']:.4f}" == "0.7711"
    assert f"{means['v4']:.4f}" == "0.6534"
    assert np.isclose(means["v3"], summary["means"]["v3"]["per_object_cm"], rtol=0, atol=1e-12)
    assert np.isclose(means["v4"], summary["means"]["candidate"]["per_object_cm"], rtol=0, atol=1e-12)
    assert summary["total_pairs"] == 36 and len(summary["unscored"]) == 10
    assert all(row["object"] == "wooden_spoon" for row in summary["unscored"])
    pair_means = {name: sum(row[name]*row["pairs"] for row in rows)/26 for name in ("v3", "v4")}
    fig = object_score_panel(rows)
    target = output_dir / TARGET.name
    fig.savefig(target, metadata={"Title": "Packed v3/v4 CD-O on all ten scanned objects",
                                  "Author": "", "Creator": "", "CreationDate": None,
                                  "ModDate": None})
    if preview:
        fig.savefig(preview, dpi=300)
    plt.close(fig)
    print(json.dumps({"objects": rows, "object_means_cm": means, "pair_means_cm": pair_means,
                      "measured_pair_entries": 26, "unmeasured_spoon_entries": 10}, indent=2))
    print(target.relative_to(ROOT))


def normalised_vertices(vertices: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Orient, centre the bounding box, and uniformly scale its longest side to one."""
    points = np.asarray(vertices) @ rotation.T
    return (points - (points.max(0) + points.min(0)) / 2) / np.ptp(points, axis=0).max()


def draw_mesh(ax, vertices: np.ndarray, faces: np.ndarray, colour: str) -> None:
    """Use one orthographic camera and light for all normalised meshes."""
    surface = Poly3DCollection(vertices[faces], facecolors=colour, linewidths=0, shade=True)
    surface.set_rasterized(True)
    surface.set_clip_on(False)
    ax.add_collection3d(surface)
    ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), zlim=(-0.55, 0.55))
    ax.set_box_aspect((1, 1, 1), zoom=1.85)
    ax.set_proj_type("ortho")
    ax.view_init(elev=40, azim=-65)
    ax.set_axis_off()


def restyle_overlay(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Restyle the recorded mask and outline without changing their geometry."""
    source = np.rint(rgb[..., :3]*255).astype(np.uint8)
    saved = np.rint(overlay[..., :3]*255).astype(np.uint8)
    observed = np.all(saved == (source*0.5 + [0, 127, 0]).astype(np.uint8), axis=-1)
    outline = np.all(saved == [255, 0, 255], axis=-1)
    image = source.astype(float)/255
    image[observed] = 0.70*image[observed] + 0.30*np.array([0, 0.8, 0])
    outline = cv2.dilate(outline.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    image[outline] = [0.82, 0, 0.65]
    return image


def qualitative_panel(rows: list[dict], spoon: list[dict]) -> plt.Figure:
    """Show matched mesh views and two recorded, identically cropped spoon overlays."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.18, 2.85))
    for j, title in enumerate(("v3 mesh", "v4 mesh", "Public scan")):
        fig.text(0.25 + j*0.30, 0.98, title, ha="center", va="top",
                 fontsize=8.5, weight="bold")
    for i, row in enumerate(rows):
        bottom = 0.65 - i*0.27
        fig.text(0.035, bottom+0.13, LABELS[row["name"]], rotation=90,
                 ha="center", va="center", fontsize=8.5, weight="bold")
        for j, key, colour in ((0, "previous", "#B5B5B5"),
                               (1, "candidate", "#B5B5B5"), (2, "scan", "#B5B5B5")):
            ax = fig.add_axes((0.10+j*0.30, bottom, 0.30, 0.26), projection="3d")
            draw_mesh(ax, *row[key], colour)
    fig.text(0.5, 0.35, "Spoon: evaluation 15, single-frame IoU", ha="center",
             va="center", fontsize=8.5, weight="bold")
    for j, row in enumerate(spoon):
        ax = fig.add_axes((0.03+j*0.50, 0.15, 0.44, 0.16))
        ax.imshow(row["crop"])
        ax.set_axis_off()
        fig.text(0.25+j*0.50, 0.12, f"{row['label']}: {row['iou']:.2f}",
                 ha="center", va="top", fontsize=8.5)
    fig.text(0.5, 0.035, "IoU is not a validated CD-O proxy", ha="center",
             va="bottom", fontsize=8.5, weight="bold")
    return fig


def qualitative_chart(preview: Path | None, output_dir: Path = TARGET.parent) -> None:
    """Load only the two previously illustrated scans and saved spoon trial overlays."""
    objects = json.loads((BUILD / "objects.json").read_text())["objects"]
    previous = json.loads((PREVIOUS_BUILD / "objects.json").read_text())["objects"]
    # Fixed display-only rotation used by the earlier qualitative figure.
    scan_rotation = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    rows = []
    for name in ("plastic_dish_rack", "mini_dust_pan"):
        row = {"name": name}
        for key, path, rotation in (
            ("previous", PREVIOUS_BUILD / previous[name]["file"], np.eye(3)),
            ("candidate", BUILD / objects[name]["file"], np.eye(3)),
            ("scan", SCANS / name / f"{name}.glb", scan_rotation),
        ):
            mesh = trimesh.load(path, force="mesh")
            row[key] = (normalised_vertices(mesh.vertices, rotation), mesh.faces)
        rows.append(row)
    trials = [json.loads(line) for line in PROXY_LEDGER.read_text().splitlines()]
    trials = [row for row in trials if row["frozen_identity"] == PROXY_IDENTITY
              and row["view"] == "evaluation-000015-wooden_spoon"]
    assert len(trials) == 2
    recorded = {}
    for row in trials:
        run_path = ROOT / row["run"]
        run = json.loads(run_path.read_text())
        result = json.loads(run_path.with_name("result.json").read_text())
        assert result["iou"] == row["iou"]
        recorded[run["request"]["candidate"]["label"]] = (row, run, result)
    spoon = []
    for label, candidate_label in (("v3 box", "v3_box"), ("v4 bowl", "v4_bowl")):
        row, run, result = recorded[candidate_label]
        overlay = plt.imread(ROOT / row["overlay"])
        rgb = plt.imread(ROOT / run["request"]["view"]["overlay"])[:, :overlay.shape[1]]
        # Identical display crop in both 768x576 overlays; includes the whole spoon.
        crop = restyle_overlay(rgb[442:535, 300:610], overlay[442:535, 300:610])
        spoon.append({"label": label, "iou": result["iou"], "crop": crop})
        print(json.dumps({"label": label, "iou": result["iou"], "overlay": row["overlay"],
                          "crop_xyxy": [300, 442, 610, 535]}))
    fig = qualitative_panel(rows, spoon)
    target = output_dir / "d1-v4-qualitative.pdf"
    fig.savefig(target, dpi=300, metadata={"Title": "Matched mesh views and unvalidated spoon silhouette check",
                                         "Author": "", "Creator": "", "CreationDate": None,
                                         "ModDate": None})
    if preview:
        fig.savefig(preview, dpi=300)
    plt.close(fig)
    print(target.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", action="store_true", help="draw the legacy mesh score bars")
    parser.add_argument("--pose-offsets", action="store_true", help="draw initial position errors")
    parser.add_argument("--qualitative", action="store_true", help="draw mesh views and saved spoon overlays")
    parser.add_argument("--preview", type=Path, help="also save the included figure as a PNG")
    parser.add_argument("--output-dir", type=Path, default=TARGET.parent,
                        help="write figure PDFs here instead of beside this script")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.qualitative:
        qualitative_chart(args.preview, output_dir)
    elif args.pose_offsets:
        pose_offset_chart(args.preview, output_dir)
    elif args.bars:
        score_chart(output_dir)
    else:
        object_score_chart(args.preview, output_dir)


if __name__ == "__main__":
    main()
