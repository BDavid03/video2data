"""Draw recorded mesh scores, qualitative examples, or initial pose errors."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/.runtime/tmp/report-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SUMMARY = ROOT / "data/outputs/mesh_eval/rules/summary.json"
POSE_SHRINK = ROOT / "data/outputs/pose_insurance/shrink/summary.json"
SCORES = ROOT / "data/outputs/mesh_eval/pairs/mesh-round3-veto/summary.json"
GENERATIVE = ROOT / "data/outputs/mesh_generative/g1-pilot"
PACKED_V5 = ROOT / "data/outputs/mesh_generative/g2-packed-check"
TARGET = Path(__file__).with_name("d1-objects.pdf")
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
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
    """Compare three packed versions, highlighting the two objects changed by v5."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.18, 3.05))
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.22, top=0.86)
    y = np.arange(len(rows))
    before = [row["v3"] for row in rows]
    after = [row["v4"] for row in rows]
    current = [row["v5"] for row in rows]
    for i, row in enumerate(rows):
        ax.plot([row["v3"], row["v4"], row["v5"]],
                [i-0.22, i, i+0.22], color="#aaaaaa", linewidth=0.8, zorder=2)
        if row["v5"] != row["v4"]:
            ax.axhspan(i-0.48, i+0.48, color=GREEN, alpha=0.08, zorder=0)
            ax.plot([row["v4"], row["v5"]], [i, i+0.22],
                    color=GREEN, linewidth=1.4, zorder=2)
            ax.text(0.025, i, f"{row['v4']:.3f}→{row['v5']:.3f}",
                    ha="left", va="center", fontsize=7.5, color="#00664c")
    ax.plot(before, y-0.22, linestyle="none", marker="o", markersize=4,
            markerfacecolor="none", markeredgewidth=1, color=BLUE,
            label="v3", zorder=3)
    ax.plot(after, y, linestyle="none", marker="x", markersize=4,
            markeredgewidth=1.1, color=ORANGE, label="v4", zorder=4)
    ax.plot(current, y+0.22, linestyle="none", marker="D", markersize=3.5,
            color=GREEN, label="v5", zorder=5)
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
               bbox_to_anchor=(1.01, 0.995), ncol=3, frameon=False, fontsize=9,
               handlelength=0.6, columnspacing=0.6, handletextpad=0.3)
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.075, 0.075],
                             transform=fig.transFigure, color="#bbbbbb", linewidth=0.6))
    fig.text(0.02, 0.055, "Spoon: 10/36 entries unmeasured (no public scan)",
             va="top", fontsize=8)
    return fig


def object_score_chart(preview: Path | None, output_dir: Path = TARGET.parent) -> None:
    """Aggregate recorded pair scores, assert the published means, and draw all objects."""
    summary = json.loads(SCORES.read_text())
    packed = json.loads((PACKED_V5 / "pairs/summary.json").read_text())
    veto = json.loads((PACKED_V5 / "veto.json").read_text())
    ledger = [json.loads(line) for line in
              (PACKED_V5 / "pairs/ledger.jsonl").read_text().splitlines()]
    exact_pairs = {(row["episode"], row["slot"], row["object"], row["seed"], row["file"]): row
                   for row in ledger if row["status"] == "completed"}
    assert len(exact_pairs) == 52
    for pair in summary["pairs"]:
        key = (pair["episode"], pair["slot"], pair["object"], pair["seed"])
        assert np.isclose(pair["candidate"], exact_pairs[(*key, "v4")]["cd_o_cm"],
                          rtol=0, atol=1e-12)
    for pair in packed["pairs"]:
        key = (pair["episode"], pair["slot"], pair["object"], pair["seed"])
        for name in ("v4", "candidate"):
            assert np.isclose(pair[name], exact_pairs[(*key, name)]["cd_o_cm"],
                              rtol=0, atol=1e-12)
    rows = []
    for saved in summary["objects"]:
        pairs = [row for row in summary["pairs"] if row["object"] == saved["object"]]
        before = float(np.mean([row["v3"] for row in pairs]))
        after = float(np.mean([row["candidate"] for row in pairs]))
        assert len(pairs) == saved["pairs"]
        assert np.isclose(before, saved["v3"], rtol=0, atol=1e-12)
        assert np.isclose(after, saved["candidate"], rtol=0, atol=1e-12)
        current = [row["cd_o_cm"] for row in exact_pairs.values()
                   if row["object"] == saved["object"] and row["file"] == "candidate"]
        recorded = next(row for row in veto["objects"] if row["object"] == saved["object"])
        assert len(current) == len(pairs) == recorded["pairs"]
        assert np.isclose(np.mean(current), recorded["candidate"], rtol=0, atol=1e-12)
        rows.append({"object": saved["object"], "pairs": len(pairs),
                     "v3": before, "v4": after, "v5": float(np.mean(current))})
    rows.sort(key=lambda row: (row["v5"]-row["v4"], row["v4"]-row["v3"], LABELS[row["object"]]))
    versions = ("v3", "v4", "v5")
    means = {name: float(np.mean([row[name] for row in rows])) for name in versions}
    assert len(rows) == 10 and sum(row["pairs"] for row in rows) == 26
    assert f"{means['v3']:.4f}" == "0.7711"
    assert f"{means['v4']:.4f}" == "0.6534"
    assert np.isclose(means["v3"], summary["means"]["v3"]["per_object_cm"], rtol=0, atol=1e-12)
    assert np.isclose(means["v4"], summary["means"]["candidate"]["per_object_cm"], rtol=0, atol=1e-12)
    assert summary["total_pairs"] == 36 and len(summary["unscored"]) == 10
    assert all(row["object"] == "wooden_spoon" for row in summary["unscored"])
    pair_means = {name: sum(row[name]*row["pairs"] for row in rows)/26 for name in versions}
    assert f"{means['v5']:.4f}" == "0.5644"
    assert f"{pair_means['v5']:.4f}" == "0.6208"
    assert np.isclose(means["v5"], veto["means"]["per_object_cm"], rtol=0, atol=1e-12)
    assert np.isclose(pair_means["v5"], veto["means"]["pair_weighted_cm"], rtol=0, atol=1e-12)
    assert {row["object"] for row in rows if row["v5"] != row["v4"]} == {"white_pot", "water_pitcher"}
    fig = object_score_panel(rows)
    target = output_dir / TARGET.name
    fig.savefig(target, metadata={"Title": "Packed v3/v4/v5 CD-O on all ten scanned objects",
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
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_proj_type("ortho")
    ax.view_init(elev=40, azim=-65)
    ax.set_axis_off()


def qualitative_panel(rows: list[dict]) -> plt.Figure:
    """Show the two fixed G1/G2 development examples, with independent display scales."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.18, 1.85))
    for j, title in enumerate(("Frame-0 crop", "Generated", "v4", "Scan")):
        fig.text(0.17 + j*0.235, 0.99, title, ha="center", va="top",
                 fontsize=7.5, weight="bold")
    for i, row in enumerate(rows):
        bottom = 0.51 - i*0.44
        fig.text(0.015, bottom+0.20, row["label"], rotation=90,
                 ha="center", va="center", fontsize=8, weight="bold")
        ax = fig.add_axes((0.055, bottom, 0.235, 0.40))
        ax.imshow(row["crop"])
        ax.set_axis_off()
        for j, key in enumerate(("generated", "v4", "scan"), 1):
            ax = fig.add_axes((0.055+j*0.235, bottom, 0.235, 0.40), projection="3d")
            draw_mesh(ax, *row[key], "#B5B5B5")
    return fig


def qualitative_chart(preview: Path | None, output_dir: Path = TARGET.parent) -> None:
    """Read frozen crops and budgeted meshes; perform no generation, fitting or scoring."""
    # The public-21 pitcher and tuning-42 pot are the pilot shapes reused by G2.
    # Match the recorded pilot's display-only Y-up to Z-up rotation for generation.
    generated_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # Fixed display rotations expose each scan's opening and handle, independently
    # of generation and scoring; no geometric registration or score-based fitting.
    scan_rotations = {
        "water_pitcher": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        "white_pot": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    }
    rows = []
    for name, label, episode in (("water_pitcher", "Pitcher", 21), ("white_pot", "Pot", 42)):
        directory = GENERATIVE / name
        row = {"label": label, "crop": plt.imread(directory / "crop.png")}
        with np.load(directory / "scores.npz", allow_pickle=False) as arrays:
            for key, prefix, rotation in (
                ("generated", "", generated_rotation),
                ("v4", "v4_", np.eye(3)),
                ("scan", "scan_", scan_rotations[name]),
            ):
                vertices, faces = arrays[prefix+"vertices"], arrays[prefix+"faces"].astype(int)
                triangles = vertices[faces]
                # Kit budgets may include degenerate padding: omit only zero-area faces.
                valid = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                                triangles[:, 2]-triangles[:, 0]), axis=1) > 0
                faces = faces[valid]
                # Ignore unused padding vertices when normalising display bounds.
                used, remapped = np.unique(faces, return_inverse=True)
                row[key] = (normalised_vertices(vertices[used], rotation), remapped.reshape(-1, 3))
        rows.append(row)
        print(json.dumps({"object": name, "public_episode": episode,
                          "crop": str((directory / "crop.png").relative_to(ROOT)),
                          "meshes": str((directory / "scores.npz").relative_to(ROOT))}))
    fig = qualitative_panel(rows)
    # Keep the existing build command and asset name; Figure 2 now describes v5.
    target = output_dir / "d1-v4-qualitative.pdf"
    fig.savefig(target, dpi=300, metadata={"Title": "G2 development crops, pretrained shapes, v4 and public scans",
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
    parser.add_argument("--qualitative", action="store_true", help="draw recorded pitcher/pot crops, generated meshes, v4 and scans")
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
