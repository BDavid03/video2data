"""Rebuild the plain-language report from recorded outputs, on CPU only.

Run: CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B \
    scripts/report/nontechnical_figures.py
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/outputs/visuals"
REPORT = ROOT / "docs/report/nontechnical"
FIGURES = REPORT / "figures"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLCONFIGDIR"] = str(OUTPUT / "matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

BLUE, AMBER, GREY, INK = "#2a78d6", "#d97706", "#6b7280", "#202833"
DYNAMIC = "data/outputs/dynamic_feasibility"
PARTIAL = f"{DYNAMIC}/m0/runs/D1p-seed0/attempt0/trajectory.parquet"
FINISHED = f"{DYNAMIC}/m0c/scoring/runs/steps-66763776/attempt0/trajectory.parquet"
FK = f"{DYNAMIC}/m0b/clearance/fk-reference.npz"
ASSETS = ("data/.runtime/flashchord/video_to_data/robotic_grounding/source/"
          "robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed")
OUTCOME = "docs/chain/moving-m0b-outcome.md"
MANIFEST: dict = {"description": "CPU presentation of saved results; no new model execution.",
                  "inputs": {}, "artifacts": {}}


def read_bytes(relative: str, used: list[str]) -> bytes:
    """Read a local input once per use and retain its content hash."""
    path = ROOT / relative
    path.resolve().relative_to(ROOT)
    data = path.read_bytes()
    MANIFEST["inputs"][relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if relative not in used:
        used.append(relative)
    return data


def read_json(relative: str, used: list[str]) -> dict:
    return json.loads(read_bytes(relative, used))


def trajectory(relative: str, used: list[str]) -> pd.DataFrame:
    """Read only the explicitly named one-object example export."""
    frame = pd.read_parquet(io.BytesIO(read_bytes(relative, used))).sort_values("step")
    if len(frame) != 390 or frame.environment_id.nunique() != 1 or frame.object_id.nunique() != 1:
        raise ValueError("Expected the recorded 390-frame, one-world tissue-box export")
    if not frame.environment_valid.all() or set(frame.object_body_name) != {"tissue_box_simple"}:
        raise ValueError("Expected a valid tissue-box export")
    return frame


def positions(frame: pd.DataFrame, kind: str) -> np.ndarray:
    return frame[[f"{kind}_position_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float64)


def box_frames(frame: pd.DataFrame, kind: str, corners: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Transform the mesh's local bounding box with recorded xyzw poses, in cm."""
    quaternion = frame[[f"{kind}_quaternion_{axis}" for axis in "xyzw"]].to_numpy()
    matrices = Rotation.from_quat(quaternion).as_matrix()
    return (np.einsum("tij,kj->tki", matrices, corners)
            + positions(frame, kind)[:, None] - origin) * 100


def register(path: Path, used: list[str], caption: str, **details) -> None:
    MANIFEST["artifacts"][str(path.relative_to(ROOT))] = {
        "inputs": {p: MANIFEST["inputs"][p]["sha256"] for p in used},
        "caption": caption, **details,
    }


def save(fig, name: str, used: list[str], caption: str) -> None:
    """Save an opaque publication-size PNG with its input attribution."""
    path = FIGURES / name
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    register(path, used, caption)


def footnote(fig, sentence: str, width: int = 112) -> None:
    fig.text(.06, .025, textwrap.fill(sentence, width), fontsize=10, color=INK, va="bottom")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c5c9ce",
        "axes.facecolor": "white", "figure.facecolor": "white", "axes.grid": True,
        "grid.color": "#e7e9ec", "grid.linewidth": .5, "axes.axisbelow": True,
        "lines.linewidth": .8, "legend.frameon": False, "legend.fontsize": 10,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 14,
    })


def height_chart() -> None:
    used = []
    frames = [trajectory(PARTIAL, used), trajectory(FINISHED, used)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.72, wspace=.08)
    fig.suptitle("A trained policy now moves the box along its goal path.", x=.06, ha="left", y=.97, fontsize=19)
    fig.text(.06, .87, "Supplied tissue-box example • height above the goal's starting height", fontsize=12)
    for ax, frame, title in zip(axes, frames, ["Earlier partial policy", "Finished policy · fresh start"], strict=True):
        zero = frame.reference_position_z.iloc[0]
        ax.plot(frame.time_s, (frame.reference_position_z - zero) * 100, color=BLUE, label="Goal path")
        ax.plot(frame.time_s, (frame.achieved_position_z - zero) * 100, color=AMBER,
                linestyle="--", label="What the policy did")
        ax.set(title=title, xlabel="Clip time (seconds)", xlim=(0, 13), ylim=(-2, 20))
        ax.set_xticks([0, 3, 6, 9, 12])
    axes[0].set_ylabel("Height above start (cm)")
    axes[0].text(8.1, 17.2, "Goal", fontsize=10)
    axes[0].text(7, 2.2, "Partial policy", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.52, .02), ncol=2)
    save(fig, "01-box-height.png", used,
         "On the supplied example, the earlier partial policy barely lifts the box; the completed fresh-start policy follows the lift and return.")


def learning_chart() -> dict:
    used = []
    a = read_json(f"{DYNAMIC}/m0b/final.json", used)
    b = read_json(f"{DYNAMIC}/m0c/final.json", used)
    for relative in [f"{DYNAMIC}/m0b/summary.md", f"{DYNAMIC}/m0c/summary.md", OUTCOME]:
        read_bytes(relative, used)
    run = read_json(f"{DYNAMIC}/m0b/trials/train/run.json", used)
    progress = run["progress"]
    step = 32814080
    before = max((p for p in progress if p["steps"] <= step), key=lambda p: p["steps"])
    after = min((p for p in progress if p["steps"] >= step), key=lambda p: p["steps"])
    bounds = [before["seconds"] / 60, after["seconds"] / 60]
    warm_x = [bounds[1], run["timing"]["wall_seconds"] / 60]
    warm_y = [a["rows"][f"steps-{n}"]["add_auc"] for n in (32814080, 66763776)]
    fresh_x = [15.0, 26.6, 49.3]
    fresh_y = [b["rows"][f"steps-{n}"]["add_auc"] for n in (16154624, 32814080, 66763776)]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    fig.subplots_adjust(left=.1, right=.97, bottom=.29, top=.73)
    fig.suptitle("Training gets there quickly.", x=.06, ha="left", y=.97, fontsize=19)
    fig.text(.06, .88, "Example task only • ADD-AUC: a 0-to-1 fit score; higher is closer to the goal", fontsize=12)
    ax.plot(fresh_x, fresh_y, "o-", color=BLUE, markersize=5, label="Seed 1 · from nothing")
    ax.plot(warm_x, warm_y, "s--", color=AMBER, markersize=5, label="Seed 0 · resumed earlier training")
    ax.errorbar(warm_x[0], warm_y[0], xerr=[[bounds[1] - bounds[0]], [0]], color=AMBER, capsize=3)
    ax.axhline(a["rows"]["S"]["add_auc"], color=GREY, linestyle=":", label="Held still · 0.459")
    ax.axhline(a["rows"]["D1p-seed0"]["add_auc"], color=GREY, linestyle="-.", label="Earlier partial policy · 0.444")
    ax.annotate("0.961 after 15.0 min", (fresh_x[0], fresh_y[0]), xytext=(19, .81),
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": .7}, fontsize=10)
    ax.text(36, .90, "Finished: 0.979–0.981", fontsize=10)
    ax.annotate(f"0.963 by {bounds[1]:.2f} min*", (warm_x[0], warm_y[0]), xytext=(1, .71),
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": .7}, fontsize=10)
    ax.set(xlim=(0, 55), ylim=(0, 1.05), xlabel="Minutes in this training run", ylabel="Fit to the goal (ADD-AUC)")
    ax.set_yticks(np.arange(0, 1.01, .2))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .12), ncol=2)
    footnote(fig, f"*Seed 0's middle checkpoint is bounded by logged times {bounds[0]:.2f}–{bounds[1]:.2f} min; shown at the upper bound. Its clock excludes earlier training. No smoothing; all evaluations remove assistance.")
    save(fig, "02-learning.png", used,
         "The fresh run reached 0.961 in 15.0 minutes; the resumed run uses a separate clock excluding its training prefix, and its middle checkpoint has only a logged time bound.")
    return {"seed0_checkpoint_time_bound_minutes": bounds, "seed0_final_minutes": warm_x[-1],
            "seed1_minutes": fresh_x, "seed1_auc": fresh_y,
            "seed0_auc": warm_y, "partial_auc": a["rows"]["D1p-seed0"]["add_auc"]}


def error_chart() -> dict:
    used = []
    frame = trajectory(FINISHED, used)
    recorded = read_json(f"{DYNAMIC}/m0c/final.json", used)["rows"]["steps-66763776"]
    error = np.linalg.norm(positions(frame, "achieved") - positions(frame, "reference"), axis=1) * 100
    difference = abs(error.mean() - recorded["mean_position_cm"])
    if difference > 1e-9:
        raise ValueError("Raw trajectory and recorded mean error disagree")
    peak = int(np.argmax(error))
    early_peak = int(np.argmax(error[frame.time_s.to_numpy() <= 2]))
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    fig.subplots_adjust(left=.1, right=.97, bottom=.2, top=.74)
    fig.suptitle("How far off the box is, moment by moment.", x=.06, ha="left", y=.97, fontsize=18)
    fig.text(.06, .87, "Finished fresh-start policy • supplied example task", fontsize=12)
    ax.axvspan(0, 2, facecolor=GREY, alpha=.09)
    ax.plot(frame.time_s, error, color=BLUE)
    ax.annotate("Early lag", (frame.time_s.iloc[early_peak], error[early_peak]), xytext=(3.2, error.max() * .92),
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": .7}, fontsize=11)
    ax.text(7.2, error.max() * .80, f"Mean error: {error.mean():.3f} cm", fontsize=11)
    ax.set(xlim=(0, 13), ylim=(0, error.max() * 1.18), xlabel="Clip time (seconds)", ylabel="Position error (cm)")
    footnote(fig, "The shaded first two seconds show the lag at the start; this is position error, not a measure of how naturally the hands grasp.")
    save(fig, "03-position-error.png", used,
         f"The finished fresh-start policy averages {error.mean():.3f} cm of position error; the shaded first two seconds expose its initial lag.")
    return {"mean_position_cm": float(error.mean()), "maximum_position_cm": float(error.max()),
            "maximum_error_time_s": float(frame.time_s.iloc[peak]), "recorded_mean_absolute_difference_cm": difference}


def entry_chart() -> None:
    used = []
    read_bytes("docs/chain/COMPLETION-PLAN.md", used)
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    fig.subplots_adjust(left=.36, right=.96, bottom=.25, top=.76)
    fig.suptitle("Where the prepared entry stands.", x=.06, ha="left", y=.97, fontsize=19)
    fig.text(.06, .87, "Challenge training episodes • current static entry and a perfect-start static ceiling", fontsize=11)
    ax.barh([1, 0], [.277, .532], height=.48, color=[BLUE, GREY], edgecolor=INK, linewidth=.4)
    ax.patches[1].set_hatch("///")
    ax.set_yticks([1, 0], ["Our entry\n(objects stay still)", "Best possible if held still\n(perfect starting poses)"])
    ax.set(xlim=(0, 1), ylim=(-.6, 1.6), xlabel="AUC: 0-to-1 fit score (higher is better)")
    ax.grid(axis="y", visible=False)
    for y, value in [(1, .277), (0, .532)]:
        ax.text(value + .025, y, f"{value:.3f}", va="center", fontsize=12)
    footnote(fig, "Reproducing motion is the route that could go beyond the static ceiling; no challenge moving policy has yet been demonstrated.")
    save(fig, "04-entry.png", used,
         "The current entry scores 0.277 on training episodes; even perfect starting poses held still reach only 0.532, so motion is a major missing part.")


def pose_table() -> str:
    used = []
    panel = read_json("data/outputs/pose_insurance/frame0-panel/stage-c/summary.json", used)
    fp = read_json("data/outputs/foundationpose/pilot/summary.json", used)
    lines = ["| Pair | Current method (T1) | Geometric registration (TW) | FoundationPose |",
             "| --- | ---: | ---: | ---: |"]
    rows = {}
    for episode in (12, 31):
        t1 = panel["results"]["0"]["T1"].get(str(episode))
        old = f"{t1['frame0']['offset_error_m'] * 100:.2f} cm" if t1 else "Not recorded"
        # TW4 retains T1 on 12; it is not an independent registration estimate there.
        tw = (f"{panel['results']['0']['TW4']['31']['frame0']['offset_error_m'] * 100:.2f} cm"
              if episode == 31 else "Not changed from T1")
        score = next(p["offset_error_cm"] for p in fp["pairs"] if p["episode"] == episode)
        lines.append(f"| {episode} | {old} | {tw} | {score:.2f} cm |")
        rows[str(episode)] = {"T1": old, "TW": tw, "FoundationPose_cm": score}
    path = FIGURES / "05-frame-zero-table.md"
    path.write_text("\n".join(lines) + "\n")
    register(path, used, "Error in one object's position relative to the other at the first frame, in centimetres; two already-used training pairs.", values=rows)
    return "\n".join(lines)


def camera_sheets() -> None:
    used = []
    read_bytes("docs/chain/foundationpose-pilot-outcome.md", used)
    specs = [(12, "blue_cup", "Cup: rim and handle sit closely", (220, 340, 650, 576)),
             (12, "plastic_dish_rack", "Rack: close fit; some doubled pegs", (175, 315, 670, 560)),
             (31, "mini_sweeper", "Sweeper: bristle end narrow and offset", (375, 280, 510, 435)),
             (31, "plastic_dish_rack", "Rack: fit ignores the foreground sweeper", (155, 270, 645, 500))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.subplots_adjust(left=.035, right=.97, bottom=.12, top=.84, wspace=.08, hspace=.34)
    fig.suptitle("What FoundationPose thinks.", x=.035, ha="left", y=.98, fontsize=21)
    fig.text(.035, .925, "Recorded camera images, cropped equally before and after the model overlay", fontsize=12)
    for ax, (ep, name, title, crop) in zip(axes.flat, specs, strict=True):
        relative = f"data/outputs/foundationpose/pilot/amendment1/episode_{ep:06}/{name}/rendered-overlay.png"
        source = Image.open(io.BytesIO(read_bytes(relative, used))).convert("RGB")
        half = source.width // 2
        x0, y0, x1, y1 = crop
        left = source.crop((x0, y0, x1, y1))
        right = source.crop((half + x0, y0, half + x1, y1))
        pair = Image.new("RGB", (left.width * 2 + 8, left.height), "white")
        pair.paste(left); pair.paste(right, (left.width + 8, 0))
        ax.imshow(pair)
        ax.set_title(f"Pair {ep} · {title}", fontsize=11, loc="left", pad=25)
        ax.text(.25, 1.02, "Camera", transform=ax.transAxes, ha="center", fontsize=10)
        ax.text(.76, 1.02, "Model over camera", transform=ax.transAxes, ha="center", fontsize=10)
        ax.axis("off")
    footnote(fig, "These are first-frame estimates using scans on two already-used training pairs, outside the entry. The sweeper is enlarged. The model's licence is noncommercial; its overlay ignores foreground occlusion.", 133)
    save(fig, "06-foundationpose.png", used,
         "The cup and racks look closely placed, but doubled pegs and the sweeper's narrow, offset bristle end show limits; these first-frame scan-based estimates are outside the entry.")

    used = []
    read_bytes("docs/chain/frame0-panel-outcome.md", used)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.subplots_adjust(left=.035, right=.98, bottom=.075, top=.88, hspace=.21)
    fig.suptitle("What the masks find—and where they fail.", x=.035, ha="left", y=.98, fontsize=20)
    fig.text(.035, .935, "A mask marks the pixels assigned to each named object; boxes show its detected area.", fontsize=12)
    for ax, ep, title in zip(axes, (12, 31, 0),
                             ("Pair 12 · cup and rack separated", "Pair 31 · sweeper and rack separated",
                              "Pair 0 · failed: pot and lid boxes nearly identical"), strict=True):
        path = f"data/outputs/pose_insurance/frame0-panel/stage-a/episode_{ep:06}/contact-sheet.png"
        if (ROOT / path).exists():
            ax.imshow(Image.open(io.BytesIO(read_bytes(path, used))))
        else:
            ax.text(.5, .5, "Recorded contact sheet missing", transform=ax.transAxes, ha="center")
        ax.set_title(title, loc="left", fontsize=12, pad=8)
        ax.axis("off")
    footnote(fig, "Saved first-frame contact sheets: camera at left, each object's mask at centre and right. Almost identical pot/lid boxes failed admission; they are not a trustworthy separation.", 133)
    save(fig, "07-masks.png", used,
         "The recorded masks separate the cup and sweeper from the rack, but the pot and lid receive almost identical boxes and fail the input check.")


def scene_axes(ax, limits: np.ndarray, title: str) -> None:
    """Use identical centimetre axes and camera angles for both policy drawings."""
    ax.set(xlim=limits[:, 0], ylim=limits[:, 1], zlim=limits[:, 2],
           xlabel="Sideways (cm)", ylabel="Depth (cm)", zlabel="Height (cm)")
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_box_aspect(np.ptp(limits, axis=0))
    ax.view_init(elev=23, azim=-62)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-10, 0, 10])
    ax.set_zticks([-10, 0, 10, 20, 30])
    ax.tick_params(labelsize=9, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"].update(color="#e7e9ec", linewidth=.4)
        axis.label.set_fontsize(10)


def policy_visuals() -> dict:
    used = []
    earlier = trajectory(PARTIAL, used)
    finished = trajectory(FINISHED, used)
    read_bytes(OUTCOME, used)
    urdf_path = f"{ASSETS}/tissue_box_simple_rigid.urdf"
    urdf = ET.fromstring(read_bytes(urdf_path, used))
    collision = urdf.find(".//collision")
    mesh_spec = collision.find("geometry/mesh")
    if (collision.find("origin").attrib != {"rpy": "0 0 0", "xyz": "0 0 0"}
            or mesh_spec.attrib["scale"] != "1 1 1"):
        raise ValueError("Expected the shipped identity mesh transform")
    mesh_relative = str((Path(ASSETS) / mesh_spec.attrib["filename"]).as_posix())
    mesh = trimesh.load(io.BytesIO(read_bytes(mesh_relative, used)), file_type="stl", force="mesh")
    bounds = mesh.bounds
    corners = np.array(list(itertools.product(*bounds.T)))
    edges = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                      if np.count_nonzero(corners[i] != corners[j]) == 1])
    origin = positions(finished, "reference")[0]
    goal = box_frames(finished, "reference", corners, origin)
    actual = [box_frames(frame, "achieved", corners, origin) for frame in (earlier, finished)]
    if not np.allclose(positions(earlier, "reference"), positions(finished, "reference"), atol=1e-9):
        raise ValueError("Example goal paths differ")
    hands = None
    if (ROOT / FK).exists():
        read_bytes("src/features/dynamic_feasibility/hand_clearance.py", used)
        with np.load(io.BytesIO(read_bytes(FK, used)), allow_pickle=False) as saved:
            hands = (saved["points"].astype(float) - origin) * 100
        if hands.shape != (390, 12, 3):
            raise ValueError("Expected two wrists and ten fingertips per reference frame")
    all_points = np.concatenate([goal.reshape(-1, 3), *[a.reshape(-1, 3) for a in actual],
                                *([] if hands is None else [hands.reshape(-1, 3)])])
    limits = np.array([all_points.min(0) - 3, all_points.max(0) + 3])
    goal_path = (positions(finished, "reference") - origin) * 100
    paths = [(positions(frame, "achieved") - origin) * 100 for frame in (earlier, finished)]
    hand_edges = [(wrist, wrist + tip) for wrist in (0, 6) for tip in range(1, 6)]
    handles = [Line2D([], [], color=BLUE, ls=":", label="Goal box and trail"),
               Line2D([], [], color=AMBER, ls="-", label="Policy box and trail")]
    if hands is not None:
        handles.append(Line2D([], [], color=GREY, alpha=.5, label="Demonstration's wrist/fingertip guides"))

    def draw(ax, arm: int, frame: int) -> None:
        ax.add_collection3d(Line3DCollection(goal[frame][edges], colors=BLUE, linewidths=.9, linestyles=":"))
        ax.add_collection3d(Line3DCollection(actual[arm][frame][edges], colors=AMBER, linewidths=.9))
        ax.plot(*goal_path[:frame + 1].T, color=BLUE, ls=":")
        ax.plot(*paths[arm][:frame + 1].T, color=AMBER)
        if hands is not None:
            ax.add_collection3d(Line3DCollection(hands[frame][hand_edges], colors=GREY, linewidths=.6, alpha=.3))
            ax.scatter(*hands[frame].T, color=GREY, s=3, alpha=.3)

    contact_indices = [0, 40, 120, 195, 285, 389]
    for arm, name, title in [(0, "08-partial-policy.png", "The earlier policy leaves most of the lift undone."),
                              (1, "09-finished-policy.png", "The finished policy's box follows the goal through the lift.")]:
        fig = plt.figure(figsize=(12, 8.6))
        fig.subplots_adjust(left=.03, right=.94, bottom=.15, top=.81, wspace=.22, hspace=.30)
        fig.suptitle(title, x=.035, ha="left", y=.98, fontsize=19)
        fig.text(.035, .92, "Supplied example • schematic boxes at recorded positions and rotations • same scale in every panel", fontsize=11)
        for n, frame in enumerate(contact_indices):
            ax = fig.add_subplot(2, 3, n + 1, projection="3d")
            scene_axes(ax, limits, f"{finished.time_s.iloc[frame]:.2f} seconds")
            draw(ax, arm, frame)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, fontsize=9)
        footnote(fig, "Faint guides show the demonstration's hands, not the policy's. The evaluator exported no policy hands. Boxes use the shipped mesh's local bounds; trails show object origins.", 133)
        save(fig, name, used, "Six matched moments from the " + ("earlier partial" if arm == 0 else "finished fresh-start")
             + " policy; the faint wrist/fingertip skeletons belong to the demonstration, not the policy.")

    # Update existing artists for the video rather than repeatedly creating 3D scenes.
    fig = plt.figure(figsize=(10, 6), dpi=160)
    fig.subplots_adjust(left=.04, right=.94, bottom=.19, top=.75, wspace=.15)
    fig.suptitle("A trained policy now moves the box along its goal path.", x=.04, ha="left", y=.98, fontsize=19)
    clock = fig.text(.04, .89, "", fontsize=12)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .08), ncol=3, fontsize=8)
    fig.text(.04, .035, "Demonstration's hands, not the policy's. Policy hands were not exported. Example task only.", fontsize=10)
    artists = []
    for arm, title in enumerate(["Earlier partial policy", "Finished policy · fresh start"]):
        ax = fig.add_subplot(1, 2, arm + 1, projection="3d")
        scene_axes(ax, limits, title)
        wire_goal = Line3DCollection(goal[0][edges], colors=BLUE, linewidths=.9, linestyles=":")
        wire_actual = Line3DCollection(actual[arm][0][edges], colors=AMBER, linewidths=.9)
        skeleton = Line3DCollection(np.zeros((1, 2, 3)), colors=GREY, linewidths=.6, alpha=.3)
        for collection in (wire_goal, wire_actual, skeleton):
            ax.add_collection3d(collection)
        trail_goal, = ax.plot([], [], [], color=BLUE, ls=":")
        trail_actual, = ax.plot([], [], [], color=AMBER)
        artists.append((wire_goal, wire_actual, skeleton, trail_goal, trail_actual))
    video_path = FIGURES / "10-policy-comparison.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1600, 960))
    opened = writer.isOpened()
    gif_frames = []
    indices = list(range(0, len(finished), 3))
    if indices[-1] != len(finished) - 1:
        indices.append(len(finished) - 1)
    try:
        for frame in indices:
            clock.set_text(f"Clip time {finished.time_s.iloc[frame]:05.2f} seconds • original timing, sampled at 10 fps")
            for arm, (wire_goal, wire_actual, skeleton, trail_goal, trail_actual) in enumerate(artists):
                wire_goal.set_segments(goal[frame][edges])
                wire_actual.set_segments(actual[arm][frame][edges])
                trail_goal.set_data_3d(*goal_path[:frame + 1].T)
                trail_actual.set_data_3d(*paths[arm][:frame + 1].T)
                if hands is not None:
                    skeleton.set_segments(hands[frame][hand_edges])
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            if opened:
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            gif_frames.append(Image.fromarray(rgb).resize((1000, 600), Image.Resampling.LANCZOS))
    finally:
        writer.release()
        plt.close(fig)
    gif_path = FIGURES / "10-policy-comparison.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=100, loop=0, optimize=False)
    caption = "Goal and achieved box positions and rotations, side by side at original clip speed; faint hands belong only to the demonstration, and the evaluator exported no policy hands."
    register(gif_path, used, caption, frames=len(indices), sampled_source_frames=indices)
    if opened:
        register(video_path, used, caption, frames=len(indices), fps=10, dimensions=[1600, 960])
    return {"mesh_bounds_m": bounds.tolist(), "mesh_dimensions_cm": (mesh.extents * 100).tolist(),
            "demonstration_hand_guides": hands is not None, "video_written": opened,
            "video_frames": len(indices), "trajectory_frames": len(finished),
            "contact_sheet_source_frames": contact_indices}


def harder_task() -> None:
    used = []
    read_bytes(OUTCOME, used)
    read_bytes("docs/chain/moving-m2-protocol.md", used)
    source = Image.open(io.BytesIO(read_bytes("data/outputs/moving_policy/ep39/strip.png", used)))
    fig, ax = plt.subplots(figsize=(13.3, 5.5))
    fig.subplots_adjust(left=.02, right=.99, top=.77, bottom=.16)
    fig.suptitle("The harder task ahead.", x=.035, ha="left", y=.98, fontsize=21)
    fig.text(.035, .88, "Episode 39 • a constructed reference, not a trained policy rollout", fontsize=12)
    ax.imshow(source)
    ax.axis("off")
    footnote(fig, "This reference peaks near 71 cm/s, about 8 times the example box's 9 cm/s, and rotates 109 degrees. The original setup's checks stopped before training; the saved strip is not a result of learning.", 148)
    save(fig, "11-harder-task.png", used,
         "Episode 39's reference moves about eight times faster than the box; the original setup's checks stopped before training, and this strip shows a constructed reference rather than a policy.")


def review_previews() -> dict:
    """Decode the finished animation and make compact visual review sheets."""
    grey = Image.new("RGB", (1600, 900), "white")
    for n, name in enumerate(["01-box-height.png", "02-learning.png", "03-position-error.png", "04-entry.png"]):
        with Image.open(FIGURES / name) as source:
            preview = source.convert("L").convert("RGB")
            preview.thumbnail((800, 450), Image.Resampling.LANCZOS)
            grey.paste(preview, ((n % 2) * 800, (n // 2) * 450))
    grey.save(OUTPUT / "review-greyscale.png", dpi=(180, 180))
    animation = Image.open(FIGURES / "10-policy-comparison.gif")
    contact = Image.new("RGB", (2000, 1200), "white")
    for n, frame in enumerate([0, 13, 65, animation.n_frames - 1]):
        animation.seek(frame)
        contact.paste(animation.convert("RGB"), ((n % 2) * 1000, (n // 2) * 600))
    contact.save(OUTPUT / "review-animation.png", dpi=(180, 180))
    decoded = 0
    capture = cv2.VideoCapture(str(FIGURES / "10-policy-comparison.mp4"))
    while capture.isOpened():
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded and decoded != animation.n_frames:
        raise ValueError("MP4 and GIF frame counts differ")
    return {"gif_frames_decoded": animation.n_frames, "mp4_frames_decoded": decoded,
            "review_sheets": ["data/outputs/visuals/review-greyscale.png",
                              "data/outputs/visuals/review-animation.png"]}


def write_report(table: str, metrics: dict) -> None:
    used = []
    for path in [OUTCOME, "docs/chain/moving-m0-outcome.md", "docs/chain/foundationpose-pilot-outcome.md",
                 "docs/chain/frame0-panel-outcome.md", "docs/chain/moving-m2-protocol.md",
                 "docs/chain/COMPLETION-PLAN.md"]:
        read_bytes(path, used)
    text = f"""# Seeing what the models do

Recorded state: 4 November 2026, after the two tissue-box runs and the original episode-39 checks.

## What the challenge asks

Watch a person's camera footage, reconstruct the objects, and reproduce their movement in simulation. A **policy** is a controller trained by trial and error in simulation. A **reference** is the goal path it tries to follow. **AUC** is a 0-to-1 fit score: higher means closer agreement across a range of allowed errors; it is not a percentage of tasks completed.

**No moving policy exists yet for any challenge episode. Our current entry's objects stay still.** The moving box below is the example task supplied with the tools, not a challenge result. This report describes the prepared entry and recorded experiments; it claims no upload or leaderboard score.

## What we have prepared to submit so far

The entry prepared for submission uses measured simple shapes plus two shapes made by a pretrained model. Its object poses come from short simulated policy runs; the objects remain at their starting positions. On challenge training episodes, its AUC is **0.277**. Even the best possible starting poses, held still, score only **0.532**. That ceiling shows why reproducing motion matters; it does not forecast what a moving policy would score.

![Our entry and the static ceiling](figures/04-entry.png)

The comparison uses challenge training episodes only; the example box's score is deliberately kept separate.

## What changed today

Two completed policies now move the supplied tissue box along its goal path. One resumed earlier training; the other started from nothing, with a different random seed. Their example-task fit scores are **0.981 and 0.979**, and mean position errors are **0.562 and 0.516 cm**. The earlier partial policy averaged **6.803 cm**. These are simulator exports with evaluation assistance removed.

The example is slow, mostly vertical, and comes with its own captured hands and mesh. It reaches only about 9 cm/s and rotates 5.4 degrees. **Its roughly 0.98 score is not a challenge score.** Several training settings changed together, so the records do not isolate which change caused the improvement.

## What the pictures show

![The earlier and finished box trajectories](figures/01-box-height.png)

The goal and achieved heights share the same axes: the earlier policy barely lifts the box, while the finished fresh-start policy follows the lift and return.

![Fit score at recorded checkpoints](figures/02-learning.png)

The fresh run reaches 0.961 in 15.0 minutes and finishes in 49.3; the resumed run finishes in 35.8 minutes after resuming, excluding prior training. Its middle checkpoint's time is bounded by progress logs, not timed exactly. Lines connect measured evaluations without smoothing; early checkpoints trained with assistance, removed during evaluation.

![Position error during the completed rollout](figures/03-position-error.png)

The first two seconds expose the initial lag; mean position error is {metrics['error']['mean_position_cm']:.3f} cm, with a peak of {metrics['error']['maximum_position_cm']:.2f} cm at {metrics['error']['maximum_error_time_s']:.2f} seconds.

![Six moments from the finished policy](figures/09-finished-policy.png)

Blue dotted outlines show the goal; amber solid outlines show what happened, with trails. These are schematic boxes using the shipped mesh's dimensions and the exported positions and rotations. Faint wrist-to-fingertip guides are **the demonstration's hands, not the policy's**; intermediate finger joints are not available. The evaluator export has no policy hands, so their actual grasp is unshown and its naturalness unmeasured.

[Earlier partial policy: the same six moments](figures/08-partial-policy.png) · [Side-by-side video](figures/10-policy-comparison.mp4)

### Telling apart two objects' position

First-frame error in one object's position relative to the other, in centimetres:

{table}

T1 is the current starting-pose method; registration fits shapes geometrically; FoundationPose is a pretrained image-and-shape pose model. Pair 12 has no separate panel registration estimate: that arm retained T1. These are two already-used training pairs, not independent validation. FoundationPose is outside the entry, uses scans here, and has a **noncommercial licence**.

![FoundationPose estimates over real camera frames](figures/06-foundationpose.png)

The cup and racks sit closely over the photograph, but some pegs double and the sweeper's bristle end looks too narrow and offset; foreground occlusion is ignored. First-frame agreement does not demonstrate tracking or reliably resolving a 180-degree flip.

![Recorded masks, including the pot and lid failure](figures/07-masks.png)

Masks mark pixels assigned to each object. Cup/rack and sweeper/rack are separated; the almost identical pot/lid boxes fail the input check.

![Episode 39's faster constructed reference](figures/11-harder-task.png)

This reference moves about eight times faster than the example box and rotates 109 degrees. Its original setup checks stopped before training: the saved strip is a proposed reference, not a learned rollout. A disclosed redesign is in the protocol; no success from it is claimed here.

## What is still unproven

Transferring the box result to a challenge episode still needs a usable object reconstruction, a reliable path from video, hands that can follow it, and a learned policy that moves the object physically. None of those links is established by the box's score. FoundationPose has only been screened on two used pairs and is not in the entry. The challenge result remains static.

### Placeholder — tracked object on a real camera clip

**Not available yet.** No per-frame tracking of a challenge episode exists in these records. The camera overlays above are single-frame pose estimates. A future clip would need recorded per-frame object poses and camera motion before a tracked object could honestly be drawn over the footage; no synthetic stand-in is shown.

---

Rebuild all figures, animations, the table and this report on CPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py
```

Input SHA-256 hashes and per-figure source mappings: `data/outputs/visuals/inputs.json`; numeric checks and build details: `data/outputs/visuals/build-summary.json`. Only saved outputs, the supplied box mesh and recorded notes are read; no model runs or challenge videos are opened.

Evidence: [box outcomes](../../chain/moving-m0b-outcome.md), [earlier policy](../../chain/moving-m0-outcome.md), [pose pilot](../../chain/foundationpose-pilot-outcome.md), [frame-zero panel](../../chain/frame0-panel-outcome.md), [episode-39 state](../../chain/moving-m2-protocol.md), [entry facts](../../chain/COMPLETION-PLAN.md#1-where-the-entry-really-stands).
"""
    path = REPORT / "report.md"
    path.write_text(text)
    register(path, list(MANIFEST["inputs"]), "Plain-language report, separate from the two-page abstract.")


def main() -> None:
    """Rebuild the fixed presentation set without running a model or reading new episodes."""
    for directory in (OUTPUT, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
        directory.resolve().relative_to(ROOT)
    style()
    height_chart()
    learning = learning_chart()
    error = error_chart()
    entry_chart()
    table = pose_table()
    camera_sheets()
    print("Charts and camera sheets saved; rendering recorded box motion on CPU.", flush=True)
    policy = policy_visuals()
    harder_task()
    review = review_previews()
    summary = {"learning": learning, "error": error, "policy_visuals": policy,
               "artifact_review": review,
               "missing": ["No per-frame challenge-object camera tracking exists.",
                           "No policy hands in evaluator export.",
                           "No independent panel registration estimate on pair 12.",
                           "Warm-start middle checkpoint has only a logged time bound."],
               "checks": {"CPU_only": os.environ["CUDA_VISIBLE_DEVICES"] == "",
                          "frames_from_recorded_exports": True, "new_model_execution": False}}
    write_report(table, summary)
    summary["files_written"] = sorted([
        "scripts/report/nontechnical_figures.py", "data/outputs/visuals/build-summary.json",
        "data/outputs/visuals/inputs.json", *MANIFEST["artifacts"],
        *[str(p.relative_to(ROOT)) for p in OUTPUT.rglob("*") if p.is_file()],
    ])
    summary["files_written"] = sorted(set(summary["files_written"]))
    (OUTPUT / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    source = Path(__file__)
    MANIFEST["generator"] = {"path": str(source.relative_to(ROOT)),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    for relative, record in MANIFEST["artifacts"].items():
        record["sha256"] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    (OUTPUT / "inputs.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    print(f"Built {len(MANIFEST['artifacts'])} artifacts from {len(MANIFEST['inputs'])} recorded inputs.", flush=True)


if __name__ == "__main__":
    main()
