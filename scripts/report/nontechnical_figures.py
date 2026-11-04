"""Rebuild the plain-language report from recorded outputs, on CPU only.

Run: CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B \
    scripts/report/nontechnical_figures.py
"""

from __future__ import annotations

import hashlib
import argparse
import io
import itertools
import json
import os
import subprocess
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/outputs/visuals"
REPORT = ROOT / "docs/report/nontechnical"
FIGURES = REPORT / "figures"

EP38_IN_ENTRY = True
POSE_SCORES = "none yet"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MPLCONFIGDIR"] = str(OUTPUT / "matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for thread_setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_setting] = "1"

import cv2
cv2.setNumThreads(1)
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
MANIFEST: dict = {"description": "CPU presentation of saved results and a separately recorded diagnostic policy replay.",
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


def mesh_box_corners(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return oriented mesh-bound corners in the original body-local frame.

    The tissue mesh is tilted within its body frame. An axis-aligned body-frame
    box therefore has the wrong orientation and inflated dimensions. Undo the
    oriented-bounds transform before applying any recorded body pose.
    """
    body_to_box, extents = trimesh.bounds.oriented_bounds(vertices)
    centered = np.array(list(itertools.product(*zip(-extents / 2, extents / 2))))
    return (centered - body_to_box[:3, 3]) @ body_to_box[:3, :3], extents


def transform_local_points(points: np.ndarray, pose_xyzw: np.ndarray) -> np.ndarray:
    """Apply one body pose, or a sequence, to the same body-local mesh geometry."""
    matrices = Rotation.from_quat(pose_xyzw[..., 3:]).as_matrix()
    return np.einsum("...ij,kj->...ki", matrices, points) + pose_xyzw[..., None, :3]


def box_frames(frame: pd.DataFrame, kind: str, corners: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Apply recorded xyzw poses to mesh-oriented corners, in centimetres."""
    quaternion = frame[[f"{kind}_quaternion_{axis}" for axis in "xyzw"]].to_numpy()
    pose = np.column_stack([positions(frame, kind), quaternion])
    return (transform_local_points(corners, pose) - origin) * 100


def orientation_error_degrees(achieved_xyzw: np.ndarray, reference_xyzw: np.ndarray) -> np.ndarray:
    """SO(3) geodesic angle between normalized xyzw rotations, without alignment."""
    return np.degrees((Rotation.from_quat(achieved_xyzw)
                       * Rotation.from_quat(reference_xyzw).inv()).magnitude())


def goal_box_audit() -> dict:
    """Record all achieved-to-goal angles independently of drawing geometry."""
    used = []
    base = "data/outputs/visuals/policy-hands/m0c"
    with np.load(io.BytesIO(read_bytes(f"{base}/replay.npz", used))) as data:
        replay = orientation_error_degrees(
            data["body_pose_xyzw"][:, int(data["object_body_ids"][0]), 3:].astype(float),
            data["reference_pose_xyzw"][:, 0, 3:].astype(float))
    official = trajectory(FINISHED, used)
    official_angles = orientation_error_degrees(
        official[[f"achieved_quaternion_{a}" for a in "xyzw"]].to_numpy(),
        official[[f"reference_quaternion_{a}" for a in "xyzw"]].to_numpy())
    summary = read_json(f"{base}/summary.json", used)
    with np.load(io.BytesIO(read_bytes(f"{base}/geometry.npz", used))) as data:
        vertices = np.concatenate([data[f"vertices_{i}"] for i, info in enumerate(summary["meshes"])
                                   if info["object"]])
    body_to_box, extents = trimesh.bounds.oriented_bounds(vertices)
    # Cuboid axes can be permuted and sign-flipped. Compare against the closest
    # proper axis permutation, so this measures the actual local mesh tilt.
    residuals = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            basis = np.eye(3)[list(permutation)] * np.array(signs)[:, None]
            if np.linalg.det(basis) > 0:
                residuals.append(Rotation.from_matrix(body_to_box[:3, :3] @ basis.T).magnitude())
    audit = {"definition": "Achieved-to-reference SO(3) geodesic angle from normalized xyzw quaternions; all 390 matched post-control-step samples, no alignment.",
             "cause": "Body-axis-aligned bounds were used for a mesh tilted within its body frame; quaternion order is correct.",
             "body_axis_bounds_dimensions_cm": (np.ptp(vertices, axis=0)*100).tolist(),
             "mesh_oriented_dimensions_cm": (extents*100).tolist(),
             "local_mesh_tilt_degrees_modulo_axis_permutation": float(np.degrees(min(residuals))),
             "body_to_oriented_bounds": body_to_box.tolist()}
    for name, angles in (("replay", replay), ("official", official_angles)):
        audit[name] = {"mean_degrees": float(angles.mean()), "max_degrees": float(angles.max()),
                       "max_step": int(angles.argmax()), "initial_degrees": float(angles[0]),
                       "per_step_degrees": angles.tolist()}
    path = OUTPUT / "goal-box-angle-audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    register(path, used, "Goal-box drawing audit: real tracking angles and mesh/body frame mismatch.")
    return {key: value for key, value in audit.items() if key not in ("replay", "official")} | {
        name: {key: value for key, value in audit[name].items() if key != "per_step_degrees"}
        for name in ("replay", "official")}


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
    corners, oriented_extents = mesh_box_corners(np.asarray(mesh.vertices))
    # Binary corner ordering: a cuboid edge changes exactly one index bit.
    edges = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                      if (i ^ j) in (1, 2, 4)])
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
        footnote(fig, "Faint guides show the demonstration's hands, not the policy's. The evaluator exported no policy hands. Box outlines follow the shipped mesh's oriented bounds; trails show object origins.", 133)
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
            "drawn_oriented_mesh_dimensions_cm": (oriented_extents * 100).tolist(),
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


def pan_learning() -> dict:
    """Present the three fixed M2 measurements and its official final height path."""
    used = []
    base = "data/outputs/moving_policy/ep39-a1"
    read_bytes("docs/chain/moving-m2-outcome.md", used)
    read_bytes(f"{base}/result.md", used)
    run = read_json(f"{base}/train.run.json", used)
    steps = (16154624, 32814080, 66763776)
    time_path = OUTPUT / "pan-checkpoint-times.json"
    if not time_path.exists():
        saved_times = []
        for step in steps:
            path = ROOT / base / f"training/policy_{step}.safetensors"
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            saved_times.append({"transitions": step, "path": str(path.relative_to(ROOT)),
                                "sha256": digest, "modified_unix": path.stat().st_mtime,
                                "minutes_from_handler_start": (path.stat().st_mtime-run["started_unix"])/60})
        time_path.write_text(json.dumps({"source": "Retained checkpoint save timestamps, including simulator startup; snapshotted before report rebuilds. These are policy-save times, not later evaluation times.",
                                        "handler_started_unix": run["started_unix"],
                                        "handler_finished_unix": run["finished_unix"],
                                        "checkpoints": saved_times}, indent=2)+"\n")
    timing = read_json("data/outputs/visuals/pan-checkpoint-times.json", used)
    rows = [read_json(f"{base}/score-{step}/mocap_metrics.json", used) for step in steps]
    minutes = [row["minutes_from_handler_start"] for row in timing["checkpoints"]]
    auc = [row["kit"]["AUC"] for row in rows]
    frame = pd.read_parquet(io.BytesIO(read_bytes(f"{base}/score-66763776/trajectory.parquet", used))).sort_values("step")
    if len(frame) != 319 or not frame.environment_valid.all() or frame.environment_id.nunique() != 1:
        raise ValueError("Expected the valid one-world, 319-step M2 export")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(left=.07, right=.98, bottom=.25, top=.73, wspace=.27)
    fig.suptitle("A pan policy learned the mocap path in under an hour.", x=.035, ha="left", y=.98, fontsize=21)
    fig.text(.035, .87, "Practice episode 39 • own reconstructed pan • one fresh-start seed • no video-derived goal", fontsize=12)
    ax = axes[0]
    ax.axhline(.17, color=GREY, ls="--", label="Held still: about 0.17")
    ax.plot(minutes, auc, "o-", color=AMBER, label="Measured saved policies")
    for x, y, label in zip(minutes, auc, ("16M", "33M", "67M"), strict=True):
        ax.annotate(f"{label}: {y:.3f}", (x, y), xytext=(0, -22 if y > .9 else 12),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set(xlabel="Training wall time when saved (minutes)", ylabel="Mocap fit score (AUC)",
           xlim=(0, 62), ylim=(0, 1.08), title="Three recorded measurements")
    ax.legend(loc="center right", fontsize=9)
    zero = float(frame.reference_position_z.iloc[0])
    axes[1].plot(frame.time_s, (frame.reference_position_z-zero)*100, color=BLUE, label="Mocap goal")
    axes[1].plot(frame.time_s, (frame.achieved_position_z-zero)*100, color=AMBER, ls="--", label="Official policy rollout")
    axes[1].set(xlabel="Clip time (seconds)", ylabel="Height above goal start (cm)",
                title="Finished policy: goal and achieved height", xlim=(0, float(frame.time_s.max())))
    axes[1].legend(fontsize=9)
    footnote(fig, "Not a challenge result; scored episodes have no motion capture. A video-trained practice policy is now recorded; its result is shown separately below. Lines connect measurements without smoothing. Save times include startup; the full run took 57.98 minutes.", 151)
    save(fig, "14-pan-learning.png", used,
         "Not a challenge result; scored episodes have no motion capture. A video-trained practice policy is now recorded; its result is shown separately below.")
    return {"checkpoint_transitions": list(steps), "checkpoint_save_minutes": minutes,
            "auc": auc, "held_still_auc": .17, "training_wall_minutes": (run["finished_unix"]-run["started_unix"])/60,
            "final_position_error_cm": rows[-1]["E_m"]*100,
            "final_angle_error_degrees": rows[-1]["Theta_deg"], "final_T": rows[-1]["motion"]["T"],
            "final_Q": rows[-1]["motion"]["Q"], "final_SP_SR": rows[-1]["kit"]["SP_SR"],
            "final_MPPE_cm": rows[-1]["kit"]["MPPE_cm"], "height_source": "Official final export"}


def policy_hands_visuals(base="data/outputs/visuals/policy-hands/m0c",
                         figure_stem="12-policy-hands", pan=False) -> dict:
    """Draw actual recorded links, using the scene's own simplified visual meshes."""
    used = []
    summary = read_json(f"{base}/summary.json", used)
    contacts = read_json(f"{base}/contacts.json", used)
    with np.load(io.BytesIO(read_bytes(f"{base}/replay.npz", used))) as data:
        poses, reference, times, object_ids = (data[k] for k in
            ("body_pose_xyzw", "reference_pose_xyzw", "time_s", "object_body_ids"))
    with np.load(io.BytesIO(read_bytes(f"{base}/geometry.npz", used))) as data:
        meshes = [(info, data[f"vertices_{i}"], data[f"faces_{i}"])
                  for i, info in enumerate(summary["meshes"])]
    high_carry = poses[:, int(object_ids[0]), 2] - poses[0, int(object_ids[0]), 2] > .05
    both = np.array([all(any(f"{side}_" in p["other_body"] for p in row["pairs"])
                         for side in ("left", "right")) for row in contacts])
    facts = {"high_carry_frames": int(high_carry.sum()),
             "both_hands_high_carry_frames": int((high_carry & both).sum()),
             "visual_description": "The box sits between two curled hands at opposite ends. The thumbs lie along its upper sides while other fingers wrap around the ends; the hands lift and lower it together, then open again. This looks like a two-handed side hold. Its security outside the simulation is unmeasured.",
             "definition": "High carry: object origin >5 cm above its replay starting height. Both: raw object contact with at least one body of each hand.",
             "inputs": {p: MANIFEST["inputs"][p]["sha256"] for p in used}}
    if pan:
        facts["visual_description"] = "The left hand props up the broad scoop edge while the right supports the circular handle end. Both palms stay quite open, with several fingers stretched out or sticking away from the pan. This looks like an unusual two-handed tray carry; a clear closed handle grip is not visible. The simplified model's circular handle is filled in. The motion looks coordinated, but the hold is not evidence of human-like dexterity."
    facts["raw_force_sum_N"] = {}
    for side in ("left", "right", "support"):
        force = np.array([sum(np.linalg.norm(p["force_N"]) for p in row["pairs"]
                              if side in p["other_body"]) for row in contacts])
        facts["raw_force_sum_N"][side] = {"peak_high_carry": float(force[high_carry].max()),
                                         "high_carry_frames_above_0_1N": int((force[high_carry] > .1).sum())}
    (ROOT / base / "contact-facts.json").write_text(json.dumps(facts, indent=2)+"\n")
    # Camera coordinates: x right, y up, z toward the viewer. Fixed over the clip.
    eye = np.array([1., -1.6, .9]); eye /= np.linalg.norm(eye)
    right = np.cross([0., 0., 1.], eye); right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    camera = np.stack([right, up, eye])
    object_id = int(object_ids[0])
    local_object = np.concatenate([v for info, v, _ in meshes if info["body"] == object_id])
    corners, oriented_extents = mesh_box_corners(local_object)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if (i ^ j) in (1, 2, 4)]
    if pan:
        # Goal contours use the same actual pan mesh, including its handle.
        goal_mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=v, faces=f, process=False)
            for info, v, f in meshes if info["body"] == object_id])
        goal_vertices = np.asarray(goal_mesh.vertices)
        adjacency = goal_mesh.face_adjacency
        feature = goal_mesh.face_adjacency_angles > np.deg2rad(25)
        boundary = goal_mesh.edges_unique[np.bincount(goal_mesh.edges_unique_inverse,
                                                     minlength=len(goal_mesh.edges_unique)) == 1]

    def world_mesh(frame):
        return [(info, transform_local_points(v, poses[frame, info["body"]]), f)
                for info, v, f in meshes]

    moments = np.linspace(0, len(poses)-1, 6).round().astype(int).tolist() if pan else [0, 50, 100, 190, 290, len(poses) - 1]
    projected = np.concatenate([v @ camera.T for t in moments for _, v, _ in world_mesh(t)])
    lower, upper = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    center = (lower + upper) / 2
    scale = min(850 / (upper[0]-lower[0]), 470 / (upper[1]-lower[1]))
    comparison = summary["official_comparison"]
    tracking_angles = orientation_error_degrees(poses[:, object_id, 3:], reference[:, 0, 3:])

    def image(frame):
        canvas = np.full((720, 960, 3), 250, np.uint8)

        def project(points):
            p = points @ camera.T
            return np.column_stack([480 + (p[:, 0]-center[0])*scale,
                                    360 - (p[:, 1]-center[1])*scale]).astype(np.int32), p[:, 2]

        polygons, depths, colours = [], [], []
        light = np.array([.3, -.4, 1.]); light /= np.linalg.norm(light)
        for info, vertices, faces in world_mesh(frame):
            points, depth = project(vertices)
            triangles = vertices[faces]
            normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)
            shade = .55 + .45 * np.abs(normal @ light)
            if info["object"]:
                colour = np.array([227, 163, 62])
            elif "left_" in info["label"]:
                colour = np.array([64, 150, 146])
            else:
                colour = np.array([151, 110, 178])
            polygons.extend(points[faces]); depths.extend(depth[faces].mean(axis=1))
            colours.extend((shade[:, None] * colour).astype(np.uint8))
        # Painter ordering gives an opaque orthographic mesh illustration.
        for i in np.argsort(depths):
            cv2.fillConvexPoly(canvas, polygons[i], tuple(int(c) for c in colours[i]), cv2.LINE_AA)
        if pan:
            goal = transform_local_points(goal_vertices, reference[frame, 0])
            normal = Rotation.from_quat(reference[frame, 0, 3:]).apply(goal_mesh.face_normals)
            facing = normal @ eye >= 0
            silhouette = facing[adjacency[:, 0]] != facing[adjacency[:, 1]]
            draw_edges = np.vstack([goal_mesh.face_adjacency_edges[feature | silhouette], boundary])
        else:
            goal = transform_local_points(corners, reference[frame, 0])
            draw_edges = edges
        outline, _ = project(goal)
        for a, b in draw_edges:
            length = np.linalg.norm(outline[b]-outline[a])
            for offset in np.arange(0, length, 12):
                begin = outline[a] + (outline[b]-outline[a]) * offset / max(length, 1)
                end = outline[a] + (outline[b]-outline[a]) * min(offset+6, length) / max(length, 1)
                cv2.line(canvas, tuple(begin.astype(int)), tuple(end.astype(int)), (42, 120, 214), 2, cv2.LINE_AA)
        title = "Mocap-39 pan policy, CPU replay" if pan else "Same policy, CPU replay"
        cv2.putText(canvas, f"{title} | {times[frame]:.2f} s", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (32, 40, 51), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Teal: left hand   Purple: right hand   Amber: object   Dashed blue: goal", (28, 75), cv2.FONT_HERSHEY_SIMPLEX, .53, (32, 40, 51), 1, cv2.LINE_AA)
        count = contacts[frame]["object_contacts"]
        cv2.putText(canvas, f"Object contacts: {count} | Goal orientation error: {tracking_angles[frame]:.2f} deg", (28, 641), cv2.FONT_HERSHEY_SIMPLEX, .65, (32, 40, 51), 1, cv2.LINE_AA)
        cv2.line(canvas, (770, 624), (770+int(.05*scale), 624), (32, 40, 51), 3)
        cv2.putText(canvas, "5 cm", (774, 614), cv2.FONT_HERSHEY_SIMPLEX, .5, (32, 40, 51), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"CPU / official path difference: mean {comparison['mean_m']*100:.3f} cm, max {comparison['max_m']*100:.3f} cm", (28, 674), cv2.FONT_HERSHEY_SIMPLEX, .56, (32, 40, 51), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Simplified Sharpa meshes | Diagnostic only; never scored or packed", (28, 703), cv2.FONT_HERSHEY_SIMPLEX, .55, (32, 40, 51), 1, cv2.LINE_AA)
        return canvas

    # Keep each panel's aspect ratio, including its readable annotations.
    sheet = Image.new("RGB", (2880, 1440), "white")
    for n, frame in enumerate(moments):
        sheet.paste(Image.fromarray(image(frame)), ((n % 3)*960, (n // 3)*720))
    sheet.save(FIGURES / f"{figure_stem}.png")
    caption = ("The policy's actual links and the box in a CPU replay of the finished checkpoint. "
               "Simplified installed Sharpa visual meshes; dashed blue goal follows the object's mesh-oriented bounds. "
               "Diagnostic only, never scored or packed; CPU physics can differ from the official export.")
    if pan:
        caption = ("Practice episode 39, own pan reconstruction, mocap goal and actual policy hands in a CPU diagnostic. "
                   "Not a challenge result; scored episodes have no motion capture. A video-trained practice policy is now recorded; its result is shown separately below.")
    register(FIGURES / f"{figure_stem}.png", used, caption, source_frames=moments)
    video = FIGURES / f"{figure_stem}.mp4"
    stride = max(1, round(summary["control_fps"] / 10))
    fps = summary["control_fps"] / stride
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 720))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    # Preserve the source clock over the whole reference, including its last step.
    sampled = list(range(0, len(poses), stride))
    if sampled[-1] != len(poses)-1:
        sampled.append(len(poses)-1)
    for frame in sampled:
        writer.write(cv2.cvtColor(image(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    if video.stat().st_size > 5_000_000:
        raise ValueError("Policy hand video exceeds 5 MB")
    capture = cv2.VideoCapture(str(video))
    decoded, previews = 0, []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded in (0, len(sampled)//6, len(sampled)//2, len(sampled)-1):
            previews.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        decoded += 1
    capture.release()
    if decoded != len(sampled):
        raise ValueError("Policy hand MP4 is incomplete")
    review = Image.new("RGB", (1920, 1440), "white")
    for i, preview in enumerate(previews):
        review.paste(preview, ((i % 2)*960, (i // 2)*720))
    review_path = OUTPUT / ("review-pan-policy-hands-video.png" if pan else "review-policy-hands-video.png")
    review.save(review_path)
    register(video, used, caption, frames=decoded, fps=fps, bytes=video.stat().st_size)
    return {"official_comparison": comparison, "frames_decoded": decoded,
            "video_bytes": video.stat().st_size, "render_geometry": summary["geometry"],
            "goal_bound_geometry": "Same body-local pan mesh, silhouette/feature edges and shared xyzw pose transform" if pan else "Oriented bounds of the same body-local visual mesh; shared xyzw pose transform",
            "drawn_oriented_mesh_dimensions_cm": (oriented_extents * 100).tolist()}


def pan_mask_visuals() -> dict:
    """Overlay saved binary masks on their exact cached real camera frames."""
    used, selections = [], {}
    colours = np.array([[235, 158, 35], [28, 169, 175], [193, 83, 160]], dtype=np.uint8)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.8))
    fig.subplots_adjust(left=.015, right=.99, top=.87, bottom=.12, hspace=.20, wspace=.035)
    fig.suptitle("The vision step marks pixels on real footage.", x=.025, ha="left", y=.98, fontsize=23)
    fig.text(.025, .92, "Amber: pan   Teal / pink: hands   |   Pixel masks, not 3D positions or measured accuracy", fontsize=12)

    def overlay(base, frame):
        raw = np.asarray(Image.open(io.BytesIO(read_bytes(f"{base}/input_frames/{frame:06d}.jpg", used))).convert("RGB"))
        with np.load(io.BytesIO(read_bytes(f"{base}/frames/frame_{frame:06d}.npz", used))) as data:
            masks, time_s = data["masks"], float(data["timestamp_seconds"])
        if raw.shape[:2] != masks.shape[1:]:
            raise ValueError("Mask and real camera frame grids differ")
        result = raw.copy()
        for mask, colour in zip(masks, colours, strict=True):
            result[mask] = (.62*result[mask]+.38*colour).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, tuple(int(c) for c in colour), 2)
        return result, time_s

    for row, (episode, folder, moments) in enumerate([
            (39, "masks", [0, 100, 200, 318]),
            (38, "masks-smallest", [0, 80, 170, 262])]):
        base = f"data/outputs/moving_reference/ep{episode}/{folder}"
        read_json(f"{base}/summary.json", used)
        selections[str(episode)] = moments
        for ax, frame in zip(axes[row], moments, strict=True):
            pixels, time_s = overlay(base, frame)
            ax.imshow(pixels); ax.axis("off")
            ax.set_title(f"Episode {episode} | {time_s:.2f} s", fontsize=11, pad=4)
    bad, time_s = overlay("data/outputs/moving_reference/ep38/masks", 0)
    axes[2, 0].imshow(bad); axes[2, 0].axis("off")
    axes[2, 0].set_title("38: first try picked the table", fontsize=11, pad=4)
    for ax in axes[2, 1:]:
        ax.axis("off")
    axes[2, 1].text(0, .8, "A fixed rule picks the smallest\nqualifying detection box.\nThe same rule applies to both clips.", fontsize=12, va="top")
    axes[2, 2].text(0, .8, "38 is a scored episode.\nIts public footage is used here;\nits labels are unavailable\nand were not used.", fontsize=12, va="top")
    axes[2, 3].text(0, .8, "Limits: occluded handle edges;\n38's late filled handle opening;\n39 at 10 s marks a bystander's\nhand. No accuracy score.", fontsize=12, va="top")
    caption = ("The vision step sees marked pixels, not yet positions in the world. "
               "Episode 38's first try picked the table; a fixed rule corrected selection on both clips. "
               "38 is a scored episode: public challenge footage is used here, but its labels are unavailable and were not used.")
    footnote(fig, caption, 166)
    save(fig, "13-pan-masks.png", used, caption)
    return {"source_frames": selections, "failure_example": {"episode": 38, "frame": 0},
            "labels_used": False, "new_inference": False}


def tracked_pan_visuals() -> dict:
    """Project the sealed references onto cached rectified camera footage only."""
    from src.features.moving_reference._object import render

    used, clips = [], {}
    read_bytes("docs/chain/moving-m1-sealed.md", used)
    read_bytes("src/features/moving_reference/_object.py", used)
    read_bytes("src/features/moving_reference/camera.py", used)
    expected = {39: "44cd03d9da3a9b017610eab00439662eb8d7bc83ff00bd0a386cff618d30b3e1",
                38: "c5da9fbe4802a62ccfeca8627f2b9aa69a2acf14eaeecbc565b3a728fb275f12"}
    asset = "data/outputs/moving_reference/ep39/candidate-c/asset/mini_dust_pan.obj"
    mesh = trimesh.load(io.BytesIO(read_bytes(asset, used)), file_type="obj", force="mesh", process=False)
    for episode in (39, 38):
        base = f"data/outputs/moving_reference/ep{episode}/candidate-c"
        reference_bytes = read_bytes(f"{base}/reference.npz", used)
        if hashlib.sha256(reference_bytes).hexdigest() != expected[episode]:
            raise ValueError("The sealed reference changed")
        with np.load(io.BytesIO(reference_bytes)) as data:
            reference, times = data["world_T_object"], data["time"]
        with np.load(io.BytesIO(read_bytes(f"{base}/tracking.npz", used))) as data:
            cameras, intrinsics = data["world_T_camera"], data["intrinsics"]
            recorded_asset = data["world_T_asset"]
        convention = read_json(f"{base}/body-convention.json", used)
        world_asset = reference @ np.asarray(convention["scan_T_asset"])
        equivalence = float(np.abs(world_asset-recorded_asset).max())
        if equivalence > 1e-9:
            raise ValueError("Reference/body constant does not reconstruct the recorded asset poses")
        clips[episode] = dict(base=base, times=times, cameras=cameras, intrinsics=intrinsics,
                              poses=world_asset, cache={}, equivalence_m=equivalence)

    def overlay(episode, index):
        clip = clips[episode]
        if index in clip["cache"]:
            return clip["cache"][index]
        path = f"{clip['base']}/frames/frame_{index:06d}.npz"
        with np.load(io.BytesIO(read_bytes(path, used))) as data:
            gray = data["gray"]
        camera_asset = np.linalg.inv(clip["cameras"][index]) @ clip["poses"][index]
        predicted = render(mesh.vertices, mesh.faces, camera_asset, clip["intrinsics"], gray.shape)
        contours, _ = cv2.findContours(predicted.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pixels = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(pixels, contours, -1, (30, 215, 75), 2, cv2.LINE_AA)
        clip["cache"][index] = pixels
        return pixels

    moments = {39: [0, 87, 202, 318], 38: [0, 71, 167, 262]}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    fig.subplots_adjust(left=.012, right=.99, bottom=.16, top=.84, hspace=.17, wspace=.025)
    fig.suptitle("A tracked pan outline on the real footage.", x=.025, ha="left", y=.98, fontsize=23)
    fig.text(.025, .91, "Green: sealed 3D reference projected through the fitted camera • rectified real camera frames", fontsize=12)
    for row, episode in enumerate((39, 38)):
        for ax, index in zip(axes[row], moments[episode], strict=True):
            ax.imshow(overlay(episode, index)); ax.axis("off")
            ax.set_title(f"Episode {episode} | {clips[episode]['times'][index]:.2f} s", fontsize=11, pad=4)
    caption = ("Green is the projected reference mesh, not the input mask or a policy rollout. "
               "The full outline ignores hand occlusion. 39 is a practice episode; 38 has no labels available to us and its world-path accuracy is unknown.")
    footnote(fig, caption, 157)
    # Register after the video has read every sampled source, retaining one source map.
    fig.savefig(FIGURES / "16-tracked-pan.png", dpi=180, facecolor="white"); plt.close(fig)
    path = FIGURES / "16-tracked-pan.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10., (960, 420))
    if not writer.isOpened():
        raise RuntimeError("Tracked-pan MP4 encoder unavailable")
    timeline = np.arange(0, max(c["times"][-1] for c in clips.values())+.0001, .1)
    video_sources = {39: [], 38: []}
    for time_s in timeline:
        canvas = np.full((420, 960, 3), 250, np.uint8)
        for col, episode in enumerate((39, 38)):
            times = clips[episode]["times"]
            index = min(len(times)-1, max(0, int(np.searchsorted(times, time_s+1e-6, side="right")-1)))
            video_sources[episode].append(index)
            canvas[40:340, col*480:(col+1)*480] = cv2.resize(overlay(episode, index), (480, 300))
            label = f"Episode {episode} | {times[index]:.2f} s"
            if time_s > times[-1]+.051:
                label += " | ended: holding last frame"
            cv2.putText(canvas, label, (col*480+10, 26), cv2.FONT_HERSHEY_SIMPLEX, .50, (32,40,51), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Green: saved reference projected onto rectified real footage; full outline ignores hands.", (12, 365), cv2.FONT_HERSHEY_SIMPLEX, .52, (32,40,51), 1, cv2.LINE_AA)
        cv2.putText(canvas, "39: practice | 38: no labels, accuracy unknown | Path only; policy shown separately.", (12, 395), cv2.FONT_HERSHEY_SIMPLEX, .52, (32,40,51), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    if path.stat().st_size > 5_000_000:
        raise ValueError("Tracked-pan video exceeds 5 MB")
    capture = cv2.VideoCapture(str(path)); decoded, previews = 0, []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if decoded in (0, len(timeline)//3, 2*len(timeline)//3, len(timeline)-1):
            previews.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        decoded += 1
    capture.release()
    if decoded != len(timeline):
        raise ValueError("Incomplete tracked-pan video")
    sheet = Image.new("RGB", (1920, 840), "white")
    for i, preview in enumerate(previews):
        sheet.paste(preview, ((i%2)*960, (i//2)*420))
    sheet.save(OUTPUT / "review-tracked-pan-video.png")
    register(FIGURES / "16-tracked-pan.png", used, caption, source_steps=moments)
    register(path, used, caption, frames=decoded, fps=10, bytes=path.stat().st_size,
             sampled_source_steps=video_sources, shorter_clip_end="Hold labelled final frame after episode 38 ends")
    return {"frames_decoded": decoded, "video_bytes": path.stat().st_size,
            "reference_hashes": expected, "reference_to_recorded_asset_max_difference": {
                str(ep): clip["equivalence_m"] for ep, clip in clips.items()},
            "world_accuracy_38": "Unknown; no labels", "new_tracking_or_scoring": False}


def video_path_comparison() -> dict:
    """Visualize the existing sealed comparison; never run the sealed pass again."""
    used = []
    base = "data/outputs/moving_reference/ep39"
    sealed = read_json(f"{base}/sealed/sealed.json", used)
    read_bytes("docs/chain/moving-m1-sealed.md", used)
    read_bytes(f"{base}/results.md", used)  # Candidate A's report is stored at episode root.
    read_bytes(f"{base}/candidate-c/results-amended.md", used)
    with np.load(io.BytesIO(read_bytes(f"{base}/sealed/errors.npz", used))) as data:
        video = data["aligned_position"]
        position_error, angle_error = data["position_error_m"]*100, data["rotation_error_deg"]
    with np.load(io.BytesIO(read_bytes(f"{base}/candidate-c/reference.npz", used))) as data:
        times = data["time"]
    # The completed mocap-policy export already caches 39's goal, within <0.3 um
    # of the mocap poses used in the sealed pass. No dataset labels or scan read.
    source = "data/outputs/moving_policy/ep39-a1/score-66763776/trajectory.parquet"
    frame = pd.read_parquet(io.BytesIO(read_bytes(source, used))).sort_values("step")
    mocap = positions(frame, "reference")
    m2 = read_json("data/outputs/moving_policy/ep39-a1/score-66763776/mocap_metrics.json", used)
    if len(video) != 319 or len(mocap) != 319 or not np.allclose(np.linalg.norm(video-mocap, axis=1), position_error/100, atol=1e-6, rtol=0):
        raise ValueError("Cached mocap goal differs from the existing sealed comparison")
    fig = plt.figure(figsize=(15, 8.6))
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, .85), left=.065, right=.98, top=.82, bottom=.17, hspace=.40, wspace=.32)
    height, top = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1])
    pe, ae, bars = fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1]), fig.add_subplot(grid[:,2])
    green = "#179944"
    fig.suptitle("A video-derived path, compared once with motion capture.", x=.025, ha="left", y=.98, fontsize=23)
    fig.text(.025, .91, "Practice episode 39 • this figure shows the path alone • learned policy: see figure 18", fontsize=13)
    zero = mocap[0]
    for path_values, colour, label in ((mocap, BLUE, "Motion capture"), (video, green, "Video-derived path")):
        height.plot(times, (path_values[:,2]-zero[2])*100, color=colour, label=label)
        top.plot((path_values[:,0]-zero[0])*100, (path_values[:,1]-zero[1])*100, color=colour, label=label)
    height.set(title="Height after first-pose alignment", xlabel="Clip time (s)", ylabel="Height above start (cm)")
    height.legend(fontsize=9)
    top.set(title="Top view after the same alignment", xlabel="World x from start (cm)", ylabel="World y from start (cm)")
    top.set_aspect("equal", adjustable="datalim")
    pe.plot(times, position_error, color=green); pe.axhline(position_error.mean(), color=GREY, ls="--")
    pe.set(title="Position error over time", xlabel="Clip time (s)", ylabel="Error (cm)")
    ae.plot(times, angle_error, color=green); ae.axhline(angle_error.mean(), color=GREY, ls="--")
    ae.set(title="Angle error over time", xlabel="Clip time (s)", ylabel="Error (degrees)")
    values = [.18, sealed["kit"]["AUC"], m2["kit"]["AUC"]]
    bars.bar(range(3), values, color=[GREY,green,AMBER], width=.62)
    bars.set(title="Fit on practice 39\nDifferent stages", ylabel="Fit score (AUC)", ylim=(0,1.1),
             xticks=range(3), xticklabels=["Held\nstill", "Video\npath only", "Policy with\nmocap goal"])
    bars.tick_params(axis="x", labelsize=10)
    for i, value in enumerate(values):
        bars.text(i, value+.025, f"{value:.2f}", ha="center", fontsize=13)
    caption = ("Saved single sealed comparison on practice episode 39: the scorer's one first-pose rigid alignment, with no scale fit or later alignment. "
               "This is a path, not a video-driven policy or challenge result. AUC bars measure different stages; none predicts a video-trained policy's result.")
    footnote(fig, caption, 154)
    save(fig, "17-video-versus-mocap.png", used, caption)
    return {"AUC": sealed["kit"]["AUC"], "mean_position_cm": sealed["mean_position_error_m"]*100,
            "mean_angle_degrees": sealed["mean_rotation_error_deg"], "allowance_AUC": sealed["allowance"]["AUC"],
            "source": "Existing sealed pass only; no new scoring", "bars": values,
            "position_error_saved_mean_difference_cm": float(abs(position_error.mean()-sealed["mean_position_error_m"]*100)),
            "angle_error_saved_mean_difference_degrees": float(abs(angle_error.mean()-sealed["mean_rotation_error_deg"]))}


def video_policy_chart(episode: int = 39) -> dict:
    """Plot saved video-policy results: aligned practice 39 or self-reference 38."""
    used = []
    base = f"data/outputs/moving_policy/ep{episode}-v1"
    read_bytes("docs/chain/moving-m2v-outcome.md", used)
    read_bytes(f"{base}/result.md", used)
    if episode == 38:
        own = read_json(f"{base}/score-66763776/self_metrics.json", used)
        first = read_json(f"{base}/score-66763776/first-pose.json", used)
        reprojection = read_json(f"{base}/score-66763776/reprojection.json", used)
        final = pd.read_parquet(io.BytesIO(read_bytes(f"{base}/score-66763776/trajectory.parquet", used))).sort_values("step")
        midpoint_path = f"{base}/boundary-gpu-32814080/trajectory.parquet"
        runs = [("Scheduled end: 66.76M", final, BLUE, "-")]
        if (ROOT / midpoint_path).is_file():
            midpoint = pd.read_parquet(io.BytesIO(read_bytes(midpoint_path, used))).sort_values("step")
            runs.insert(0, ("Failed midpoint: 32.81M", midpoint, AMBER, "--"))
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.5))
        fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.27, wspace=.30)
        fig.suptitle("38 follows its video goal at the end; the midpoint failed.",
                     x=.025, ha="left", y=.98, fontsize=22)
        fig.text(.025, .90, "Goal from video, not truth; no labels • unaligned saved exports • no challenge score", fontsize=12)
        measured = {}
        for label, frame, colour, line in runs:
            error = np.linalg.norm(positions(frame, "achieved")-positions(frame, "reference"), axis=1)*100
            angle = orientation_error_degrees(
                frame[[f"achieved_quaternion_{a}" for a in "xyzw"]].to_numpy(),
                frame[[f"reference_quaternion_{a}" for a in "xyzw"]].to_numpy())
            measured[label] = {"position_cm": float(error.mean()), "angle_degrees": float(angle.mean())}
            for axis, values, unit in zip(axes, (error, angle), ("cm", "deg")):
                axis.plot(frame.time_s, values, color=colour, ls=line,
                          label=f"{label}: mean {values.mean():.2f} {unit}")
        for axis, value, unit, title in zip(axes, (own["E_S_m"]*100, own["Theta_S_deg"]),
                                           ("cm", "deg"), ("Position difference", "Turning difference")):
            axis.axhline(value, color=GREY, ls=":", label=f"Held still: mean {value:.2f} {unit}")
            axis.set(title=title, xlabel="Clip time (s)", ylabel=f"Error from video goal ({unit})\nNot truth; no labels", ylim=(0, None))
            axis.legend(fontsize=9, loc="upper left")
        final_values = measured["Scheduled end: 66.76M"]
        if abs(final_values["position_cm"]-own["E_m"]*100)>1e-5 or abs(final_values["angle_degrees"]-own["Theta_deg"])>1e-5:
            raise ValueError("Figure 20 does not reproduce the saved final self-reference errors")
        caption = ("Goal from video, not truth; no labels. Curves show error over the clip at two saved checkpoints, not a training curve. "
                   "Held still is a mean over the clip, not its per-frame error. Why the midpoint failed is undiagnosed. "
                   "Following this goal does not show that the goal is right or that 38 beats its still-object export.")
        footnote(fig, caption, 147)
        save(fig, "20-episode38-video-goal.png", used, caption)
        return {"position_cm": own["E_m"]*100, "angle_degrees": own["Theta_deg"],
                "held_still_position_cm": own["E_S_m"]*100, "held_still_angle_degrees": own["Theta_S_deg"],
                "checkpoints": measured, "first_pose_cm": first["position_error_cm"],
                "first_pose_degrees": first["rotation_error_deg"], "reprojection_median": reprojection["median_iou"],
                "reference_end_fraction": own["reference_end_fraction"],
                "completed_without_reset": own["completed_without_reset"], "frames": len(final),
                "label_access": False, "in_entry": EP38_IN_ENTRY}
    aligned = read_json(f"{base}/score-66763776/mocap_metrics_aligned.json", used)
    raw = read_json(f"{base}/score-66763776/mocap_metrics.json", used)
    own = read_json(f"{base}/score-66763776/self_metrics.json", used)
    frame = pd.read_parquet(io.BytesIO(read_bytes(f"{base}/score-66763776/trajectory.parquet", used))).sort_values("step")
    mocap_frame = pd.read_parquet(io.BytesIO(read_bytes("data/outputs/moving_policy/ep39-a1/score-66763776/trajectory.parquet", used))).sort_values("step")
    mocap = positions(mocap_frame, "reference")
    aq = frame[[f"achieved_quaternion_{a}" for a in "xyzw"]].to_numpy()
    mq = mocap_frame[[f"reference_quaternion_{a}" for a in "xyzw"]].to_numpy()
    turn = Rotation.from_quat(mq[0]) * Rotation.from_quat(aq[0]).inv()
    actual = turn.apply(positions(frame,"achieved")-positions(frame,"achieved")[0])+mocap[0]
    error = np.linalg.norm(actual-mocap,axis=1)*100
    theta = np.degrees((turn*Rotation.from_quat(aq)*Rotation.from_quat(mq).inv()).magnitude())
    if abs(error.mean()/100-aligned["E_m"]) > 2e-6 or abs(theta.mean()-aligned["Theta_deg"]) > .001:
        raise ValueError("Presentation alignment does not reproduce the saved corrected metrics")
    sealed = read_json("data/outputs/moving_reference/ep39/sealed/sealed.json", used)
    m2 = read_json("data/outputs/moving_policy/ep39-a1/score-66763776/mocap_metrics.json", used)
    fig=plt.figure(figsize=(14.6,7.5))
    grid=fig.add_gridspec(2,2,width_ratios=(1.05,1.2),left=.065,right=.98,top=.80,bottom=.18,wspace=.26,hspace=.38)
    bars=fig.add_subplot(grid[:,0]);height=fig.add_subplot(grid[0,1]);errors=fig.add_subplot(grid[1,1])
    fig.suptitle("A learned policy now follows a path taken from video.",x=.025,ha="left",y=.98,fontsize=23)
    fig.text(.025,.90,"Practice episode 39 only • different stages, not challenge results • one fresh-start video-policy seed",fontsize=12)
    values=[.18,sealed["kit"]["AUC"],aligned["kit"]["AUC"],m2["kit"]["AUC"]]
    bars.bar(range(4),values,color=[GREY,"#179944",AMBER,BLUE],width=.65)
    bars.set(title="Fit on practice 39: different stages",ylabel="AUC",ylim=(0,1.1),xticks=range(4),
             xticklabels=["Held\nstill","Video\npath alone","Policy on\nvideo path","Policy on\nmocap"])
    bars.tick_params(axis="x",labelsize=10)
    for i,value in enumerate(values):bars.text(i,value+.025,f"{value:.3f}" if i==2 else f"{value:.2f}",ha="center",fontsize=13)
    zero=mocap[0,2]
    height.plot(frame.time_s,(mocap[:,2]-zero)*100,color=BLUE,label="Mocap")
    height.plot(frame.time_s,(actual[:,2]-zero)*100,color=AMBER,ls="--",label="Video-trained policy")
    height.set(title="Height after first-pose alignment",ylabel="Height above start (cm)")
    height.legend(fontsize=9)
    errors.plot(frame.time_s,error,color=AMBER)
    errors.axhline(error.mean(),color=GREY,ls="--",label=f"Mean {error.mean():.2f} cm")
    errors.set(title="Aligned position error over time",ylabel="Error (cm)",xlabel="Clip time (s)")
    errors.legend(fontsize=9)
    caption="The first automatic verdict was NULL because it compared unrelated world frames. First-pose alignment was applied after the fact; the reader should weigh this. No numeric limit changed. No guarantee this beats a still object on 38."
    footnote(fig,caption,153)
    save(fig,"18-policy-on-video-path.png",used,caption)
    return {"AUC":aligned["kit"]["AUC"],"position_cm":aligned["E_m"]*100,"angle_degrees":aligned["Theta_deg"],
            "T":aligned["T"],"Q":aligned["Q"],"self_position_cm":own["E_m"]*100,
            "self_angle_degrees":own["Theta_deg"],"raw_position_cm":raw["E_m"]*100,
            "raw_angle_degrees":raw["Theta_deg"],"raw_verdict":"NULL","corrected_verdict":"GO with alignment",
            "aligned_error_mean_difference_m":float(abs(error.mean()/100-aligned["E_m"])),
            "aligned_angle_mean_difference_degrees":float(abs(theta.mean()-aligned["Theta_deg"])),"bars":values}


def video_policy_footage(episode: int = 39) -> dict:
    """Draw the actual official stand-in and CPU replay hands in the frozen camera."""
    from src.features.moving_reference._object import render
    from src.features.moving_reference.camera import apply, project

    used=[];base=f"data/outputs/moving_policy/ep{episode}-v1";replay_base=f"data/outputs/visuals/policy-hands/ep{episode}-video"
    figure_name = "19-policy-on-real-footage" if episode == 39 else "21-episode38-on-real-footage"
    episode_label = "Practice 39" if episode == 39 else "38 | No labels"
    review_name = "review-policy-on-real-footage.png" if episode == 39 else "review-episode38-on-real-footage.png"
    reference=read_json(f"{base}/reference.json",used)
    asset=read_json(f"{base}/asset.json",used)
    shift=np.asarray(reference["translation_m"])
    frame=pd.read_parquet(io.BytesIO(read_bytes(f"{base}/score-66763776/trajectory.parquet",used))).sort_values("step")
    achieved=np.column_stack([positions(frame,"achieved"),frame[[f"achieved_quaternion_{a}" for a in "xyzw"]].to_numpy()])
    summary=read_json(f"{replay_base}/summary.json",used)
    comparison=summary["official_comparison"]
    with np.load(io.BytesIO(read_bytes(f"{replay_base}/replay.npz",used))) as data:
        robot=data["body_pose_xyzw"]
    with np.load(io.BytesIO(read_bytes(f"{replay_base}/geometry.npz",used))) as data:
        meshes=[(info,data[f"vertices_{i}"],data[f"faces_{i}"]) for i,info in enumerate(summary["meshes"]) if not info["object"]]
    # The simulation OBJ is already in body coordinates; apply no extra M1 body
    # constant and no scale correction. This deliberately shows the larger asset.
    mesh=trimesh.load(io.BytesIO(read_bytes(f"{base}/mini_dust_pan.obj",used)),file_type="obj",force="mesh",process=False)
    tracking=f"data/outputs/moving_reference/ep{episode}/candidate-c"
    with np.load(io.BytesIO(read_bytes(f"{tracking}/tracking.npz",used))) as data:
        cameras,k=data["world_T_camera"],data["intrinsics"]
    with np.load(io.BytesIO(read_bytes(f"{tracking}/reference.npz",used))) as data:
        source=data["pose"];times=data["time"]
    task_reference=positions(frame,"reference")-shift
    relocation_error=float(np.abs(task_reference-source[:,:3]).max())
    if relocation_error>1e-6:raise ValueError("Recorded task translation does not recover the frozen video path")
    moments=([0,64,127,191,254,318] if episode == 39 else np.rint(np.linspace(0,len(frame)-1,6)).astype(int).tolist());cache={};colours={"left":np.array([28,175,180]),"right":np.array([181,94,209])}

    def pair(index):
        if index in cache:return cache[index]
        with np.load(io.BytesIO(read_bytes(f"{tracking}/frames/frame_{index:06d}.npz",used))) as data:
            gray,pan_mask=data["gray"],data["object_mask"]
        image=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);mask_view=image.copy();policy_view=image.copy()
        mask_view[pan_mask]=(.7*mask_view[pan_mask]+.3*np.array([40,210,70])).astype(np.uint8)
        contours,_=cv2.findContours(pan_mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_view,contours,-1,(30,215,75),2,cv2.LINE_AA)
        camera_world=np.linalg.inv(cameras[index])
        for side,colour in colours.items():
            hand_mask=np.zeros(gray.shape,np.uint8)
            for info,vertices,_ in meshes:
                if f"{side}_" not in info["label"]:continue
                world=transform_local_points(vertices,robot[index,info["body"]])-shift
                points=apply(camera_world,world)
                visible=points[:,2]>.05
                if visible.sum()<3:continue
                pixels=np.rint(np.clip(project(points[visible],k),-10000,10000)).astype(np.int32)
                # Per-link projected visual-mesh hulls keep this diagnostic light
                # while retaining the policy's actual articulated link transforms.
                cv2.fillConvexPoly(hand_mask,cv2.convexHull(pixels),1)
            hit=hand_mask.astype(bool)
            policy_view[hit]=(.70*policy_view[hit]+.30*colour).astype(np.uint8)
            hand_contours,_=cv2.findContours(hand_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(policy_view,hand_contours,-1,tuple(map(int,colour)),2,cv2.LINE_AA)
        body_pose=np.eye(4);body_pose[:3,:3]=Rotation.from_quat(achieved[index,3:]).as_matrix()
        body_pose[:3,3]=achieved[index,:3]-shift
        pan=render(mesh.vertices,mesh.faces,camera_world@body_pose,k,gray.shape)
        contours,_=cv2.findContours(pan.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(policy_view,contours,-1,(235,135,20),3,cv2.LINE_AA)
        view=np.full((420,960,3),250,np.uint8)
        view[40:340,:480]=cv2.resize(mask_view,(480,300));view[40:340,480:]=cv2.resize(policy_view,(480,300))
        cv2.putText(view,f"{episode_label} | {times[index]:.2f} s | Pan mask",(10,26),cv2.FONT_HERSHEY_SIMPLEX,.53,(32,40,51),1,cv2.LINE_AA)
        cv2.putText(view,"Exported pan + CPU robot hands",(490,26),cv2.FONT_HERSHEY_SIMPLEX,.53,(32,40,51),1,cv2.LINE_AA)
        cv2.putText(view,f"Replay / official: mean {comparison['mean_m']*100:.3f} cm, max {comparison['max_m']*100:.3f} cm",(12,366),cv2.FONT_HERSHEY_SIMPLEX,.55,(32,40,51),1,cv2.LINE_AA)
        cv2.putText(view,"Orange: larger simulation pan | Teal / purple: robot hands, not the person's",(12,396),cv2.FONT_HERSHEY_SIMPLEX,.53,(32,40,51),1,cv2.LINE_AA)
        cache[index]=view
        return view

    sheet=Image.new("RGB",(1920,1260),"white")
    for n,index in enumerate(moments):sheet.paste(Image.fromarray(pair(index)),((n%2)*960,(n//2)*420))
    sheet.save(FIGURES/f"{figure_name}.png")
    video=FIGURES/f"{figure_name}.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),10.,(960,420))
    if not writer.isOpened():raise RuntimeError("Policy-on-footage MP4 encoder unavailable")
    sampled=list(range(0,len(frame),2))
    for index in sampled:writer.write(cv2.cvtColor(pair(index),cv2.COLOR_RGB2BGR))
    writer.release()
    if video.stat().st_size>5_000_000:raise ValueError("Policy-on-footage video exceeds 5 MB")
    capture=cv2.VideoCapture(str(video));decoded=0;previews=[]
    while True:
        ok,bgr=capture.read()
        if not ok:break
        if decoded in (0,40,80,len(sampled)-1):previews.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
        decoded+=1
    capture.release()
    if decoded!=len(sampled):raise ValueError("Incomplete policy-on-footage video")
    review=Image.new("RGB",(1920,840),"white")
    for i,picture in enumerate(previews):review.paste(picture,((i%2)*960,(i//2)*420))
    review.save(OUTPUT/review_name)
    caption=(f"CPU replay differs from official by {comparison['mean_m']*100:.3f} cm mean and {comparison['max_m']*100:.3f} cm maximum. "
             "Official achieved pan poses and CPU replay robot hands projected through the frozen camera after undoing the recorded task translation. "
             "The simulated hands are not the person's, and the larger stand-in is drawn as simulated, without shrinking or a new camera fit.")
    if episode == 38:
        caption += (" No labels. Floating robot hands are not the person's grip; foreground occlusion is ignored. "
                    "Image agreement is blind to camera drift. The 0.778 overlap used the smaller tracker mesh, not this larger stand-in.")
    register(FIGURES/f"{figure_name}.png",used,caption,source_steps=moments)
    register(video,used,caption,frames=decoded,fps=10,bytes=video.stat().st_size)
    return {"official_comparison":comparison,"translation_undone_m":shift.tolist(),
            "reference_translation_equivalence_m":relocation_error,"frames_decoded":decoded,"video_bytes":video.stat().st_size,
            "simulation_dimensions_m":asset["identity"]["dimensions_m"],
            "hands":"CPU replay Sharpa visual meshes, projected per-link hulls; no human hand fitting",
            "pan":"Official achieved poses, actual larger simulated mesh; no rescaling or first-pose mocap alignment in image view"}


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
    comparison = metrics["policy_hands"]["official_comparison"]
    contact_facts = read_json("data/outputs/visuals/policy-hands/m0c/contact-facts.json", used)
    goal_audit = read_json("data/outputs/visuals/goal-box-angle-audit.json", used)
    pan_facts = read_json("data/outputs/visuals/policy-hands/ep39/contact-facts.json", used)
    pan_metrics = metrics["pan_learning"]
    pan_comparison = metrics["pan_policy_hands"]["official_comparison"]
    read_bytes("docs/chain/moving-m2-outcome.md", used)
    read_bytes("data/outputs/moving_policy/ep39-a1/result.md", used)
    video_metrics = metrics["video_path_comparison"]
    m2v = metrics["video_policy"]
    m2v_view = metrics["video_policy_footage"]
    m2v_replay = m2v_view["official_comparison"]
    ep38 = metrics["episode38_video_policy"]
    ep38_replay = metrics["episode38_video_policy_footage"]["official_comparison"]
    entry_sentence = ("It is in the moving v5 file, packed after the owner moved the 12:00 UTC cut-off and after it passed its frozen admission gates; the submission ledger records whether that file was uploaded and marked final." if EP38_IN_ENTRY else
                      "It is not in the entry unless the owner moves the 12:00 UTC cut-off; the entry still uses 38's still-object export.")
    read_bytes("docs/chain/SUBMISSION.md", used)
    read_bytes("docs/chain/moving-m2v-outcome.md", used)
    read_bytes("data/outputs/moving_policy/ep39-v1/result.md", used)
    read_bytes("docs/chain/moving-m1-sealed.md", used)
    read_bytes("data/outputs/moving_reference/ep39/results.md", used)
    read_bytes("data/outputs/moving_reference/ep39/candidate-c/results-amended.md", used)
    text = f"""# Seeing what the models do

Recorded state: 4 November 2026, including the shape leaderboard result, the rejected pose file, and episode 38's completed video-goal policy.

## What the challenge asks

Watch a person's camera footage, reconstruct the objects, and reproduce their movement in simulation. A **policy** is a controller trained by trial and error in simulation. A **reference** is the goal path it tries to follow. **AUC** is a 0-to-1 fit score: higher means closer agreement across a range of allowed errors; it is not a percentage of tasks completed.

**A first real challenge score exists for the object shapes; pose scores are {POSE_SCORES}.** The moving policies below are separate experiments. Practice 39's first automatic verdict was NULL and was corrected after the fact for a coordinate-frame mistake; that matters when weighing the result. The moving tissue box is the example supplied with the tools. Episode 38 has no labels, so its completed run measures following a video goal, not challenge accuracy.

### What the leaderboard has actually recorded

The submitted object shapes scored **CD-O 0.53562 cm** on the leaderboard, a measure of shape difference where lower is better. The pose file was rejected on **all five pose boards** because the organisers replaced the submission kit on **2 November** and our copy was from **30 September**. Pose scores: **{POSE_SCORES}**. The new pose metrics register each submitted shape to the hidden scan to set the object's coordinate frame, so shapes affect pose scores too. **Every pose number in this report was measured with the old scorer.** A rough re-check with the new scorer's own code, on our labelled practice clips with public shapes, gives a sense of the change, though it is not a prediction: the still poses move from 0.277 to about 0.29 to 0.30 on the training clips, and the video-trained policy on 39 falls from **0.775 to 0.669** with its simulated shape, or 0.591 with the tracker's shape. Our box-like shapes can also register to the hidden scan the wrong way round, by up to about 180 degrees. **Why the still-pose number is so low:** the real objects move during each clip and still poses never follow them. Even the true starting poses held still reach only about 0.53 on the training clips with the old scorer, so about half of the metric is lost to motion we do not reproduce, and the rest to placement error. Episode 38's moving policy is the first step at changing that, on one of twenty episodes. A repaired pose file with unchanged poses (combined v5) and a second file with 38's moving rollout (moving v5) have been built and checked; the uploads were held for the owner's approval. The shape score does not establish how well the poses work. This report records the finding without using the leaderboard to tune a model.

## What we have prepared to submit so far

Two files are prepared. Combined v5 uses measured simple shapes (the two pretrained shapes for pot 5 and pitcher 20 are not in it, because they have no recorded mapping to the simulation's frame) and object poses from short simulated policy runs in which the objects remain at their starting positions. On challenge training episodes, its still poses score an AUC of **0.277** with the old scorer (about 0.30 with the new one). Even the best possible starting poses, held still, score only **0.532**: objects move in every clip and still poses never follow them. Moving v5 is the same file with one change, episode 38's learned moving policy and its simulated pan shape, and is the file to upload and mark final. That ceiling shows why reproducing motion matters; it does not forecast what a moving policy will score.

![Our entry and the static ceiling](figures/04-entry.png)

The comparison uses challenge training episodes only; the example box's score is deliberately kept separate.

## What changed today

Two completed policies now move the supplied tissue box along its goal path. One resumed earlier training; the other started from nothing, with a different random seed. Their example-task fit scores are **0.981 and 0.979**, and mean position errors are **0.562 and 0.516 cm**. The earlier partial policy averaged **6.803 cm**. These are simulator exports with evaluation assistance removed.

The example is slow, mostly vertical, and comes with its own captured hands and mesh. It reaches only about 9 cm/s and rotates 5.4 degrees. **Its roughly 0.98 score is not a challenge score.** Several training settings changed together, so the records do not isolate which change caused the improvement.

The pan result below is the next step: our first moving result on a challenge-type clip, on practice episode 39. It uses a motion-capture goal, which scored episodes do not provide. It is not a challenge result.

## What the pictures show

![The earlier and finished box trajectories](figures/01-box-height.png)

The goal and achieved heights share the same axes: the earlier policy barely lifts the box, while the finished fresh-start policy follows the lift and return.

![Fit score at recorded checkpoints](figures/02-learning.png)

The fresh run reaches 0.961 in 15.0 minutes and finishes in 49.3; the resumed run finishes in 35.8 minutes after resuming, excluding prior training. Its middle checkpoint's time is bounded by progress logs, not timed exactly. Lines connect measured evaluations without smoothing; early checkpoints trained with assistance, removed during evaluation.

![Position error during the completed rollout](figures/03-position-error.png)

The first two seconds expose the initial lag; mean position error is {metrics['error']['mean_position_cm']:.3f} cm, with a peak of {metrics['error']['maximum_position_cm']:.2f} cm at {metrics['error']['maximum_error_time_s']:.2f} seconds.

![Six moments from the finished policy](figures/09-finished-policy.png)

Blue dotted outlines show the goal; amber solid outlines show what happened, with trails. These are schematic boxes using the shipped mesh's oriented bounds and the exported positions and rotations. Faint wrist-to-fingertip guides in this older figure are **the demonstration's hands, not the policy's**. The separate replay below now shows the actual policy hands.

[Earlier partial policy: the same six moments](figures/08-partial-policy.png) · [Side-by-side video](figures/10-policy-comparison.mp4)

### What the policy's own hands do

![Actual policy hands in a CPU diagnostic replay](figures/12-policy-hands.png)

Teal and purple are the controller's actual left and right hands, drawn with simplified versions of the installed Sharpa hand meshes. Amber is the box; the dashed blue outline is its goal. We reran the finished fresh-start checkpoint on the CPU, with the same explicit reset, one simulated world and assistance switched off. Across all 390 steps, its box path differs from the official export by **{comparison['mean_m']*100:.3f} cm on average and {comparison['max_m']*100:.3f} cm at most**, without aligning the paths. This is a CPU replay of the same policy, not an exact recording of the official run. **It is diagnostic only, never scored or packed.**

The box's angle from its goal averages **{goal_audit['replay']['mean_degrees']:.3f} degrees**, with a maximum of **{goal_audit['replay']['max_degrees']:.3f} degrees**, in this CPU replay. The official export gives **{goal_audit['official']['mean_degrees']:.3f} and {goal_audit['official']['max_degrees']:.3f} degrees**. The blue outline now follows the mesh's own orientation and dimensions. An earlier drawing used a differently oriented, oversized bounding box; the apparent large rotation was a drawing error. Figures 8–10 use the corrected bounds too.

{contact_facts['visual_description']}

The simulator reports contact with both hands in **{contact_facts['both_hands_high_carry_frames']} of {contact_facts['high_carry_frames']} steps** while the box is more than 5 cm above its starting height. These are raw contact points from the last physics substep, not a measure of grip strength or human-like dexterity. The pictures and contacts do not establish a secure, natural grasp.

There are no support contacts in those raised steps. Unverified solver force readings, summed as magnitudes over each hand's contact points, peak around **{contact_facts['raw_force_sum_N']['left']['peak_high_carry']/1000:.1f} kN on the left and {contact_facts['raw_force_sum_N']['right']['peak_high_carry']/1000:.1f} kN on the right**. These large raw values reinforce the limit: following the box path does not demonstrate realistic grip strength.

[Watch the whole CPU replay](figures/12-policy-hands.mp4)

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

The earlier episode-39 setup launched the pan into the air in simulation. A smaller support, and the supplied example's stand-in weight (0.3 kg, not the real pan's), kept it inside the setup check's limits: it moved 2.3 mm and tilted 1.84 degrees against limits of 1 cm and 2 degrees, a narrow margin. **Training has now finished: the policy follows the motion-capture goal in the recorded evaluation.** The result and its limits are below.

### The pan now moves: a practice result with a motion-capture goal

**This is our first moving result on a challenge-type clip, on practice episode 39.** The policy was trained by trial and error in simulation, **from nothing, in {pan_metrics['training_wall_minutes']:.2f} minutes—under an hour**. It uses our own reconstructed pan shape. The goal path comes from motion capture, and the hand arrangement comes from a fixed rule on our shape. Neither the motion goal nor the hand arrangement was recovered from video. The real-video masks elsewhere in this report did not feed this policy.

![Three saved pan policies and the final goal-versus-achieved height](figures/14-pan-learning.png)

| Saved policy | Wall time when saved | Fit score (AUC) | Mean position error | Mean angle error |
| --- | ---: | ---: | ---: | ---: |
| 16 million training steps | {pan_metrics['checkpoint_save_minutes'][0]:.2f} min | 0.168 | 13.74 cm | 119.74° |
| 33 million training steps | {pan_metrics['checkpoint_save_minutes'][1]:.2f} min | 0.951 | 1.03 cm | 1.46° |
| 67 million training steps, finished | {pan_metrics['checkpoint_save_minutes'][2]:.2f} min | 0.982 | 0.62 cm | 0.92° |

Held still on this episode, the fit score is about **0.17**. The finished policy's mean position error is **{pan_metrics['final_position_error_cm']:.3f} cm**, and its mean angle error is **{pan_metrics['final_angle_error_degrees']:.3f} degrees**. It reaches the end of the reference without resets during the rollout. Its recorded motion gain is **{pan_metrics['final_T']:.3f}**, close to the goal's movement scale, and motion agreement is **{pan_metrics['final_Q']:.3f}**; the point-based pose error is **{pan_metrics['final_MPPE_cm']:.2f} cm**. The threshold-based single-object tracking success is 1.0 on this one rollout; it is not success across many tasks. This is one seed on one practice episode. Lines join the three measurements without smoothing; the saved-policy times include simulator startup, and the whole run took {pan_metrics['training_wall_minutes']:.2f} minutes. These checkpoints were fixed in advance, not selected as the best of a search.

The height plot uses the official final export. Its goal and achieved paths are close, with visible lag around the return towards the table at roughly 7–8 seconds. Agreement here does not establish that a noisy path estimated from camera footage will work. The reported AUC uses the kit's recorded measurement against public motion capture and scan surface points; the simulated pan itself contains no scan geometry. This report rebuild reads saved measurements and does not open a scan or rescore the policy.

![The pan and the actual policy hands at six moments](figures/15-pan-policy-hands.png)

Teal and purple show the simulated robot hands actually driven by the policy. Amber is our pan mesh; the dashed blue goal uses that same mesh and pose transform, including the handle. These are a CPU replay of the finished checkpoint. Against the official achieved path, the replay differs by **{pan_comparison['mean_m']*100:.3f} cm on average and {pan_comparison['max_m']*100:.3f} cm at most**, without alignment. The pictures draw what this CPU run actually did; this diagnostic is never scored or packed.

{pan_facts['visual_description']}

Both hands have raw object contacts on **{pan_facts['both_hands_high_carry_frames']} of {pan_facts['high_carry_frames']} steps** with the pan's origin more than 5 cm above its starting height. Support contact force is still reported on **{pan_facts['raw_force_sum_N']['support']['high_carry_frames_above_0_1N']}** of those steps during the moving and tilting sequence. Raw contact-force sums peak at **{pan_facts['raw_force_sum_N']['left']['peak_high_carry']/1000:.1f} and {pan_facts['raw_force_sum_N']['right']['peak_high_carry']/1000:.1f} kN** for the two hands. These are large, unverified simulator readings, not evidence of realistic grip strength or a secure grasp.

[Watch the whole pan-policy CPU replay](figures/15-pan-policy-hands.mp4)

**Not a challenge result; scored episodes have no motion capture. A video-trained practice policy is now recorded; its result is shown separately below.**

## What is still unproven

M2 shows a policy moving our pan on a practice episode with a motion-capture goal. M2v now follows a video-derived goal on the same practice episode. The tracker failed earlier checks and the policy verdict needed a retrospective coordinate-frame correction, so these limits remain central. Neither experiment establishes a natural hand grip or performance on other objects. FoundationPose has only been screened on two used pairs and is not in the entry. No result here guarantees that a moving policy improves the scored entry.

### Real-video masks: what the vision step sees

![Pan and hand masks on public camera footage, including the failed table selection](figures/13-pan-masks.png)

These are real camera frames with the pan marked in amber and hands in teal and pink. This is what the vision step sees: **marked pixels, not yet positions in the world**. Episode 38's first try picked the table; a fixed rule choosing the smallest qualifying detection box corrected the selection and is applied to both clips. Episode 38 is a scored episode. Its footage is public challenge data used here; its labels are not available to us and were not used. Hidden handle edges and a filled handle opening in late frames of 38 remain visible limitations. At 10 seconds in 39, a hand mask marks a bystander's hand. No mask accuracy score, reliable hand identity or 3D tracking result is claimed.

### From marked pixels to a moving pan path

This step estimates where the pan moves and how it turns, while accounting for the camera moving with the person's head. The camera is fitted across the whole clip using background features seen in the stereo pair. Our pan shape is then fitted to its outline and stereo depth in each frame. This produces a **reference path for a policy to try to follow**; it does not make the policy move the pan.

The first tracker failed its quality checks. On 39 it marked only 51 of 319 frames usable, with poor outline agreement and camera coverage; 38 failed too. The later version used pan/hand masks, a whole-clip camera solve, and forward/backward fits to the silhouette and stereo surface. A check also found that the constant relating our shape's axes to the challenge body's axes was wrong: it was corrected using the handle direction, with the same correction on both clips. The constant uses previously derived public scan bounding axes and an origin convention; no scan geometry is drawn here or used as our fitted shape. This report reads that saved constant, not a scan.

**The later tracker still failed the checks written first, then failed a revised set. It went ahead under a conditional rule written before the comparison with motion capture. The reader should weigh this.** Under the first frozen rule it passed 16 of 23 checks on 39 and 17 of 23 on 38. Under the amendment it passed 22 of 28 and 24 of 28. Failures included camera-position consistency, table offset and apparent table tilt; 39 also had too few frames for the table and background-point checks. Going on broke the earlier promise of only one amendment. The conditional rule allowed one sealed comparison on 39 and allowed a different way of measuring camera error only if that comparison passed; table defects still had to be resolved under the same limits. These decisions followed inspection of both clips. This is weaker evidence than a validation whose rules stayed fixed throughout, even though the conditional rule preceded the label comparison.

![The sealed tracked outline on real footage from 39 and 38](figures/16-tracked-pan.png)

Green is our sealed reference projected through the fitted camera, **not the input mask**. These are the straightened camera frames used by the stereo tracker. The outline stays close to the broad pan body at the start and end of both clips, and follows the lift and turn. It is less convincing around the rim and handle during gripping: at 39's 4.35-second sample and 38's 3.55- and 8.35-second samples, parts of the outline sit outside visible edges or run across the hand. The full outline ignores foreground hand occlusion, and the simple extrusion misses some wall detail. Good-looking image overlap alone does not prove a correct position in the world.

[Watch both whole clips with the tracked outline](figures/16-tracked-pan.mp4). The shorter 38 clip ends first; its last frame is held and labelled while 39 finishes.

### One sealed comparison, on practice episode 39

After the references were frozen, the saved sealed pass compared 39's path with its motion capture **once**. We reuse that result here; this rebuild does not run another comparison or scorer. It gives **AUC {video_metrics['AUC']:.3f}, mean position error {video_metrics['mean_position_cm']:.2f} cm and mean angle error {video_metrics['mean_angle_degrees']:.2f} degrees**. The scorer aligns the first pose once, with one rigid transform and no scale fit; it does not keep realigning the path later. The separate arithmetic allowance check gives AUC **{video_metrics['allowance_AUC']:.2f}** after composing the earlier mocap-policy errors onto this path. That is a screen for possible headroom, not a policy rollout or a prediction of success.

![Practice 39: saved video-path comparison with motion capture](figures/17-video-versus-mocap.png)

The lift timing is close, but the top view shows sideways differences and the saved position error rises past 5 cm near 9–10 seconds. The angle error also varies, reaching about 17 degrees. These bars compare held still (**0.18**), the video path by itself (**0.83**) and the earlier policy trained with a mocap goal (**0.98**). They measure different stages. **39 is a practice episode; this figure shows the path alone. The learned video-policy result is shown separately below.** None of these bars is a challenge result.

**38 has no labels available to us. We do not know how good its world path is.** The original frozen and amended checks failed; the later reviewed table check and conditional change are described below. Passing those video checks is not an accuracy measurement or policy adoption. One label comparison on another clip cannot establish 38's accuracy.

**The table checks.** Two of those failures came from a fault in the check itself: its "table" area included the person's lap and the table's edge, and its plane fit was poor. With that fixed and the limits unchanged, too few points were left, so a fresh set of table points, never used to fit the camera, was tracked once. Then both clips stayed inside the limits for table height (about 5 and 3 mm against 10) and sideways drift (about 8 mm on both against 10, close to the limit). Under the rule written before the comparison, a stereo measure dominated by depth noise was swapped for a sideways-only version after 39 passed. So 38 passes the final set of video checks. **That final set was only settled after seeing how the earlier ones fared on both clips.** It is weaker evidence than a test fixed in advance, and it does not measure 38's path against the truth.

### The policy on a video-derived path: M2v

**For the first time here, a learned policy trained by trial and error in simulation moved an object along a motion path taken from video alone, on a practice episode.** This is practice episode 39, one fresh-start seed and 66.76 million training transitions. Training took 61.57 minutes. The motion goal came from the frozen video tracker, not motion capture. The hand targets came from our shape rule; they were not the person's recovered hand motion. The existing body-axis convention and the tracker caveats above still apply.

![Different stages on practice 39, including the video-trained policy](figures/18-policy-on-video-path.png)

Against its own video goal, the policy averages **{m2v['self_position_cm']:.2f} cm position error and {m2v['self_angle_degrees']:.2f} degrees angle error**, with a recorded fit score of 0.936. Against motion capture after the scorer's first-pose alignment, its fit score is **{m2v['AUC']:.3f}**, mean position error **{m2v['position_cm']:.2f} cm** and mean angle error **{m2v['angle_degrees']:.2f} degrees**. Its movement gain is {m2v['T']:.3f}, and motion agreement {m2v['Q']:.3f}. It reaches the reference end with no resets during the rollout. The position limit was 3 cm, so 2.67 cm passes with little margin. Following the path loses about 0.06 AUC from the path alone, and remains below the 0.98 result with a mocap goal. The bars describe different stages on one practice episode, not challenge scores.

**The first automatic verdict read NULL because it compared two different coordinate frames. It was corrected after the fact, and the reader should weigh that.** The original calculation compared the simulator's video-tracker world directly with mocap's world, giving {m2v['raw_position_cm']:.2f} cm and {m2v['raw_angle_degrees']:.2f} degrees. Applying the scorer's single first-pose rigid alignment gives the corrected figures above and the recorded GO decision. The original NULL report is preserved. Numeric limits did not change, but the protocol had not explicitly said to align these arithmetic checks; that interpretation was applied after seeing NULL. The evaluator ran once, while the export and mocap were read again for the correction. Alignment hides initial absolute placement and orientation error; it does not remove later drift, scale or timing error. This report reuses the corrected record and reproduces only the plot arithmetic, without running a scorer.

![Official achieved pan and CPU replay robot hands projected onto real footage, beside the pan mask](figures/19-policy-on-real-footage.png)

**The CPU replay differs from the official achieved path by {m2v_replay['mean_m']*100:.3f} cm on average and {m2v_replay['max_m']*100:.3f} cm at most**, without alignment. The left panel shows the estimated pan mask; the right projects the official achieved pan pose and the replay's robot hands through the frozen camera. The recorded task translation is undone for both. There is no new camera fit, scale adjustment or mocap alignment in this image view.

**The simulated hands are not the person's, and the shape is our larger stand-in, drawn as simulated.** The orange pan visibly overshoots the real pan's rim and handle, with a different apparent tilt at some moments. The robot hands float away from the person's fingers, sometimes extend off the image, and maintain an unusual two-handed support while the person uses a different grip. The stand-in has thick walls and a filled handle; the person's pan does not. This is an honest view of the simulated shape and robot link poses, not a reconstruction of the person's grasp. Hands use translucent projected hulls of the replay's Sharpa link meshes to keep this drawing light; foreground occlusion is not reconstructed.

The saved 0.80 median image-overlap check used the smaller tracker mesh, whereas this image shows the actual larger simulation mesh. That check therefore does not describe the orange contour here. Image agreement through a fitted camera is also blind to camera drift. The picture and scores do not establish realistic grip strength or a natural, secure grasp.

[Watch the video-trained policy on the real footage](figures/19-policy-on-real-footage.mp4)

### Episode 38: the finished run follows its own video goal

**38 has no labels. These measurements say it follows its video goal, not that the goal is right or that it beats the still object on 38.** Its midpoint check failed badly: movement along the goal was only about a quarter of what was asked, with mean position error 25.16 cm and turning error 74.56 degrees. Nobody yet knows why the midpoint failed so badly and the final result did not. The run was stopped twice, once by its own rule and once by the owner, then resumed, losing part of its training memory each time. It is therefore a different experiment from 39's uninterrupted fresh run.

![38: midpoint and final errors against its video goal, not truth; no labels](figures/20-episode38-video-goal.png)

At its scheduled end, after 66.76 million training transitions, it follows its goal at **{ep38['position_cm']:.2f} cm and {ep38['angle_degrees']:.2f} degrees** on average, across all **{ep38['frames']} frames with no resets**. Held still against that same goal gives **{ep38['held_still_position_cm']:.2f} cm and {ep38['held_still_angle_degrees']:.2f} degrees**. The first pose differs by **{ep38['first_pose_cm']:.2f} cm** without alignment. Its median outline overlap is **{ep38['reprojection_median']:.3f}**, measured with the smaller tracker shape against predicted masks; this check cannot see camera drift. **It passes every written check. {entry_sentence}** Those checks were written for the old scorer; any later adoption also needs a shape in the rollout's object frame under the replacement kit.

![38: official achieved pan and CPU replay hands on real footage, beside the pan mask](figures/21-episode38-on-real-footage.png)

The CPU replay differs from the official achieved path by **{ep38_replay['mean_m']*100:.3f} cm on average and {ep38_replay['max_m']*100:.3f} cm at most**, without alignment. As in figure 19, the pan uses the official achieved poses and the hands use the CPU replay, projected through the frozen camera after undoing the saved task translation. There is no new camera fit, scale fit or pose alignment. The larger, thicker stand-in has a filled handle and can overshoot the real pan. The floating robot hands are not the person's fingers; their translucent link hulls ignore foreground occlusion. The 0.778 overlap used the smaller tracker shape, not this larger orange outline. Both the picture and that overlap are blind to camera drift and establish neither a natural grasp nor the correctness of the video goal.

[Watch 38's finished policy on the real footage](figures/21-episode38-on-real-footage.mp4)

---

Rebuild all figures, animations, the table and this report on CPU:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. nice -n 19 data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py
```

Add `--replay` to explicitly rebuild all four CPU diagnostics, `--replay-pan` for M2, `--replay-video-policy` for M2v on 39 or `--replay-video-policy 38` for 38; the default reads saved arrays. See [replay instructions](README.md). Input SHA-256 hashes and per-figure source mappings: `data/outputs/visuals/inputs.json`; numeric checks and build details: `data/outputs/visuals/build-summary.json`. The default reads saved outputs, cached camera frames, installed visual meshes and recorded notes; it does not run a model or read a scan.

Evidence: [box outcomes](../../chain/moving-m0b-outcome.md), [earlier policy](../../chain/moving-m0-outcome.md), [pose pilot](../../chain/foundationpose-pilot-outcome.md), [frame-zero panel](../../chain/frame0-panel-outcome.md), [pan outcome](../../chain/moving-m2-outcome.md), [video-policy outcome and correction](../../chain/moving-m2v-outcome.md), [video-path sealing record](../../chain/moving-m1-sealed.md), [episode-39 protocol](../../chain/moving-m2-protocol.md), [entry facts](../../chain/COMPLETION-PLAN.md#1-where-the-entry-really-stands).
"""
    path = REPORT / "report.md"
    path.write_text(text)
    register(path, list(MANIFEST["inputs"]), "Plain-language report, separate from the two-page abstract.")


def main() -> None:
    """Rebuild presentation; policy inference requires the explicit --replay flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Rebuild the CPU diagnostic before rendering")
    parser.add_argument("--replay-pan", action="store_true", help="Rebuild only the pan CPU diagnostic before rendering")
    parser.add_argument("--replay-video-policy", nargs="?", const=39, type=int, choices=(38, 39), help="Rebuild only this video-policy CPU hand replay (default: 39)")
    args = parser.parse_args()
    for directory in (OUTPUT, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
        directory.resolve().relative_to(ROOT)
    if args.replay:
        replay_output = OUTPUT / "policy-hands/m0c"
        replay_output.mkdir(parents=True, exist_ok=True)
        with (replay_output / "replay.log").open("w") as log:
            subprocess.run([str(ROOT / "data/.runtime/flashchord/venv/bin/python"), "-u", "-B",
                            str(ROOT / "scripts/report/policy_hands.py")], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.replay or args.replay_pan:
        replay_output = OUTPUT / "policy-hands/ep39"
        replay_output.mkdir(parents=True, exist_ok=True)
        base = ROOT / "data/outputs/moving_policy/ep39-a1"
        with (replay_output / "replay.log").open("w") as log:
            subprocess.run([str(ROOT / "data/.runtime/flashchord/venv/bin/python"), "-u", "-B",
                            str(ROOT / "scripts/report/policy_hands.py"),
                            "--checkpoint", str(base / "training/policy_66763776.safetensors"),
                            "--task", str(base / "motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet"),
                            "--output", str(replay_output),
                            "--official", str(base / "score-66763776/trajectory.parquet")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    for episode in ([39, 38] if args.replay else ([args.replay_video_policy] if args.replay_video_policy else [])):
        replay_output = OUTPUT / f"policy-hands/ep{episode}-video"
        replay_output.mkdir(parents=True, exist_ok=True)
        base = ROOT / f"data/outputs/moving_policy/ep{episode}-v1"
        with (replay_output / "replay.log").open("w") as log:
            subprocess.run([str(ROOT / "data/.runtime/flashchord/venv/bin/python"), "-u", "-B",
                            str(ROOT / "scripts/report/policy_hands.py"),
                            "--checkpoint", str(base / "training/policy_66763776.safetensors"),
                            "--task", str(base / f"motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet"),
                            "--output", str(replay_output),
                            "--official", str(base / "score-66763776/trajectory.parquet")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    style()
    goal_audit = goal_box_audit()
    height_chart()
    learning = learning_chart()
    error = error_chart()
    entry_chart()
    table = pose_table()
    camera_sheets()
    print("Charts and camera sheets saved; rendering recorded box motion on CPU.", flush=True)
    policy = policy_visuals()
    harder_task()
    policy_hands = policy_hands_visuals()
    pan_masks = pan_mask_visuals()
    pan_result = pan_learning()
    pan_hands = policy_hands_visuals("data/outputs/visuals/policy-hands/ep39", "15-pan-policy-hands", pan=True)
    tracked_pan = tracked_pan_visuals()
    video_comparison = video_path_comparison()
    m2v = video_policy_chart()
    m2v_footage = video_policy_footage()
    ep38 = video_policy_chart(38)
    ep38_footage = video_policy_footage(38)
    review = review_previews()
    summary = {"learning": learning, "error": error, "policy_visuals": policy,
               "policy_hands": policy_hands, "pan_masks": pan_masks,
               "goal_box_audit": goal_audit,
               "pan_learning": pan_result, "pan_policy_hands": pan_hands,
               "tracked_pan": tracked_pan, "video_path_comparison": video_comparison,
               "video_policy": m2v, "video_policy_footage": m2v_footage,
               "episode38_video_policy": ep38, "episode38_video_policy_footage": ep38_footage,
               "leaderboard": {"shape_CD_O_cm": 0.53562, "pose_scores": POSE_SCORES, "pose_scorer": "old kit throughout this report"},
               "artifact_review": review,
               "missing": ["No pose challenge score or labelled episode-38 policy assessment.",
                           "No labels or world-path accuracy measurement for episode 38.",
                           "No independent panel registration estimate on pair 12.",
                           "Warm-start middle checkpoint has only a logged time bound."],
               "checks": {"CPU_only": os.environ["CUDA_VISIBLE_DEVICES"] == "",
                          "frames_from_recorded_exports": True, "new_model_execution": bool(args.replay or args.replay_pan or args.replay_video_policy)}}
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
