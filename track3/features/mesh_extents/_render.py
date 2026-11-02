"""Candidate-only figures and development-score plots."""

from ._rule_scores import NAMES


def plot(summary: dict):
    """Return a figure of the completed development measurements."""
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(NAMES))
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    series = [(summary.get("baseline_label", "Box v1"), summary["baseline_per_object_cm"])]
    series += [(row["rule"], row["per_object_cm"]) for row in summary["ranking"]]
    for i, (label, values) in enumerate(series):
        ax.bar(x + (i - 1.5) * 0.2, [values[n] for n in NAMES], 0.2, label=label)
    ax.set(xticks=x, xticklabels=[name.replace("_", " ") for name in NAMES],
           xlabel="Object (public scan)", ylabel="CD-O (cm; lower is better)",
           title="Mesh rules: development estimates against public scans")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return fig


def render_candidate(mesh, label: str):
    """Return candidate geometry alone in three views with equal axis scale."""
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    center = mesh.bounds.mean(axis=0)
    radius = mesh.extents.max() * 0.55
    for index, (elevation, azimuth) in enumerate(((25, -60), (90, -90), (0, -90)), 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        collection = Poly3DCollection(mesh.triangles, facecolor="#80b9cd", edgecolor="#305060", linewidth=0.15)
        ax.add_collection3d(collection)
        ax.set(xlim=center[0] + np.array([-radius, radius]), ylim=center[1] + np.array([-radius, radius]),
               zlim=center[2] + np.array([-radius, radius]), xlabel="Long axis (video units)",
               ylabel="Short axis", zlabel="Height")
        ax.set_box_aspect((1, 1, 1))
        from matplotlib.ticker import MaxNLocator

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        ax.set_proj_type("ortho")
        if elevation == 90:
            ax.set_zticks([])
            ax.set_zlabel("")
        elif elevation == 0:
            ax.set_yticks([])
            ax.set_ylabel("")
        ax.set_title(("Perspective", "Top", "Side")[index - 1])
        ax.view_init(elev=elevation, azim=azimuth)
    fig.suptitle(label + " — candidate only")
    return fig


def render_video_comparison(render, video, name: str, crop=None):
    """Return the decoded video crop beside the decoded candidate figure."""
    import matplotlib.pyplot as plt

    if crop:
        video = video.crop(crop)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), width_ratios=(1, 2), constrained_layout=True)
    axes[0].imshow(video)
    axes[0].set_title("Opening video frame (markers ignored)")
    axes[1].imshow(render)
    axes[1].set_title("Frozen candidate geometry")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name.replace("_", " ") + " / video and candidate; views are not registered")
    return fig
