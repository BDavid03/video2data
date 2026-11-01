"""Draw the report's column-sized CD-O figure from saved scores; run from any directory."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
SCORES = ROOT / "outputs/mesh_eval"
SERIES = [
    ("baseline-cube.json", "Cube", "#4477AA"),
    ("analysis-oracle-cube.json", "Scan-PCA box (analysis)", "#CC6677"),
    ("d1-ctx02/v1-box.json", "Video box", "#228833"),
]
LABELS = {
    "beige_cup": "Beige cup", "blue_cup": "Blue cup",
    "mini_dust_pan": "Dust pan", "mini_sweeper": "Sweeper",
    "plastic_dish_rack": "Dish rack", "water_pitcher": "Pitcher",
    "white_pot": "Pot", "white_pot_lid": "Lid",
    "wooden_piece_1": "Wood 1", "wooden_piece_2": "Wood 2",
}


def main() -> None:
    """Read recorded scores, save a vector PDF, and print the derived means for review."""
    data = []
    for filename, _, _ in SERIES:
        record = json.loads((SCORES / filename).read_text())
        data.append({r["object"]: r["cd_o_cm"] for r in record["objects"] if r["status"] == "scored"})
    objects = sorted(data[-1], key=data[-1].get)
    build = json.loads((SCORES / "d1-ctx02/v1-box.json").read_text())["source"]
    pairs = json.loads((ROOT / build).parent.joinpath("objects.json").read_text())["pairs_written"]
    weights = Counter(pair.split(":", 1)[1] for pair in pairs)
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.65), layout="constrained")
    for i, ((_, label, colour), values) in enumerate(zip(SERIES, data)):
        ax.barh([k + (i - 1) * 0.25 for k in range(len(objects))],
                [values[o] for o in objects], height=0.23, label=label, color=colour)
        print(f"{label}: object mean {sum(values[o] for o in objects) / len(objects):.6f} cm; "
              f"pair weighted {sum(values[o] * weights[o] for o in objects) / sum(weights[o] for o in objects):.6f} cm")
    ax.set_yticks(range(len(objects)), [LABELS[o] for o in objects])
    ax.invert_yaxis()
    ax.set_xlabel("Local CD-O (cm; lower is better)")
    ax.set_xlim(0, 7)
    ax.set_xticks([0, 2, 4, 6])
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    target = Path(__file__).with_name("d1-v1-cdo.pdf")
    fig.savefig(target, metadata={"Title": "Development CD-O on public scans", "Author": "", "Creator": ""})
    plt.close(fig)
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
