# Procedural synthetic depth

The authorized 18 October 2026 pilot generated **128 samples in 3.330 seconds**.
Its [dataset](../../../outputs/synthetic_depth/20261018T115837.781490Z-06a78723/dataset.json),
[owning run](../../../outputs/synthetic_depth/20261018T115837.781490Z-06a78723/run.json)
and [preview](../../../outputs/synthetic_depth/20261018T115837.781490Z-06a78723/preview.png)
remain on Z:. No synthetic samples entered the completed model campaign, and no
training improvement has been measured.

This slice generates bounded CPU
training pairs using approved NumPy/OpenCV and analytic geometry. It uses no
downloaded assets, learned models, pose labels, licensed resources or simulation
engine. Generation requires its own `runtime.json` `generate_authorized` flag,
the canonical `Z:/hugging-face/video2data` checkout and approved existing Python.

`SyntheticDepthRequest` / `generate_synthetic_depth` defaults to 128 samples and
seed 31; requests allow 1–512 samples. `reference_dataset` names the completed
train-only scratch preparation. Only its metadata supplies the 128×96 camera
grid, pinhole intrinsics and positive depth bounds. Admission rejects every
episode outside the exact eight default training episodes: no episode 41,
tuning 39/42 or holdout 43/44 can enter the reference. No real sample arrays,
teacher outputs or held-out data are read by this generator.

```text
python -m track3.features.synthetic_depth --request track3/features/synthetic_depth/generate.json
```

Each seeded scene has a table plane and one to three table-resting spheres or
oriented boxes. Footprint separation prevents overlaps; placement allows at most
60 attempts. Layouts, sizes, diffuse materials, directional lighting and table
texture vary. Analytic nearest-positive intersections resolve visibility and
occlusion. Rays have z component one, so their intersection parameter is optical
camera-z depth in metres. Labels retain only finite stored float32 depths within
the reference bounds. This is a stylized renderer: no cast shadows, physical
simulation, photorealism or validated sensor-noise model is claimed. Analytic
labels are renderer geometry, not public-dataset ground truth.

Outputs live in `outputs/synthetic_depth/<run-id>/`: `dataset.json`, owning
`run.json`, compressed `samples/synthetic_XXXXXX.npz` and a first-16 RGB montage
`preview.png`. Arrays are `rgb` uint8, `depth_m` float32, `valid` boolean and
`input_intrinsics_pixels`. Invalid depth is NaN. Independent `sample_id` values
and scene/seed records replace episode IDs and PTS; none are fabricated.
`instance_id` is an H×W int16 map from the same nearest intersection: zero is
table/background and 1–3 identify the procedural objects in that sample's scene.
Intersect `valid` with `instance_id == id` for an exact visible-object mask;
`valid & (instance_id == 0)` identifies the observed table. These IDs carry no
competition object identity or real semantic-label claim.

The distinct `synthetic_depth_training_pairs` contract records
`data_origin: procedural_synthetic`, `training_only: true`, analytic targets,
reference camera hash, source hashes, request parameter hash and sample hashes.
Its inherited `video_only` provenance records the permitted camera source and
absence of oracle assets; it does not disguise the explicit synthetic origin.
These artifacts do not satisfy existing public prepared-pairs admission and are
never appended to those datasets automatically. Existing epochs and data remain
untouched. No tests, smoke checks or dependency experiments accompany generation.

A subsequent, separately scoped mixed-training experiment should compare matched
real-only and real-plus-synthetic training recipes, then select using real tuning
episodes. Synthetic-only tuning cannot establish improvement on the videos. The
current frozen final holdout must not be reused to adapt recipes or mixing ratios;
report any already observed holdout outcome without making new choices from it.
