# Literature and public evidence for structural pose improvements (context 19, 2026-11-03 14:10 UTC)

Asked by the owner ("worth looking at research on the subject"). Third-party numbers below are self-reported and
unverified; they compare methods, not Kaggle scores. Our own train numbers: static T1 AUC 0.2767 (8 episodes);
oracle with true frame-0 poses held static 0.532.

## What the scorer's success names point to

- **SPIDER** (FAIR at Meta / CMU, arXiv 2511.09484): sampling-based, physics-informed retargeting of kinematic
  human-object data into dynamically feasible robot trajectories, with virtual contact guidance; about an order of
  magnitude faster than RL retargeting. The kit's SP-SR (10 cm / 0.5 rad) is named `_spider_success`.
- **ManipTrans** (CVPR 2025, arXiv 2503.21860): a pretrained generalist hand-trajectory imitator plus a residual
  module fine-tuned under interaction constraints. The kit's MP-SR (3 cm / 30 deg) is named `_maniptrans_success`.
  Both are retargeting methods that assume an accurate object reference. The reference, not the policy, is the hard
  part from video.

## Object 6-DoF tracking from egocentric RGB-D

- **BundleSDF** (CVPR 2023, arXiv 2303.14158): model-free tracking and reconstruction of an unknown rigid object from
  RGB-D video, given a first-frame mask. It fits our data: stereo depth plus frame-0 masks, no scan needed.
- **FoundationPose**: model-based or model-free 6-DoF pose and tracking. The model-free mode needs reference views and
  Kaolin. **ObjectForesight** (arXiv 2601.05237) builds egocentric object trajectories with FoundationPose,
  initialised on up to five clean views and refined by depth alignment and silhouette consistency.
- **HOT3D** (CVPR 2025, arXiv 2411.19167): egocentric hand-object benchmark; multi-view pose methods clearly beat
  single-view ones. Our stereo pair is a two-view setup.

## The official pipeline and public team evidence

- `nvidia-isaac/video_to_data` (pinned under `upstream/`): video -> depth, masks, mesh, 6-DoF object pose, MANO hands
  -> `task.parquet` -> FlashCHORD (Sharpa Wave), the reference route.
- Team repo `zjifan121-source/NVIDIA_V2D_Egocentric` (README, 2026-10-13/15), kit metric code on the 20 public
  episodes:

  | Case | AUC | SP-SR | MP-SR | RPE cm | MPPE cm |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | ground truth | 1.000 | 1.00 | 1.00 | 0.04 | 0.05 |
  | trajectory scaled x1.2 | 0.775 | 1.00 | 0.45 | 2.56 | 2.56 |
  | + 2 cm position noise | 0.567 | 1.00 | 0.00 | 2.70 | 4.47 |
  | object frozen at frame 0 | 0.447 | 0.45 | 0.00 | 9.60 | 9.30 |

  NVIDIA's tissue-box sample, through the evaluator: 0.6 cm object error, **AUC 0.98**. Their video reconstruction:
  AUC 0.216 -> 0.240 (rest hold) -> 0.291 (scan rescaling). Stack: HaMeR/WiLoR + MANO hands, SAM3D mesh,
  FoundationPose pose, Grounding DINO + SAM2 masks, MoGe, DROID-SLAM, GeoCalib. Known failures: about 180-degree object
  flips (the same failure as our HR2 sweeper), objects starting 5-27 deg off a resting face and tipping over, jitter
  defeating support detection. Cost: full Sharpa recipe 250M steps, about 3.2 h per episode on an A6000 (about 64
  GPU-hours for 20 episodes), about 46 GB at `njmax` 1024 (they recommend 80 GB GPUs). Ours is an RTX 5080 with
  16 GB. Settings: `sim.fps` set to the stored fps (about 20.25) to skip resampling; `motion_speed` 1.0; wrist
  termination raised 0.15 -> 0.3 m.
- Team repo `tianny0078/v2d-track3`: multi-frame TSDF, WiLoR hands, Sharpa IK; training gate not passed; local
  success 0.

## What this means for us

1. An AUC near 0.99 needs a near-exact object trajectory (2 cm of noise already costs about 0.43 AUC) plus a policy
   that tracks it. Public honest video reconstructions reach about 0.24-0.29, comparable to our static 0.277. How the
   leader reaches 0.99 is not known to us; we do not guess at it, and we never use confidential references.
2. The structural levers in order: (a) object motion in the reference, which only pays if tracking error stays well
   under a few cm; (b) policies that follow a moving reference within our 16 GB GPU and the remaining hours; (c) mesh
   quality (CD-O) from whole-clip fusion.
3. The cheapest decisive test is an upper bound on train labels only: train our policy on one train episode's moving
   mocap reference and score it. If the policy stage cannot follow motion on 16 GB in the time left, dynamic tracking
   cannot land by 12:00 UTC 4 Nov, and that is recorded as a measured finding.

Sources: https://arxiv.org/abs/2511.09484, https://arxiv.org/abs/2503.21860, https://arxiv.org/abs/2303.14158,
https://arxiv.org/abs/2601.05237, https://arxiv.org/abs/2411.19167,
https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric, https://github.com/tianny0078/v2d-track3,
https://github.com/nvidia-isaac/video_to_data.
