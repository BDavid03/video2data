# M2 outcome (4 Nov 2026, 08:37 UTC): GO on a mocap upper bound, tuning episode 39

Protocol `moving-m2-protocol.md`, Amendment 1, committed before any number here. Evidence: `data/outputs/moving_policy/ep39-a1/` (`result.md`, `stage0.json`, `summary.md`, `score-*/`). Nothing here changes a submitted file. **This is not a challenge result:** a learned policy tracks a reference taken from motion capture, on a tuning episode, and scored episodes have no mocap. A pass here says the recipe can do this physically; M2v and M1 must show a video-derived reference can feed it.

| Checkpoint (transitions) | Role | `T` | `Q` | `E` cm | `Theta` deg | Kit AUC | SP-SR | MPPE cm | Verdict |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16,154,624 | descriptive | 0.273 | 0.054 | 13.74 | 119.74 | 0.168 | 0 | 15.08 | NULL |
| 32,814,080 | descriptive | 0.996 | 0.977 | 1.03 | 1.46 | 0.951 | 1 | 1.08 | GO |
| 66,763,776 | **gated final** | **1.006** | **0.992** | **0.62** | **0.92** | **0.982** | 1 | 0.70 | **GO** |

Beside it: held still on 39, `E_S` 16.43 cm, `Theta_S` 51.08 deg, AUC about 0.17 (shipped T1 0.175). Seed 0, fresh start, one seed. 57.98 minutes of wall time for the whole run at about 20.8k transitions/s, 11.4 GiB sampled peak on the device.

## What made Stage 0 pass

The first Stage 0 launched the pan at 22 m/s (`ep39/`, kept). The diagnosis found the support-to-hull contact, and lowering the cube never fixed it. From the queue fixed first: S1, a footprint-sized 1 cm cylinder 1 mm below the imported hulls, failed at 4.14 degrees over 60 hands-cleared steps; **S2, S1 with the shipped task's 0.3 kg and diagonal 0.001 inertia, passed at 2.31 mm and 1.84 degrees**; S3 did not run. The grasp rule chose the longitudinal pair (308 of 308 frames closed, peak wrist 1.59 m/s); the transverse pair failed closure. Reachability passed at 1.90 and 1.52 cm mean, 11.07 cm maximum, against 3 and 15.

## Independent check by the lead

Separate script from raw parquets and its own formula (`data/.runtime/tmp/recheck-m2-27.py`): the task reference equals mocap within 2.4e-7 m and 1e-4 degrees; `E` 0.6187 cm (p95 1.77, maximum 2.67), `Theta` 0.922 degrees (p95 2.17, maximum 3.97); the achieved path is not a copy of the reference (no identical rows; 1.945 m travelled against mocap's 1.916 m; net rotation 108.3 against 108.9 degrees; height range 0.616 to 0.885 against 0.618 to 0.892 m). **Own AUC 0.9834 with a different 1,000-point sample of the scan and its own alignment, against the recorded 0.9822;** a held-still own AUC of 0.172 sits beside the shipped 0.175, so the metric is behaving. The export's checkpoint hash matches `policy_66763776.safetensors` (`62f75e22...d649`).

## What it does and does not show

- **Shown:** RL in simulation, from nothing, in under an hour, carries an object in simulation through a fast human motion (peak 71 cm/s, 109 degrees net), with the object moved only by hand contact, through the kit's unedited evaluator with a learned checkpoint. **The physics has limits, disclosed:** the evaluator's configuration turns robot-support collision off, and the hands start 58 and 39 mm through the support plane (critic 27); the object's 0.3 kg and diagonal inertia are not the real pan's.
- **Not shown:** that any reference from video is good enough; that the hold resembles a person's (the rigid grasp reference is our own rule, contact is not scored); anything about objects that are not this pan. One seed on one tuning episode. The simulated asset is our own reconstruction and has no scan geometry; the scan appears only in the kit measurement. The two descriptive rows show the recipe changes quickly between 16M and 33M transitions, which is not a reason to stop early on another episode.
- **The bottleneck has moved:** the policy's error here (AUC 0.98) is now far below what a video reference is likely to carry. M1's reference quality decides what the scored episodes can reach.

## Consequence by the protocol

**GO: M2v runs, once M1 passes its gates and sealed pass (`moving-m1-protocol.md`).** The per-frame achieved-minus-reference error for M1's allowance screen is at `score-66763776/achieved-minus-reference.npz`.
