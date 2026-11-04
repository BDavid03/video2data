# Questions for the V2D organisers

Drafted 2026-09-30 for the user to post. Nothing here has been sent. The
organisers named two channels: the `#v2d-challenge` channel on the NVIDIA
Omniverse Discord, with a thread per track, and the participant webinar on
7 October 2026 at 11:00 Pacific, which is 8 October at 05:00 in Sydney.

The questions are in priority order. The first three change the design if the
answer goes against it. Each is written to stand alone, so they can be posted one
at a time. The reasoning behind each is in
[risks and open questions](../docs/design/risks-and-open-questions.md).

---

## Post 1: frame alignment (Track 3)

> In the Submission Kit, `pack_submission.py` matches rollout rows to the roster
> by integer frame, and `evaluate_policy.py` writes `reference_frame = step +
> start_frame` on the reference after it has been resampled to the control rate.
> The released `sharpa_flash_sac` recipe uses `task.motion_speed: 0.5` at 20
> control steps per second, so a rollout has about two steps per dataset frame.
> The Track 3 roster has one row per video frame.
>
> How should rollout frames be aligned with dataset frames? Is training with
> `task.motion_speed=1.0` and a 20 fps reference the intended way, or should a
> half-speed rollout be resampled before packing?

## Post 2: `wooden_spoon` (Track 3)

> `data/track_3_evaluation_objects.json` lists `wooden_spoon` in ten scored
> episodes (3, 5, 6, 15, 16, 32, 33, 35, 36, 37). The public release has meshes
> and URDFs for ten objects and `wooden_spoon` is not among them.
>
> Will its scan be released? If not, where is the origin of its reference body
> frame and how are its axes oriented? The pose metrics align one world
> transform per episode, so a different body frame would be scored as error.

## Post 3: allowed inputs (Track 3)

> At evaluation, may a method use:
> 1. the two mono stereo streams, `ego_cam_b` and `ego_cam_c`;
> 2. the calibration file published under `track_3/public/meta/`;
> 3. the public object scans and URDFs, for pose tracking and for the simulated
>    scene;
> 4. a small amount of manual annotation on an evaluation video, such as one
>    click to start a segmentation mask?

## Post 4: evaluator and embodiment

> Must Track 3 rollouts be produced by the official `evaluate_policy.py`? The
> evaluator-hash allowlists in `config/tracks.json` are empty. Will they be
> filled, and would that affect submissions made before then?
>
> Is any robot embodiment required for Track 3, or may participants choose among
> those FlashCHORD provides?

## Post 5: mesh leaderboard

> For the Track 3 CD-O leaderboard: may the public scans be submitted, or must
> every mesh be reconstructed from the evaluation videos? May one reconstruction
> of an object be submitted for several episodes that contain that object?

## Post 6: quota and final selection

> When does the five-per-week submission count reset, and at what time do the
> final three days of unlimited submissions begin? How many final submissions
> are selected per leaderboard, and how are the six Track 3 leaderboards combined
> into a ranking?

## Post 7: reference and physics

> Does the confidential Track 3 reference contain a pose for frames where motion
> capture lost the object, and are those frames scored?
>
> What mass and friction do your reference scenes use for the objects? The public
> URDFs describe their mass as a placeholder.

## Post 8: hardware and licences

> The FlashCHORD README names a 48 GiB GPU for full training. Is training on a
> 16 GB GPU supported, and which world count, replay size and step budget would
> you recommend?
>
> May a method use pretrained models released under non-commercial or
> research-only licences, such as MANO-based hand estimators?

## Post 9: FAQ

> The announcement mentions an FAQ. Where is it? It is not linked from the
> challenge page.

---

## Recording the answers

Put each answer in `reports/contract.md` under the requirement it settles, with
the date, the channel and the name of the person who answered. An answer that
reverses a decision is also recorded in
[decisions.md](../docs/design/decisions.md) under that decision.

---

## Added 2026-10-03

Two further posts, from the [error budget](../docs/design/error-budget.md). Not
sent. Post 10 belongs with Post 1 and Post 11 with Post 2.

## Post 10: the first rollout row (Track 3)

> `evaluate_policy.py` records object poses after each control step, so the row
> with `reference_frame = 0` holds the state one step after the reset, not the
> reset state. The roster's frame 0 is the first video frame. Is that one-step
> offset intended, and is the reference built the same way?

## Post 11: scan body frames (Track 3)

> Do the public Track 3 scans share a body-frame convention, for example origin
> at the centre of the base with z up when the object rests on the table? If
> `wooden_spoon`'s scan cannot be released, does its reference frame follow the
> same convention, and which way does its long axis point?
