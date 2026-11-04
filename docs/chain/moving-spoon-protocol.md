# Spoon policies on 35, 36, 37: rules fixed before any spoon number (4 Nov 2026, 13:32 UTC)

Owner, live, about 13:30 UTC: "i need a better model, my auc is decent but the rest of my metrics are shit". `moving-m2-protocol.md`, Scored stage, already names 35, 36 and 37 as the next episodes after 38. No score selected this route or any setting below, and none will: the board holds no pose score for this account.

## Method

The same as 38 in `moving-m2v-protocol.md`: fixed-prompt masks, camera solve, stereo and silhouette tracking of our own video-derived spoon, then the fixed rigid-grasp rule, seed 0, 66,763,776 transitions, update ratio 2.0, unedited evaluator. Only the object changes. One hashed spoon asset across the three clips, made from our own measurements, with no scan. The prompt is the object's name.

**No scan constant is needed.** The spoon has no public scan, and under the 2 Nov kit Kaggle registers each submitted mesh to the hidden scan. So the submitted mesh is the simulated asset, in its own body frame.

## Rules for each episode, in order 35, 36, 37

An episode changes only if all of these hold:

1. The reference passes the final M1 gate set unchanged (`moving-m1-sealed.md`, clause 2; `src/features/moving_reference/final_gates.py`), with the world residual on the perpendicular instrument, on that clip. **Limits do not move.** A code defect may be fixed only if it is a real defect, disclosed, with limits unchanged and no second look at the same clip's result to choose among fixes.
2. Stage 0 passes (`moving-m2-protocol.md`, Amendment 1, as in M2v). Any failure stops that episode.
3. Against its own reference, through the unedited evaluator at the final checkpoint: `T` and `Q` at least 0.50, `E` at most min(0.5 `E_S`, 3 cm), `Theta` at most min(0.5 `Theta_S`, 15 deg), end reached with no reset.
4. First pose within 1 cm and 2 degrees, unaligned.
5. Reprojection: median IoU 0.60, p10 0.30, 90 % of frames.
6. Its pack passes `check_release.sh --combined`.

**Any miss keeps that episode's v5 export, with no variation.** No midpoint kill rule: on 38 it would have stopped a run that finished well. No resume after a failed final result. Nothing else is tried in the time left. One training run per episode.

## What this cannot show

**There is no label for the spoon on any public episode we hold for this purpose, so no validation of the video path exists for this object.** 39's sealed pass was on the dust pan, and says less about a thin object that hands largely hide. Passing means the policy follows its goal, not that the goal is right. A spoon could score below its static export and pass every rule. A reader should weigh it accordingly, and the report says so.

## Clock

Planning stays to 17:00 UTC until the owner says otherwise. About an hour's GPU per policy. Last training launch 15:00 UTC, and the last pack starts by 16:15 UTC, with whatever has passed. If the owner confirms Kaggle's 22:00 UTC close, last launch 20:00 and last pack 21:00. Episode 38's pack does not wait for the spoons: it goes first.

The outcome goes in `moving-spoon-outcome.md`, whichever way it falls.
