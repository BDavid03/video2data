# Caveman Report: video2data (4 Nov 2026, about 15:10 UTC)

Kaggle close: 22:00 UTC today. Chain planned 17:00 UTC.

## What project

Entry in NVIDIA V2D Challenge Track 3. Watch human video. Rebuild object shapes (meshes). Train robot-hand policy in simulation to move object like person did.
Six Kaggle boards. CD-O scores shapes: lower is better. AUC, SP-SR, MP-SR, RPE, MPPE score poses. Since 2 Nov one file holds shapes and poses for all six.

## What done today

- Shapes scored: CD-O 0.53562 cm. First real score.
- Old pose file rejected: organisers changed kit on 2 Nov. New kit wants shapes and poses in one file.
- Fixed file built, checked: combined v5. Same poses. Owner approved exact command. UPLOADED 14:15 UTC. Scores: AUC 0.307, SP-SR 0.15, MP-SR 0, RPE 20.1 cm, MPPE 14.6 cm, CD-O 0.566 cm. Matches our local guess.
- Episode 38 (dust pan): moving policy trained from video goal. Passed every frozen gate. Packed as moving v5 (b476eb1c, commit c8ff8c1). UPLOADED 14:28 UTC. Scores: AUC 0.323, SP-SR 0.20, MP-SR 0, RPE 20.1 cm, MPPE 14.3 cm, CD-O 0.565 cm. Better or same as static on every board. Owner: mark it final on all six.
- Technical report (2 pages) built, on GitHub main: docs/report/final/track3-technical-report.pdf. Rewritten plain. All scores in. No AI-agent mention (owner asked).
- Spoons 35, 36, 37: masks too thin. Stopped at first gate. Keep static. Owner said keep strict gates.
- Code pushed to GitHub main.

## Why pose AUC low (0.277 old scorer, about 0.30 new)

Objects move in video. Our poses sit still on 19 of 20 episodes. Even perfect start pose held still gets only 0.532. Half of metric lost to motion. Only 38 moves. Leader has 0.99 because their objects move.

## What owner must do

1. Mark moving v5 final on all six boards in browser (IDs in NEEDS-YOU top).
2. Share GitHub repo with Nvidia-V2D-Challenge. Fill the form: report URL, repo URL, commit c8ff8c1, container link.
3. Container needs Docker group: sudo usermod -aG docker ceolwolf, new shell, scripts/container/owner-build.sh.
4. Approve any new upload. Chain uploads nothing without you now.

## Running now

Owner said improve the model. Codex on GPU: moving policy for two-object episodes (one object carried, other rests), same strict gates. Maybe 3 to 6 hours. Codex on CPU: pretrained pot and pitcher shapes into the file if frame map is clear.
