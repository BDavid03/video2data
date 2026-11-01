# Depth student

Status: implemented and authorized on 2026-10-29 ("go across the board, this is a
PhD-level experiment"). Fully convolutional depth networks trained from random
weights on MoGe2 pseudo-depth, assessed by the shared research scorer, exported
to ONNX with an onnxruntime parity check.

```text
_config.py   torch-free architecture bounds/validation (shared with the research loop)
_models.py   unet (residual encoder-decoder, GroupNorm/SiLU, dilated or plain bottleneck,
             optional bottleneck self-attention), dilated (single-resolution stack),
             coursework (the original coursework CNN behind the same interface)
_losses.py   log-L1 (metric accuracy), SILog (scale-invariant structure),
             multi-scale gradient matching (edge fidelity); weighted sum
train.py     AdamW, cosine schedule with linear warmup, EMA weights, BF16 autocast,
             channels-last, augmentation (flip, colour, random crop), grad clipping
predict.py   per-row NPZ for a dataset group, optional flip TTA; kind depth_student.predict
export.py    torch.onnx.export (dynamic N/H/W, opset 18) + onnxruntime parity on real samples
```

CLI: `python -m track3.features.depth_student {train|predict|export} --request <json>`
(use `scripts/linux/use-linux-cuda.sh` or let `scripts/research_loop.py` drive it).

## Contracts

- Input: `scratch_depth.prepare` datasets (calibrated RGB and MoGe2 depth pairs),
  any side from 64 to 512 px (the shared loader limit was raised from 256).
  Only training-episode rows are loaded for training; the tuning group is
  predicted and scored by `learning_experiments.score`, holdout only by the
  research loop's `finalize`.
- Checkpoint: `checkpoints/final.pt` with `kind: depth_student.checkpoint`,
  architecture, data contract, partition and the request; EMA weights when
  `ema_decay > 0`.
- Prediction run: same NPZ arrays and record fields as `learning_experiments.predict`,
  so `score` and `ensemble` admit it (kind added to their allow lists).
- Export: `depth_student.onnx`, input `rgb` float32 NCHW in [0,1], output
  `depth_m` float32 N1HW metres; the run records opset, hash, bytes and the
  maximum relative difference against PyTorch on tuning samples and a 1.5×
  resolution; parity must be below 1e-3 or the export fails. The coursework
  scratch checkpoint is also exportable (wrapped as `arch=coursework`).

## Limitations

Random initialisation only; accuracy is bounded by the MoGe2 teacher and
assessed on two tuning episodes. Attention at the bottleneck is quadratic in
bottleneck tokens (fine up to 512 px with four stages).
