# Research reference for egocentric video to robot policies

Research cutoff: **18 October 2026**. These eight standalone volumes explain the architectures, supervision, mathematics and reasoning relevant to the project, with primary-source citations and project-specific deductions. The [complete compendium](egocentric-video-to-policy.md) brings all eight together, with a shared contents list and reading map.

| Volume | Subject |
|---|---|
| [Problem and mathematical foundations](chapters/01-problem-and-mathematics.md) | Task definition, evidence boundaries, camera geometry, transforms, gauges, observability and joint inference |
| [Depth models](chapters/02-depth-models.md) | Eigen, MiDaS, DPT, DINOv2/v3, MoGe/2/3, Depth Anything V2/3, Depth Pro, Metric3Dv2, UniDepthV2 and video depth |
| [Stereo motion and surfaces](chapters/03-stereo-motion-and-surfaces.md) | Epipolar geometry, correspondence, uncertainty, scale anchoring, bundle adjustment, dynamic multiview models, TSDF and neural surfaces |
| [Hand and object interaction](chapters/04-hand-object-interaction.md) | Detection, segmentation, point tracking, MANO, HaMeR, WiLoR, object pose, joint reconstruction, contact and retargeting |
| [Learning and synthetic data](chapters/05-learning-and-synthetic-data.md) | Exact local architectures, losses, distillation, synthetic mixtures, ensembles, uncertainty, grouped selection and optimizers |
| [Retargeting and policies](chapters/06-retargeting-and-policies.md) | Contact mechanics, CHORD and pinned-runtime differences, residual control, PPO, SAC, curricula and achieved behavior |
| [Rust Candle and Burn](chapters/07-rust-candle-burn.md) | Version-specific APIs, backends, operator support, bilinear derivatives, dropout, optimizer and checkpoint migration, CubeCL and GPU requirements |
| [Systems evaluation and agenda](chapters/08-systems-evaluation-and-research-agenda.md) | Precision, compilation, performance/memory reasoning, independent metrics, pinned scoring, consumed holdout and feature-owned research priorities |

For depth/synthetic work, start with volumes 1, 2 and 5. For full reconstruction, follow 1, 3 and 4. For robot learning, read 4 and 6 before the evaluation sections in 8. For the Rust request, volume 7 provides the detailed comparison and volume 8 explains how to assess systems claims.

The principal conclusions are:

- Teacher agreement, independent geometric accuracy and achieved simulation behavior require separate evidence.
- Metric scale, camera motion, canonical object frames and timing connect every stage; local image quality alone does not establish a usable interaction trajectory.
- Synthetic labels can be exact within a renderer while their appearance distribution remains unlike real video. Mixture decisions need real tuning evidence.
- The reviewed Candle 0.11.0 and Burn 0.22.0/CubeK 0.3.0 GPU paths lack the bilinear backward operation needed to train the existing decoder unchanged. The Rust volume cites the exact release sources and develops faithful alternatives.

The chapter files are the maintained sources. The compendium reproduces them with chapter-scoped navigation and adjusted local links; edit the standalone source before refreshing that collection. Each chapter contains its own primary bibliography or source ledger. Published findings, source-visible implementation facts and proposed adaptations are distinguished throughout.

No tests, model runs, simulation, compiler experiments or dependency changes accompanied this research. Episode 41 stays excluded; the already consumed 43/44 holdout cannot select later recipes. Current instructions in [AGENTS.md](../../AGENTS.md) and the pinned [contract interpretation](../../reports/contract.md) continue to govern execution and competition claims.
