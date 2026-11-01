"""Consult Codex (GPT astra, high reasoning) on the ledger and obtain bounded next recipes.

The advisor only reads: it runs in Codex's read-only sandbox with the repository
as its working root, receives the ledger summary, the recipe space and pointers
to the skill files, and must answer with JSON matching the schema below. Every
proposal is validated against the recipe bounds; invalid or absent answers fall
back to the heuristic proposer. The full exchange is persisted per iteration.
"""

import json
import os
import shutil
import subprocess
import time

from ._common import FEATURE, ROOT, now, read_json, save
from .recipes import recipe_identity, space_description, validate_recipe

def _nullable(kind):
    return {"type": [kind, "null"]}


def _strict_object(fields):
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


PROPOSAL = _strict_object({
    "technique": {"type": "string", "enum": ["ann", "cnn", "transformer", "gradient_boosting", "scratch"]},
    "parameters": _strict_object({name: _nullable("integer") for name in
                                  ("width", "layers", "heads", "patch_size", "max_tokens", "estimators", "max_thresholds",
                                   "stages", "depth", "blocks_per_stage")}
                                 | {"learning_rate": _nullable("number"), "dropout": _nullable("number"),
                                    "arch": _nullable("string"), "norm": _nullable("string"), "bottleneck": _nullable("string"),
                                    "attention": _nullable("boolean")}),
    "training": _strict_object({"epochs": _nullable("integer"), "batch_size": _nullable("integer"), "warmup_epochs": _nullable("integer"),
                                "learning_rate": _nullable("number"), "weight_decay": _nullable("number"),
                                "ema_decay": _nullable("number"), "augment_color": _nullable("number"), "augment_crop": _nullable("number"),
                                "grad_clip": _nullable("number"), "loss_log_l1": _nullable("number"), "loss_silog": _nullable("number"),
                                "loss_gradient": _nullable("number"), "brightness_jitter": _nullable("number"),
                                "schedule": _nullable("string"),
                                "bf16_autocast": _nullable("boolean"), "channels_last": _nullable("boolean"),
                                "fused_adam": _nullable("boolean"), "torch_compile": _nullable("boolean"),
                                "augment_flip": _nullable("boolean"), "flip_tta": _nullable("boolean")}),
    "data": _strict_object({"teacher_samples_per_episode": _nullable("integer"),
                            "prepared_max_image_side": _nullable("integer")}),
    "seed": {"type": "integer"}, "replicates": _nullable("integer"), "note": {"type": "string"}})

SCHEMA = _strict_object({
    "analysis": {"type": "string", "description": "What the ledger shows: which factors matter, failures, trends"},
    "hypothesis": {"type": "string", "description": "The single hypothesis the next proposals test"},
    "proposals": {"type": "array", "items": PROPOSAL,
                  "description": "1-3 recipes; use null for fields that do not apply to the technique"},
    "stop": {"type": "boolean", "description": "True only if further iterations are very unlikely to improve"},
    "stop_reason": {"type": "string"}})


def _strip(proposal):
    """Drop null fields the strict schema forces the advisor to spell out."""
    return {key: ({k: v for k, v in value.items() if v is not None} if isinstance(value, dict) else value)
            for key, value in proposal.items() if value is not None}


SKILLS = ("ML Evaluation.md", "Neural Network Architecture.md", "Computer Vision Pipeline.md", "ML Model Selection.md")


def _prompt(summary, budget):
    skills = ", ".join(f".agents/skills/ai-ml-skills/{name}" for name in SKILLS)
    return f"""You are the research advisor for a closed-loop depth-estimation study (NVIDIA Video-to-Data
Challenge, track 3, egocentric RGB -> metric depth). Small models are trained from scratch on MoGe2
pseudo-depth targets and selected on tuning episodes 39/42 by equal-episode log-depth MAE (lower is
better). Companion metrics: abs_rel, rmse_log, silog (scale-invariant), delta1 (fraction within 1.25x,
higher is better), mean_log_bias (sign of scale error), support_fraction, generalisation_gap
(tuning MAE - final training MAE), epochs_to_95pct (convergence speed), cost (train seconds, GPU MiB).

Selection is NOT MAE alone. A candidate must pass usability gates (support, delta1, |bias|, edge
correlation = gradient agreement with the teacher i.e. real structure rather than blur, episode spread)
and is then ranked by the composite in selection_rule (lower is better). Propose recipes that improve the
composite: sharper structure (edge_correlation, silog), correct scale (bias), high delta1, not just MAE.

This is a doctoral-level study: favour controlled comparisons (one factor at a time against the
incumbent), replicate the leader across seeds (replicates=3) before trusting small margins
(tuning_log_mae_std tells you the noise floor), run ablations that attribute gains (loss terms,
resolution, attention, EMA, augmentation, capacity), and scale resolution/data when the model is
capacity- or data-limited. The "student" technique (fully convolutional UNet/dilated/coursework with
SILog + gradient losses, cosine warmup, EMA, augmentation, flip TTA, up to 512px) is the main model
line; the older ann/cnn/transformer/boosting families are baselines and are capped at 128px.

Your job: examine the ledger, explain what is driving the results, state ONE hypothesis and propose
1-3 recipes that test it inside the bounded space. Prefer changes with mechanistic justification
(capacity vs data, receptive field, learning-rate/epoch interaction, resolution, BF16 numerics,
boosting depth). Do not repeat tried identities. Say stop=true only when the evidence shows the
space is exhausted or the budget is better saved.

Rules: holdout episodes 43/44 are never used here; episode 41 is excluded; no new dependencies;
no pretrained weights. The project guidance you may read: {skills}, and the slice code at
track3/features/auto_research/ (recipes.py defines the space). Reply ONLY with the JSON object.

Remaining budget: {json.dumps(budget)}

Recipe space:
{json.dumps(space_description(), indent=1)}

Ledger summary:
{json.dumps(summary, indent=1)}
"""


def consult(summary, budget, directory, settings):
    """Run codex exec; return a validated proposal dict (possibly with no valid proposals)."""
    record = {"started_utc": now(), "settings": settings, "status": "running", "proposals": [], "invalid": []}
    prompt = _prompt(summary, budget)
    (directory / "advisor-prompt.md").write_text(prompt, encoding="utf-8")
    schema_path = directory / "advisor-schema.json"
    save(schema_path, SCHEMA)
    output_path = directory / "advisor-output.json"
    command = settings.get("command", "codex")
    if shutil.which(command) is None:
        record.update(status="unavailable", error=f"{command} not on PATH")
        save(directory / "consultation.json", record)
        return record
    args = [command, "exec", "-m", settings["model"], "-c", f"model_reasoning_effort={settings['reasoning_effort']}",
            "-s", settings.get("sandbox", "read-only"), "-C", str(ROOT), "--skip-git-repo-check", "--ephemeral",
            "--color", "never", "--output-schema", str(schema_path), "-o", str(output_path), "-"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(args, input=prompt, capture_output=True, text=True,
                                   timeout=settings.get("timeout_seconds", 900), cwd=ROOT,
                                   env={**os.environ, "NO_COLOR": "1"})
        (directory / "advisor-stdout.log").write_text(completed.stdout[-200000:], encoding="utf-8")
        (directory / "advisor-stderr.log").write_text(completed.stderr[-50000:], encoding="utf-8")
        record["exit_code"] = completed.returncode
        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"codex exited {completed.returncode}")
        answer = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["wall_seconds"] = time.perf_counter() - started
        save(directory / "consultation.json", record)
        return record
    record.update(analysis=answer.get("analysis"), hypothesis=answer.get("hypothesis"),
                  stop=bool(answer.get("stop")), stop_reason=answer.get("stop_reason"))
    tried = set(summary.get("tried_identities", []))
    for proposal in answer.get("proposals", [])[:3]:
        try:
            recipe = validate_recipe(_strip(proposal))
            identity = recipe_identity(recipe)
            if identity in tried:
                raise ValueError("identity already tried")
            tried.add(identity)
            record["proposals"].append(recipe)
        except Exception as error:
            record["invalid"].append({"proposal": proposal, "error": f"{type(error).__name__}: {error}"})
    record.update(status="completed", wall_seconds=time.perf_counter() - started, finished_utc=now())
    save(directory / "consultation.json", record)
    return record


def default_settings():
    return dict(read_json(FEATURE / "runtime.json")["advisor"])
