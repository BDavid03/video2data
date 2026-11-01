"""The circular pipeline: propose -> run -> record -> visualise -> examine -> improve.

Each iteration owns outputs/auto_research/iterations/<run-id>/ with recipe,
outcome, any consultation, and the ledger line that references them. The loop
resumes from the ledger; stopping rules are iteration, time, patience and an
advisor stop that is honoured only after the minimum iteration count.
"""

from dataclasses import dataclass
import random
import time

from ._common import authorize, heavy_execution, new_run_id, now, read_json, save, state_dir
from . import advisor as advisor_module
from . import campaign, ledger, visualize
from .recipes import baseline_recipes, describe, mutate, recipe_identity, validate_recipe


@dataclass(frozen=True)
class LoopRequest:
    max_iterations: int = 24
    max_hours: float = 8.0
    min_iterations: int = 8
    patience: int = 8
    baselines: bool = True
    consult: bool = True
    consult_every: int = 1
    proposals_per_consultation: int = 2
    seed: int = 31

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if not 1 <= self.max_iterations <= 500 or not 0 < self.max_hours <= 168 or not 1 <= self.patience <= 100:
            raise ValueError("Loop budget out of bounds")
        if not 1 <= self.consult_every <= 10 or not 1 <= self.proposals_per_consultation <= 3:
            raise ValueError("Consultation cadence out of bounds")


def write_summary(entries, directory, status, extra=None):
    top = ledger.best(entries)
    lines = ["# Auto-research loop", "", f"Updated {now()}  status: **{status}**", "",
             f"Iterations: {len(entries)}  completed: {len(ledger.completed(entries))}  "
             f"since improvement: {ledger.iterations_since_improvement(entries)}", ""]
    if top:
        comp = f"{top['composite']:.4f}" if isinstance(top.get("composite"), (int, float)) else "-"
        edge = f"{top['edge_correlation']:.3f}" if isinstance(top.get("edge_correlation"), (int, float)) else "-"
        lines += [f"**Selected (usable, lowest composite):** iteration {top['iteration']} `{top['description']}`  "
                  f"composite **{comp}**  tuning log-depth MAE {top['tuning_log_mae']:.5f}  abs_rel {top['abs_rel']:.4f}  "
                  f"delta1 {top['delta1']:.3f}  edge {edge}  bias {top['mean_log_bias']:+.3f}", "",
                  "Selection rule: pass every usability gate (support >= 0.98, delta1 >= 0.70, |bias| <= 0.10, "
                  "edge correlation >= 0.40, episode spread <= 0.05), then minimise composite = "
                  "log-MAE + SILog + (1 - delta1) + |bias| + 0.5 (1 - edge). MAE alone never selects.", ""]
    lines += ["![dashboard](dashboard.png)", "", "| iter | source | technique | recipe | usable | composite | tuning MAE | delta1 | edge | bias | train s | status |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        mae = f"{e['tuning_log_mae']:.5f}" if isinstance(e.get("tuning_log_mae"), (int, float)) else "-"
        ar = f"{e['abs_rel']:.4f}" if isinstance(e.get("abs_rel"), (int, float)) else "-"
        d1 = f"{e['delta1']:.3f}" if isinstance(e.get("delta1"), (int, float)) else "-"
        ts = f"{e['train_wall_seconds']:.0f}" if isinstance(e.get("train_wall_seconds"), (int, float)) else "-"
        comp = f"{e['composite']:.4f}" if isinstance(e.get("composite"), (int, float)) else "-"
        edge = f"{e['edge_correlation']:.3f}" if isinstance(e.get("edge_correlation"), (int, float)) else "-"
        bias = f"{e['mean_log_bias']:+.3f}" if isinstance(e.get("mean_log_bias"), (int, float)) else "-"
        usable_flag = "yes" if e.get("usable") else ("no: " + "; ".join(e.get("failed_gates", [])[:2]))
        lines.append(f"| {e['iteration']} | {e['source']} | {e['technique']} | `{e['description']}` | {usable_flag} | {comp} | {mae} | {d1} | {edge} | {bias} | {ts} | {e['status']} |")
    if extra:
        lines += ["", *extra]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_loop(request: LoopRequest):
    authorize("loop")
    heavy_execution("loop")
    root = state_dir()
    iterations_dir = root / "iterations"
    iterations_dir.mkdir(exist_ok=True)
    entries = ledger.load()
    registry = campaign.Registry()
    rng = random.Random(request.seed + len(entries))
    settings = advisor_module.default_settings()
    tried = {e["identity"] for e in entries}
    queue = []  # (source, recipe)
    if request.baselines:
        # A baseline whose latest attempt failed (e.g. a numerics bug since fixed) is retried once per session.
        latest = {}
        for e in entries:
            latest[e["identity"]] = e["status"]
        for recipe in baseline_recipes():
            recipe = validate_recipe({**recipe, "seed": request.seed})
            identity = recipe_identity(recipe)
            if identity not in tried or latest.get(identity) == "failed":
                tried.discard(identity)
                queue.append(("baseline", recipe))
    session = {"run_id": new_run_id(), "request": request.__dict__, "started_utc": now(), "iterations": [],
               "status": "running", "stop_reason": None}
    save(root / "loop-session.json", session)
    started = time.perf_counter()
    inputs_cache = {}
    advisor_stop = False
    completed_this_session = 0
    try:
        while True:
            elapsed_hours = (time.perf_counter() - started) / 3600
            budget = {"iterations_left": request.max_iterations - completed_this_session,
                      "hours_left": round(request.max_hours - elapsed_hours, 2),
                      "iterations_since_improvement": ledger.iterations_since_improvement(entries),
                      "patience": request.patience}
            if completed_this_session >= request.max_iterations:
                session["stop_reason"] = "max_iterations"; break
            if elapsed_hours >= request.max_hours:
                session["stop_reason"] = "max_hours"; break
            if len(entries) >= request.min_iterations and ledger.iterations_since_improvement(entries) >= request.patience:
                session["stop_reason"] = "patience_exhausted"; break
            if advisor_stop and len(entries) >= request.min_iterations:
                session["stop_reason"] = "advisor_stop"; break
            if not queue:
                consulted = None
                if request.consult and (len(entries) % request.consult_every == 0 or not ledger.completed(entries)):
                    consult_dir = iterations_dir / f"{new_run_id()}-consult"
                    consult_dir.mkdir()
                    print(f"[auto_research] consulting {settings['model']} ({settings['reasoning_effort']})", flush=True)
                    consulted = advisor_module.consult(ledger.summary(entries), budget, consult_dir, settings)
                    session["iterations"].append({"consultation": str(consult_dir / "consultation.json"),
                                                  "status": consulted["status"]})
                    save(root / "loop-session.json", session)
                    for recipe in consulted.get("proposals", [])[:request.proposals_per_consultation]:
                        queue.append(("codex", recipe))
                    advisor_stop = bool(consulted.get("stop"))
                if not queue:
                    top = ledger.best(entries)
                    seed_recipe = top["recipe"] if top else validate_recipe(baseline_recipes()[0])
                    proposal = mutate(seed_recipe, rng, tried)
                    if proposal is None:
                        session["stop_reason"] = "no_untried_neighbour"; break
                    queue.append(("heuristic", proposal))
            source, recipe = queue.pop(0)
            identity = recipe_identity(recipe)
            if identity in tried:
                continue
            iteration = len(entries) + 1
            directory = iterations_dir / new_run_id()
            directory.mkdir()
            save(directory / "recipe.json", {"iteration": iteration, "source": source, "recipe": recipe})
            print(f"[auto_research] iteration {iteration} ({source}): {describe(recipe)}", flush=True)
            data_key = tuple(sorted(recipe["data"].items()))
            try:
                if data_key not in inputs_cache:
                    inputs_cache[data_key] = campaign.ensure_inputs(recipe["data"], registry)
                outcome = campaign.run_trial(recipe, inputs_cache[data_key], directory)
            except Exception as error:  # input preparation failed: record the iteration, keep looping
                outcome = {"recipe": recipe, "identity": identity, "status": "failed", "stage": "inputs",
                           "error": f"{type(error).__name__}: {error}", "started_utc": now(), "finished_utc": now(),
                           "wall_seconds": None}
                save(directory / "outcome.json", outcome)
            entry = ledger.append(ledger.entry_from_outcome(iteration, source, outcome, directory))
            entries.append(entry)
            tried.add(identity)
            completed_this_session += 1
            session["iterations"].append({"iteration": iteration, "run_id": directory.name, "status": outcome["status"],
                                          "tuning_log_mae": outcome.get("score")})
            top = ledger.best(entries)
            print(f"[auto_research] iteration {iteration} {outcome['status']} tuning MAE {outcome.get('score')} "
                  f"usable {entry.get('usable')} composite {entry.get('composite')} "
                  f"best iteration {top['iteration'] if top else None} composite {top.get('composite') if top else None}", flush=True)
            figures = visualize.render_all(entries, root)
            write_summary(entries, root, "running")
            save(root / "loop-session.json", session)
        session["status"] = "completed"
    except KeyboardInterrupt:
        session.update(status="interrupted", stop_reason="keyboard_interrupt")
    except Exception as error:
        session.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        session["finished_utc"] = now()
        session["wall_seconds"] = time.perf_counter() - started
        if entries:
            visualize.render_all(entries, root)
        write_summary(entries, root, session["status"], [f"Stop reason: {session['stop_reason']}"])
        save(root / "loop-session.json", session)
    return root / "summary.md"
