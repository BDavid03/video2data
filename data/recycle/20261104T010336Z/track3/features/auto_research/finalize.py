"""Freeze the tuning-selected winner (and tuning-selected ensembles), then assess holdout 43/44 once.

Selection uses only tuning evidence from the ledger. The frozen file is written
before any holdout teacher is generated; holdout outcomes never change it.
"""

from dataclasses import dataclass
import traceback

from track3.paths import project_path
from track3.provenance import input_file, read_run, sha256_file
from ._common import ROOT, authorize, heavy_execution, new_run_id, now, read_json, relative, save, state_dir
from . import campaign, ledger
from .metrics import extended_metrics


@dataclass(frozen=True)
class FinalizeRequest:
    top_k: int = 3
    ensembles: bool = True

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")


def _assess_ensemble(prediction_runs, weights, dataset, group):
    from track3.features.learning_experiments.ensemble_operations import EnsembleRequest, ensemble_predictions
    from track3.features.learning_experiments.score import ScoreRequest, score_predictions
    prediction = ensemble_predictions(EnsembleRequest(prediction_runs=list(prediction_runs), weights=list(weights)))
    scoring = score_predictions(ScoreRequest(dataset=dataset, prediction_run=relative(prediction), group=group))
    record = read_run(ROOT, scoring, kind="learning_experiments.score")
    return relative(prediction), relative(scoring), record


def finalize(request: FinalizeRequest):
    authorize("finalize")
    heavy_execution("finalize")
    entries = ledger.load()
    done = sorted(ledger.completed(entries), key=lambda e: (ledger._objective(e), e["tuning_log_mae"], e["iteration"]))
    if not done:
        raise RuntimeError("No completed iteration to finalize")
    directory = state_dir() / "finalize" / new_run_id()
    directory.mkdir(parents=True)
    registry = campaign.Registry()
    # Distinct top-k by identity, all on the same data setting as the best (ensembles need one grid)
    best = done[0]
    data = best["recipe"]["data"]
    same_data = [e for e in done if e["recipe"]["data"] == data]
    top = []
    for e in same_data:
        if all(e["identity"] != t["identity"] for t in top):
            top.append(e)
        if len(top) == request.top_k:
            break
    candidates = [{"name": f"iteration_{e['iteration']}_{e['technique']}", "technique": e["technique"], "iteration": e["iteration"],
                   "recipe": e["recipe"], "checkpoint": input_file(ROOT, e["checkpoint"]),
                   "tuning": {"status": "completed", "score": e["tuning_log_mae"], "prediction_run": e["prediction_run"],
                              "scoring_run": e["scoring_run"]}} for e in top]
    record = {"run_id": directory.name, "started_utc": now(), "request": request.__dict__, "status": "running",
              "data": data, "candidates": candidates, "stage": "tuning_ensembles"}
    save(directory / "run.json", record)
    inputs = campaign.ensure_inputs(data, registry, include_holdout=False)
    if request.ensembles and len(top) >= 2:
        specs = [("top2_equal", top[:2], [0.5, 0.5])]
        if len(top) >= 3:
            inverse = [1 / max(c["tuning"]["score"], 1e-12) for c in candidates[:3]]
            specs += [("top3_equal", top[:3], [1 / 3] * 3), ("top3_inverse_mae", top[:3], [v / sum(inverse) for v in inverse])]
        for name, group_members, weights in specs:
            candidate = {"name": name, "technique": "ensemble", "members": [f"iteration_{m['iteration']}_{m['technique']}" for m in group_members],
                         "weights": weights, "tuning": {"status": "running"}}
            candidates.append(candidate)
            try:
                prediction, scoring, score = _assess_ensemble([m["prediction_run"] for m in group_members], weights,
                                                              inputs["dataset"], "tuning")
                candidate["tuning"].update(status="completed", score=score["score_value"], prediction_run=prediction,
                                           scoring_run=scoring)
            except Exception as error:
                candidate["tuning"].update(status="failed", error=f"{type(error).__name__}: {error}")
            save(directory / "run.json", record)
    eligible = [c for c in candidates if c["tuning"]["status"] == "completed"]
    # Ensembles are ranked on MAE among themselves; the frozen winner is the usable-composite leader unless an
    # ensemble beats its MAE by a clear margin (ensembles of usable members inherit their structure).
    winner = candidates[0]
    for c in eligible:
        if c["technique"] == "ensemble" and c["tuning"]["score"] < winner["tuning"]["score"] * 0.97:
            winner = c
    frozen = {"schema_version": 1, "research_only": True, "submission_candidate": False, "frozen_utc": now(),
              "selection_group": "tuning", "selection_metric": "equal_episode_log_depth_mae",
              "selected_candidate": winner["name"], "candidates": candidates, "data": data,
              "final_holdout_used_for_selection": False, "tie_break": "minimum tuning score then name"}
    save(directory / "frozen-selection.json", frozen)
    frozen_hash = sha256_file(directory / "frozen-selection.json")
    record.update(frozen_selection=relative(directory / "frozen-selection.json"), frozen_sha256=frozen_hash,
                  selected_candidate=winner["name"], stage="holdout_after_freeze")
    save(directory / "run.json", record)
    inputs = campaign.ensure_inputs(data, registry, include_holdout=True)
    record["holdout_dataset"] = inputs["holdout_dataset"]
    for candidate in candidates:
        if candidate["tuning"]["status"] != "completed":
            continue
        candidate["holdout"] = {"status": "running"}
        save(directory / "run.json", record)
        try:
            if candidate["technique"] == "ensemble":
                by_name = {c["name"]: c for c in candidates}
                prediction, scoring, score = _assess_ensemble([by_name[m]["holdout"]["prediction_run"] for m in candidate["members"]],
                                                              candidate["weights"], inputs["holdout_dataset"], "holdout")
            else:
                prediction, scoring, score = campaign.predict_and_score(candidate["technique"], candidate["checkpoint"]["path"],
                                                                        inputs, "holdout", "holdout_dataset")
            candidate["holdout"].update(status="completed", score=score["score_value"], prediction_run=prediction,
                                        scoring_run=scoring,
                                        metrics=extended_metrics(inputs["holdout_dataset"], prediction, "holdout", score)["overall"])
        except Exception as error:
            candidate["holdout"].update(status="failed", error=f"{type(error).__name__}: {error}",
                                        traceback=traceback.format_exc()[-3000:])
        save(directory / "run.json", record)
    if sha256_file(directory / "frozen-selection.json") != frozen_hash:
        raise ValueError("Frozen selection changed during holdout assessment")
    export = None
    try:
        from track3.features.depth_student.export import ExportStudentRequest, export_student
        exportable = winner if winner["technique"] in ("student", "scratch") else next(
            (c for c in sorted(eligible, key=lambda c: c["tuning"]["score"]) if c["technique"] in ("student", "scratch")), None)
        if exportable is not None:
            path = export_student(ExportStudentRequest(checkpoint=exportable["checkpoint"]["path"], dataset=inputs["dataset"]))
            export = {"candidate": exportable["name"], "onnx": relative(path), "sha256": sha256_file(path),
                      "run": relative(path.parent / "run.json")}
    except Exception as error:
        export = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    record["onnx_export"] = export
    record.update(status="completed", stage="completed", finished_utc=now())
    save(directory / "run.json", record)
    save(directory / "result.json", {"selected_candidate": winner["name"], "frozen_selection": record["frozen_selection"],
                                     "candidates": candidates, "holdout_dataset": inputs["holdout_dataset"],
                                     "onnx_export": export, "final_holdout_used_for_selection": False})
    return directory / "result.json"
