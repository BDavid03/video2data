"""Bounded research comparisons, frozen before a single final-holdout phase."""

from copy import deepcopy
from dataclasses import dataclass
import math

from track3.paths import project_path
from track3.provenance import input_file, read_run, write_json
from ._common import ROOT, authorize, episode_partition, heavy_execution, integer, operation_run, read_json
from .planning import PlanRequest


@dataclass(frozen=True)
class InvestigateRequest:
    previous_tuning_run: str = "outputs/learning_experiments/tune/20261018T105658.305032Z-7407936a/run.json"
    teacher_samples_per_episode: int = 16
    prepared_max_image_side: int = 128
    neural_epochs: int = 20
    scratch_epochs: int = 10
    batch_size: int = 4
    seed: int = 31
    compiler_experiments: bool = True

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if not isinstance(self.previous_tuning_run, str) or not self.previous_tuning_run.strip():
            raise ValueError("Supply an explicit completed previous tuning run.json")
        for name, low, high in (("teacher_samples_per_episode", 1, 64),
                                ("prepared_max_image_side", 64, 128), ("neural_epochs", 1, 100),
                                ("scratch_epochs", 1, 100), ("batch_size", 1, 8),
                                ("seed", 0, 2**31 - 1)):
            integer(getattr(self, name), name, low, high)
        if type(self.compiler_experiments) is not bool:
            raise ValueError("compiler_experiments must be boolean")


def _recipes(previous, request):
    completed = [trial for trial in previous.get("trials", [])
                 if trial.get("status") == "completed" and type(trial.get("score")) in (int, float)
                 and math.isfinite(trial["score"]) and trial["score"] >= 0]
    candidates = []
    for technique in ("ann", "cnn", "transformer", "gradient_boosting"):
        trials = [trial for trial in completed if trial.get("technique") == technique]
        if not trials:
            raise ValueError(f"Previous tuning must contain a completed {technique} recipe")
        best = min(trials, key=lambda trial: (trial["score"], trial["trial_id"]))
        parameters = deepcopy(best["parameters"])
        changes = {"seed": {"before": best["trial_seed"], "after": request.seed}}
        if technique != "gradient_boosting":
            for name, value in (("epochs", request.neural_epochs), ("batch_size", request.batch_size)):
                changes[name] = {"before": parameters.get(name, 10 if name == "epochs" else 2),
                                 "after": value}
                parameters[name] = value
        candidates.append({"name": f"refit_{technique}", "technique": technique,
                           "parameters": parameters, "seed": request.seed,
                           "previous_trial_id": best["trial_id"], "recipe_changes": changes})
        if technique == "transformer":
            candidates.append({"name": "incumbent_transformer", "technique": technique,
                               "parameters": deepcopy(best["parameters"]), "seed": best["trial_seed"],
                               "checkpoint": deepcopy(best["checkpoint"]),
                               "previous_trial_id": best["trial_id"], "recipe_changes": {},
                               "frozen_incumbent": True})
    variants = {"eager": {}, "bf16": {"bf16_autocast": True},
                "channels_last": {"channels_last": True}, "fused_adam": {"fused_adam": True},
                "bf16_channels_last_fused": {"bf16_autocast": True, "channels_last": True,
                                             "fused_adam": True}}
    for name, options in variants.items():
        candidates.append({"name": f"scratch_{name}", "technique": "scratch",
                           "parameters": {"epochs": request.scratch_epochs, "batch_size": request.batch_size,
                                          "seed": request.seed, **options}})
    return candidates


def investigate_models(request: InvestigateRequest):
    authorize("investigate")
    heavy_execution("investigate")
    from track3.features.depth.generate import GenerateDepthRequest, generate_depth
    from track3.features.scratch_depth.prepare import PrepareDepthDataRequest, prepare_depth_data
    from track3.features.scratch_depth.train import TrainDepthRequest, train_depth
    from track3.features.scratch_depth.predict import PredictDepthRequest, predict_depth
    from .train import TrainRequest, train_model
    from .predict import PredictRequest, predict_model
    from .boosting_operations import BoostTrainRequest, BoostPredictRequest, train_boost, predict_boost
    from .score import ScoreRequest, score_predictions
    from .ensemble_operations import EnsembleRequest, ensemble_predictions
    from .scratch_adapter import ScratchAdapterRequest, adapt_scratch_predictions

    defaults = PlanRequest()
    partition = episode_partition(defaults.train_episodes, defaults.tuning_episodes, defaults.holdout_episodes)
    groups = {name: tuple(partition[name]) for name in ("train_episodes", "tuning_episodes", "holdout_episodes")}
    previous_path = project_path(ROOT, request.previous_tuning_run)
    previous = read_run(ROOT, previous_path, kind="learning_experiments.tune")
    if previous_path.name != "run.json" or previous.get("episode_partition") != partition:
        raise ValueError("Previous tuning must use the unchanged default episode partition")
    candidates = _recipes(previous, request)
    videos_path = project_path(ROOT, "reports/data-audit/videos.json")
    episodes_path = project_path(ROOT, "manifests/dataset/episodes.json")
    videos, episodes = read_json(videos_path), read_json(episodes_path)
    if (videos.get("status") != "completed" or videos.get("revision") != partition["dataset_revision"]
            or episodes.get("revision") != partition["dataset_revision"]):
        raise ValueError("Teacher selection requires completed pinned video/episode inventories")
    clocks = {}
    for episode in (*defaults.train_episodes, *defaults.tuning_episodes, *defaults.holdout_episodes):
        matches = [row for row in episodes["episodes"]
                   if row["split"] == "public" and row["episode_index"] == episode]
        if len(matches) != 1 or matches[0]["issues"]:
            raise ValueError("Teacher episode must have one usable public inventory entry")
        rows = [row for row in videos["videos"] if row["path"] == matches[0]["videos"]["ego_cam_a"]]
        if len(rows) != 1 or rows[0]["status"] != "ok":
            raise ValueError("Teacher episode lacks a usable camera clock")
        count = integer(rows[0]["decoded_frame_count"], "decoded_frame_count", 1, 1000000)
        clocks[episode] = {"count": count, "stride": math.ceil(count / request.teacher_samples_per_episode)}

    with operation_run("investigate", request, inputs=[videos_path, episodes_path],
                       parents=[previous_path]) as (directory, record):
        record.update(stage="teacher_training_tuning", episode_partition=partition, candidates=candidates,
                      teacher_schedule={str(key): value for key, value in clocks.items()},
                      teacher_runs={}, child_runs=[], compiler_experiments=[],
                      selection_metric="equal_episode_log_depth_mae", final_holdout_used_for_selection=False)

        def persist():
            write_json(directory / "run.json", record)

        def retain(path):
            path = project_path(ROOT, path)
            entry = input_file(ROOT, path)
            record["inputs"].append(entry)
            owner = path if path.name == "run.json" else (
                path.parent.parent / "run.json" if path.suffix == ".pt" else path.parent / "run.json")
            owner_entry = input_file(ROOT, owner)
            if owner != path:
                record["inputs"].append(owner_entry)
            child = read_json(owner)
            record["child_runs"].append({**owner_entry, **{key: child[key] for key in
                ("kind", "status", "timing", "gpu", "gpu_torch", "error") if key in child}})
            persist()
            return entry

        def failed(error, outcome):
            outcome.update(status="failed", error=f"{type(error).__name__}: {error}")
            if getattr(error, "feature_run_path", None) is not None:
                outcome["failed_run"] = retain(error.feature_run_path)
            persist()

        def teachers(selected):
            for episode in selected:
                print(f"Investigate teacher episode {episode}, stride {clocks[episode]['stride']}", flush=True)
                path = generate_depth(GenerateDepthRequest(episode_index=episode, stop_frame=None,
                    frame_stride=clocks[episode]["stride"], max_image_side=768))
                record["teacher_runs"][str(episode)] = retain(path)
                persist()

        def prepare(selected):
            return retain(prepare_depth_data(PrepareDepthDataRequest(
                teacher_runs=[record["teacher_runs"][str(e)]["path"] for e in selected],
                frame_stride=1, max_image_side=request.prepared_max_image_side)))

        def predict(candidate, dataset, group):
            checkpoint = candidate["checkpoint"]["path"]
            if candidate["technique"] == "scratch":
                runs = []
                for episode in partition[f"{group}_episodes"]:
                    path = predict_depth(PredictDepthRequest(checkpoint=checkpoint, episode_index=episode,
                        stop_frame=clocks[episode]["count"], frame_stride=clocks[episode]["stride"]))
                    runs.append(retain(path)["path"])
                return adapt_scratch_predictions(ScratchAdapterRequest(
                    dataset=dataset, prediction_runs=runs, group=group, **groups))
            count = sum(row["episode_index"] in partition[f"{group}_episodes"]
                        for row in read_json(dataset)["samples"])
            if candidate["technique"] == "gradient_boosting":
                return predict_boost(BoostPredictRequest(dataset=dataset, checkpoint=checkpoint,
                                                        group=group, max_samples=count))
            return predict_model(PredictRequest(dataset=dataset, checkpoint=checkpoint,
                                                group=group, max_samples=count))

        def assess(candidate, dataset, group):
            outcome = candidate[group] = {"status": "running", "stage": "prediction"}
            persist()
            try:
                if candidate["technique"] == "ensemble":
                    members = {entry["name"]: entry for entry in candidates}
                    path = ensemble_predictions(EnsembleRequest(
                        prediction_runs=[members[name][group]["prediction_run"]["path"]
                                         for name in candidate["members"]], weights=candidate["weights"]))
                else:
                    path = predict(candidate, dataset, group)
                outcome["prediction_run"] = retain(path)
                outcome["stage"] = "scoring"
                persist()
                score_path = score_predictions(ScoreRequest(dataset=dataset,
                    prediction_run=outcome["prediction_run"]["path"], group=group))
                outcome["scoring_run"] = retain(score_path)
                score = read_run(ROOT, score_path, kind="learning_experiments.score")
                value = score["score_value"]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Candidate score must be finite and nonnegative")
                outcome.update(status="completed", stage="completed", score=value)
            except Exception as error:
                failed(error, outcome)
            persist()

        training_tuning = [*defaults.train_episodes, *defaults.tuning_episodes]
        teachers(training_tuning)
        record["dataset"] = prepare(training_tuning)
        record["scratch_dataset"] = prepare(defaults.train_episodes)
        dataset, scratch_dataset = record["dataset"]["path"], record["scratch_dataset"]["path"]
        record["stage"] = "candidate_training_tuning"
        for candidate in candidates:
            candidate["training"] = {"status": "running"}
            persist()
            try:
                if candidate.get("frozen_incumbent"):
                    if input_file(ROOT, candidate["checkpoint"]["path"]) != candidate["checkpoint"]:
                        raise ValueError("Incumbent checkpoint changed since previous tuning")
                    retain(candidate["checkpoint"]["path"])
                else:
                    config = candidate["parameters"]
                    if candidate["technique"] == "scratch":
                        checkpoint = train_depth(TrainDepthRequest(dataset=scratch_dataset, **config))
                    elif candidate["technique"] == "gradient_boosting":
                        checkpoint = train_boost(BoostTrainRequest(dataset=dataset, parameters=config,
                                                                  seed=request.seed, **groups))
                    else:
                        architecture = {key: value for key, value in config.items()
                                        if key not in ("epochs", "batch_size", "learning_rate", "weight_decay")}
                        checkpoint = train_model(TrainRequest(dataset=dataset, technique=candidate["technique"],
                            model_parameters=architecture, epochs=request.neural_epochs, batch_size=request.batch_size,
                            learning_rate=config.get("learning_rate", 0.001),
                            weight_decay=config.get("weight_decay", 0.00001), seed=request.seed, **groups))
                    candidate["checkpoint"] = retain(checkpoint)
                candidate["training"]["status"] = "reused" if candidate.get("frozen_incumbent") else "completed"
            except Exception as error:
                failed(error, candidate["training"])
                continue
            assess(candidate, dataset, "tuning")

        if request.compiler_experiments:
            record["stage"] = "compiler_experiments"
            persist()
            try:
                from track3.features.scratch_depth.compiler_experiments import (
                    CompilerExperimentRequest, run_compiler_experiments)
                record["compiler_experiments"] = run_compiler_experiments(
                    CompilerExperimentRequest(dataset=scratch_dataset, seed=request.seed))
                for experiment in record["compiler_experiments"]:
                    if experiment.get("run"):
                        experiment["run_identity"] = retain(experiment["run"])
            except Exception as error:
                outcome = {"backend": "compiler_experiments"}
                record["compiler_experiments"].append(outcome)
                failed(error, outcome)

        successful = sorted((candidate for candidate in candidates
                             if candidate.get("tuning", {}).get("status") == "completed"),
                            key=lambda candidate: (candidate["tuning"]["score"], candidate["name"]))
        if not successful:
            raise RuntimeError("No candidate has an admitted tuning score; holdout remains untouched")
        top = successful[:3]
        record["stage"] = "ensemble_tuning"
        if len(top) >= 2:
            ensembles = [("top2_equal", top[:2], [0.5, 0.5])]
            if len(top) == 3:
                inverse = [1 / max(candidate["tuning"]["score"], 1e-12) for candidate in top]
                ensembles += [("top3_equal", top, [1 / 3] * 3),
                              ("top3_inverse_mae", top, [value / sum(inverse) for value in inverse])]
            for name, members, weights in ensembles:
                candidate = {"name": name, "technique": "ensemble",
                             "members": [member["name"] for member in members], "weights": weights}
                candidates.append(candidate)
                assess(candidate, dataset, "tuning")
        eligible = [candidate for candidate in candidates if candidate.get("tuning", {}).get("status") == "completed"]
        winner = min(eligible, key=lambda candidate: (candidate["tuning"]["score"], candidate["name"]))
        frozen = {"schema_version": 1, "research_only": True, "submission_candidate": False,
                  "episode_partition": partition, "candidates": deepcopy(candidates),
                  "selected_candidate": winner["name"], "selection_group": "tuning",
                  "final_holdout_used_for_selection": False, "compiler_results_used_for_selection": False,
                  "tie_break": "Minimum tuning score then lexicographic candidate name"}
        write_json(directory / "frozen-selection.json", frozen)
        record["frozen_selection"] = input_file(ROOT, directory / "frozen-selection.json")
        record["inputs"].append(record["frozen_selection"])
        record["selected_candidate"] = winner["name"]
        record["stage"] = "holdout_teacher_generation_after_freeze"
        persist()
        teachers(defaults.holdout_episodes)
        record["holdout_dataset"] = prepare([*training_tuning, *defaults.holdout_episodes])
        record["stage"] = "frozen_holdout_assessment"
        for candidate in candidates:
            if candidate["technique"] == "ensemble" or "checkpoint" in candidate:
                assess(candidate, record["holdout_dataset"]["path"], "holdout")
        if input_file(ROOT, directory / "frozen-selection.json") != record["frozen_selection"]:
            raise ValueError("Frozen candidate definitions changed during holdout assessment")
        outcomes = [outcome for candidate in candidates for group in ("training", "tuning", "holdout")
                    if (outcome := candidate.get(group)) is not None] + record["compiler_experiments"]
        record.update(stage="completed", completed_experiments=sum(o["status"] == "completed" for o in outcomes),
                      failed_experiments=sum(o["status"] == "failed" for o in outcomes))
        write_json(directory / "result.json", {key: record[key] for key in
            ("selected_candidate", "frozen_selection", "candidates", "compiler_experiments", "teacher_runs",
             "dataset", "scratch_dataset", "holdout_dataset", "completed_experiments", "failed_experiments",
             "final_holdout_used_for_selection", "child_runs")})
    return directory / "result.json"
