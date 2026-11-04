"""Sequential bounded search with tuning-only selection; no execution on import."""

from dataclasses import dataclass
import json
import math

from track3.paths import project_path
from track3.provenance import input_file, read_run, sha256_file, write_json
from ._common import FEATURE, ROOT, authorize, episode_partition, integer, number, operation_run, read_json
from .planning import PlanRequest, plan_trials
from .train import TrainRequest, train_model
from .predict import PredictRequest, predict_model


@dataclass(frozen=True)
class TuneRequest:
    plan: str | None = None
    dataset: str | None = None
    continue_on_failure: bool = True
    minimum_prediction_fraction: float = 0.9

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        for name, value in (("plan", self.plan), ("dataset", self.dataset)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be an explicit local artifact path")
        if type(self.continue_on_failure) is not bool:
            raise ValueError("continue_on_failure must be boolean")
        number(self.minimum_prediction_fraction, "minimum_prediction_fraction", 0.000001, 1)


def _admit_plan(path):
    owner = read_run(ROOT, path.parent / "run.json", kind="learning_experiments.plan")
    if path.name != "plan.json" or owner.get("artifacts", {}).get("plan.json") != sha256_file(path):
        raise ValueError("Plan must be the exact artifact of its completed planning run")
    document = read_json(path)
    # Rebuild metadata only. The plan operation has its own required flag;
    # tuning never implicitly grants permission to plan or train.
    expected = plan_trials(PlanRequest(**owner["request"]))
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Plan differs from its owning request and current finite search contract")
    definitions = owner.get("sources", [])
    for value in (FEATURE / "planning.py", FEATURE / "_config.py", FEATURE / "_common.py",
                  FEATURE / "search.json"):
        current = input_file(ROOT, value)
        if current not in definitions:
            raise ValueError("Planning sources changed; create a new explicitly authorized plan")
    integer(len(document["trials"]), "trial count", 1, 128)
    return document


def _tuning_count(data, partition):
    """Reject incomplete group metadata before spending any trial budget."""
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset must contain prepared sample rows")
    declared = set().union(*(partition[name] for name in
                             ("train_episodes", "tuning_episodes", "holdout_episodes")))
    seen, covered = set(), set()
    count = 0
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Prepared sample rows must be objects")
        episode, frame = row.get("episode_index"), row.get("frame_index")
        if (type(episode) is not int or episode not in declared
                or type(frame) is not int or frame < 0 or (episode, frame) in seen):
            raise ValueError("Prepared rows require unique episode/frame identities inside the partition")
        seen.add((episode, frame))
        covered.add(episode)
        count += episode in partition["tuning_episodes"]
    required = set(partition["train_episodes"]) | set(partition["tuning_episodes"])
    if not required.issubset(covered):
        raise ValueError("Prepared rows must cover every training and tuning episode")
    return integer(count, "complete tuning sample count", 1, 10000)


def tune_models(request: TuneRequest):
    authorize("tune")
    for operation in ("plan", "model", "train", "predict", "score"):
        authorize(operation)
    if request.plan is None or request.dataset is None:
        raise ValueError("Supply a completed plan artifact and a completed prepared dataset")
    plan_path, dataset_path = (project_path(ROOT, value) for value in (request.plan, request.dataset))
    admitted_inputs = [input_file(ROOT, path) for path in
                       (plan_path, plan_path.parent / "run.json",
                        dataset_path, dataset_path.parent / "run.json")]
    plan = _admit_plan(plan_path)
    partition = plan["episode_partition"]
    if partition != episode_partition(partition["train_episodes"], partition["tuning_episodes"],
                                      partition["holdout_episodes"]):
        raise ValueError("Plan no longer describes the frozen episode partition")
    if any(trial["technique"] == "gradient_boosting" for trial in plan["trials"]):
        authorize("boost_train")
        authorize("boost_predict")
    prepared = read_run(ROOT, dataset_path.parent / "run.json", kind="scratch_depth.prepare")
    if (dataset_path.name != "dataset.json"
            or prepared.get("artifacts", {}).get("dataset.json") != sha256_file(dataset_path)):
        raise ValueError("Dataset must be owned by its completed preparation run")
    data = read_json(dataset_path)
    tuning_count = _tuning_count(data, partition)
    if any(input_file(ROOT, entry["path"]) != entry for entry in admitted_inputs):
        raise ValueError("Plan, dataset or owning producer changed during admission")
    # Child imports are feature-local, and none imports model libraries eagerly.
    from .score import ScoreRequest, score_predictions
    from .boosting_operations import BoostTrainRequest, BoostPredictRequest, train_boost, predict_boost

    with operation_run("tune", request, inputs=[plan_path, dataset_path],
                       parents=[plan_path.parent / "run.json", dataset_path.parent / "run.json"]) as (directory, record):
        record.update(episode_partition=partition, plan_sha256=sha256_file(plan_path),
                      dataset_sha256=sha256_file(dataset_path), trials=[],
                      objective=plan["objective"], final_holdout_used_for_selection=False,
                      scheduling="Sequential trials; each neural child acquires the cooperative GPU lease",
                      selection=None)
        for planned in plan["trials"]:
            # A changed input invalidates the whole comparison, even when ordinary
            # child failures are allowed to consume their trial and continue.
            if any(input_file(ROOT, entry["path"]) != entry for entry in admitted_inputs):
                raise ValueError("Plan, dataset or owning producer changed during tuning")
            trial = {"trial_id": planned["trial_id"], "technique": planned["technique"],
                     "parameters": planned["parameters"], "trial_seed": planned["trial_seed"],
                     "status": "running", "stage": "training", "score": None}
            record["trials"].append(trial)
            write_json(directory / "run.json", record)
            try:
                groups = {name: tuple(partition[name]) for name in
                          ("train_episodes", "tuning_episodes", "holdout_episodes")}
                config = planned["parameters"]
                if planned["technique"] == "gradient_boosting":
                    checkpoint = train_boost(BoostTrainRequest(
                        dataset=request.dataset, parameters=config, seed=planned["trial_seed"], **groups))
                else:
                    architecture = {key: value for key, value in config.items()
                                    if key not in ("learning_rate", "weight_decay", "epochs", "batch_size")}
                    checkpoint = train_model(TrainRequest(
                        dataset=request.dataset, technique=planned["technique"], model_parameters=architecture,
                        seed=planned["trial_seed"], epochs=config.get("epochs", 10),
                        batch_size=config.get("batch_size", 2),
                        learning_rate=config.get("learning_rate", 0.001),
                        weight_decay=config.get("weight_decay", 0.00001), **groups))
                trial["checkpoint"] = input_file(ROOT, checkpoint)
                record["inputs"].append(trial["checkpoint"])
                trial["stage"] = "prediction"
                write_json(directory / "run.json", record)
                if planned["technique"] == "gradient_boosting":
                    prediction_path = predict_boost(BoostPredictRequest(
                        dataset=request.dataset, checkpoint=trial["checkpoint"]["path"],
                        group="tuning", max_samples=tuning_count))
                else:
                    prediction_path = predict_model(PredictRequest(
                        dataset=request.dataset, checkpoint=trial["checkpoint"]["path"],
                        group="tuning", max_samples=tuning_count))
                trial["prediction_run"] = input_file(ROOT, prediction_path)
                record["inputs"].append(trial["prediction_run"])
                trial["stage"] = "scoring"
                write_json(directory / "run.json", record)
                score_path = score_predictions(ScoreRequest(
                    dataset=request.dataset, prediction_run=trial["prediction_run"]["path"], group="tuning",
                    minimum_prediction_fraction=request.minimum_prediction_fraction))
                score = read_run(ROOT, score_path, kind="learning_experiments.score")
                trial["scoring_run"] = input_file(ROOT, score_path)
                record["inputs"].append(trial["scoring_run"])
                if (score.get("group") != "tuning" or score.get("metric") != "log_depth_mae"
                        or score.get("episode_partition") != partition
                        or score.get("dataset_sha256") != record["dataset_sha256"]
                        or score.get("prediction_run_sha256") != trial["prediction_run"]["sha256"]
                        or score.get("minimum_prediction_fraction") != request.minimum_prediction_fraction
                        or score.get("complete_frames") != tuning_count
                        or score.get("final_holdout_used_for_selection") is not False
                        or score.get("provenance") != record["provenance"]):
                    raise ValueError("Trial score differs from its tuning inputs or lineage")
                value = score["score_value"]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Trial has no finite nonnegative tuning score")
                trial.update(status="completed", stage="completed", score=value)
            except Exception as error:
                trial.update(status="failed", error=f"{type(error).__name__}: {error}")
                failed_run = getattr(error, "feature_run_path", None)
                if failed_run is not None:
                    trial["failed_run"] = input_file(ROOT, project_path(ROOT, failed_run))
                    record["inputs"].append(trial["failed_run"])
                write_json(directory / "run.json", record)
                if not request.continue_on_failure:
                    raise
            write_json(directory / "run.json", record)
        if any(input_file(ROOT, entry["path"]) != entry for entry in admitted_inputs):
            raise ValueError("Plan, dataset or owning producer changed before selection")
        completed = [trial for trial in record["trials"] if trial["status"] == "completed"]
        if not completed:
            raise RuntimeError("No trial produced an admitted complete tuning score; no model was selected")
        best = min(completed, key=lambda trial: (trial["score"], trial["trial_id"]))
        selection = {"schema_version": 1, "kind": "learning_experiments.selection",
                     "research_only": True, "submission_candidate": False,
                     "producer_run": (directory / "run.json").relative_to(ROOT).as_posix(),
                     "provenance": record["provenance"], "tuning_request": record["request"],
                     "plan": input_file(ROOT, plan_path), "dataset_sha256": record["dataset_sha256"],
                     "objective": plan["objective"],
                     "trial_id": best["trial_id"], "technique": best["technique"],
                     "parameters": best["parameters"], "checkpoint": best["checkpoint"],
                     "prediction_run": best["prediction_run"], "scoring_run": best["scoring_run"],
                     "metric": "log_depth_mae", "tuning_score": best["score"],
                     "episode_partition": partition, "final_holdout_used_for_selection": False,
                     "final_holdout_score": None, "planned_trials": len(plan["trials"]),
                     "completed_trials": len(completed), "failed_trials": len(record["trials"]) - len(completed),
                     "tie_break": "Lexicographic trial ID after minimum equal-episode tuning MAE"}
        write_json(directory / "selection.json", selection)
        record["selection"] = "selection.json"
        record["best_trial_id"] = best["trial_id"]
        record["completed_trials"] = len(completed)
        record["failed_trials"] = selection["failed_trials"]
    return directory / "selection.json"
