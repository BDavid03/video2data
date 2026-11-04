"""Run one recipe end to end through the owning feature APIs and return its outcome.

Teachers (MoGe2 pseudo-depth) and prepared datasets are cached in the slice
registry so a data setting is generated once. Tuning episodes 39/42 are the only
assessment group here; holdout 43/44 is reserved for finalize.
"""

from copy import deepcopy
import math
import time
import traceback

from track3.gpu import clear_stale_lock
from track3.paths import project_path
from track3.provenance import input_file, read_run
from ._common import (HOLDOUT_EPISODES, ROOT, TRAIN_EPISODES, TUNING_EPISODES, heavy_execution, now,
                      read_json, relative, save, state_dir)
from .recipes import validate_recipe, recipe_identity
from . import metrics as metrics_module

GROUPS = {"train_episodes": TRAIN_EPISODES, "tuning_episodes": TUNING_EPISODES, "holdout_episodes": HOLDOUT_EPISODES}
TEACHER_SIDE = 768


def partition():
    from track3.features.learning_experiments._common import episode_partition
    return episode_partition(TRAIN_EPISODES, TUNING_EPISODES, HOLDOUT_EPISODES)


def episode_clocks(samples_per_episode, episodes):
    videos = read_json("reports/data-audit/videos.json")
    inventory = read_json("manifests/dataset/episodes.json")
    clocks = {}
    for episode in episodes:
        matches = [row for row in inventory["episodes"] if row["split"] == "public" and row["episode_index"] == episode]
        if len(matches) != 1 or matches[0]["issues"]:
            raise ValueError(f"Episode {episode} lacks one usable public inventory entry")
        rows = [row for row in videos["videos"] if row["path"] == matches[0]["videos"]["ego_cam_a"]]
        if len(rows) != 1 or rows[0]["status"] != "ok":
            raise ValueError(f"Episode {episode} lacks a usable camera clock")
        count = int(rows[0]["decoded_frame_count"])
        clocks[episode] = {"count": count, "stride": math.ceil(count / samples_per_episode)}
    return clocks


class Registry:
    """Reusable completed teacher runs and prepared datasets, keyed by their generating settings."""

    def __init__(self):
        self.path = state_dir() / "registry.json"
        self.data = read_json(self.path) if self.path.exists() else {"teachers": {}, "datasets": {}}

    def _alive(self, path):
        try:
            record = read_run(ROOT, project_path(ROOT, path))
        except Exception:
            return False
        return record.get("status") == "completed"

    def teacher(self, episode, stride, generate):
        key = f"{episode}:{stride}:{TEACHER_SIDE}"
        path = self.data["teachers"].get(key)
        if path and self._alive(path):
            return path
        print(f"[auto_research] teacher episode {episode} stride {stride}", flush=True)
        path = relative(generate(episode, stride))
        self.data["teachers"][key] = path
        save(self.path, self.data)
        return path

    def dataset(self, teacher_runs, side, prepare):
        key = "|".join(sorted(teacher_runs)) + f"|{side}"
        entry = self.data["datasets"].get(key)
        if entry and self._alive(project_path(ROOT, entry).parent / "run.json"):
            return entry
        print(f"[auto_research] preparing dataset side {side} from {len(teacher_runs)} teacher runs", flush=True)
        entry = relative(prepare(teacher_runs, side))
        self.data["datasets"][key] = entry
        save(self.path, self.data)
        return entry


def ensure_inputs(data, registry, include_holdout=False):
    """Teachers and datasets for a data setting; holdout only when finalize asks."""
    heavy_execution("trial")
    stale = clear_stale_lock(ROOT)
    if stale:
        print(f"[auto_research] removed stale GPU lock from dead pid {stale.get('pid')}", flush=True)
    from track3.features.depth.generate import GenerateDepthRequest, generate_depth
    from track3.features.scratch_depth.prepare import PrepareDepthDataRequest, prepare_depth_data

    episodes = [*TRAIN_EPISODES, *TUNING_EPISODES, *(HOLDOUT_EPISODES if include_holdout else ())]
    clocks = episode_clocks(data["teacher_samples_per_episode"], episodes)

    def generate(episode, stride):
        return generate_depth(GenerateDepthRequest(episode_index=episode, stop_frame=None, frame_stride=stride,
                                                   max_image_side=TEACHER_SIDE))

    def prepare(runs, side):
        return prepare_depth_data(PrepareDepthDataRequest(teacher_runs=list(runs), frame_stride=1, max_image_side=side))

    teachers = {episode: registry.teacher(episode, clocks[episode]["stride"], generate) for episode in episodes}
    side = data["prepared_max_image_side"]
    development = [teachers[e] for e in (*TRAIN_EPISODES, *TUNING_EPISODES)]
    inputs = {"clocks": {str(k): v for k, v in clocks.items()}, "teachers": {str(k): v for k, v in teachers.items()},
              "dataset": registry.dataset(development, side, prepare),
              "scratch_dataset": registry.dataset([teachers[e] for e in TRAIN_EPISODES], side, prepare)}
    if include_holdout:
        inputs["holdout_dataset"] = registry.dataset([teachers[e] for e in episodes], side, prepare)
    return inputs


def _count(dataset, group):
    episodes = set(GROUPS[f"{group}_episodes"])
    return sum(row["episode_index"] in episodes for row in read_json(dataset)["samples"])


def predict_and_score(technique, checkpoint, inputs, group, dataset_key, flip_tta=False):
    """Predict on a dataset group and score it; returns (prediction_run, scoring_run, score_record)."""
    from track3.features.learning_experiments.predict import PredictRequest, predict_model
    from track3.features.learning_experiments.boosting_operations import BoostPredictRequest, predict_boost
    from track3.features.learning_experiments.score import ScoreRequest, score_predictions
    from track3.features.learning_experiments.scratch_adapter import ScratchAdapterRequest, adapt_scratch_predictions
    from track3.features.scratch_depth.predict import PredictDepthRequest, predict_depth
    from track3.features.depth_student.predict import PredictStudentRequest, predict_student

    dataset = inputs[dataset_key]
    if technique == "student":
        prediction = predict_student(PredictStudentRequest(dataset=dataset, checkpoint=checkpoint, group=group,
                                                           flip_tta=bool(flip_tta)))
    elif technique == "scratch":
        runs = []
        for episode in GROUPS[f"{group}_episodes"]:
            clock = inputs["clocks"][str(episode)]
            runs.append(relative(predict_depth(PredictDepthRequest(checkpoint=checkpoint, episode_index=episode,
                                                                   stop_frame=clock["count"], frame_stride=clock["stride"]))))
        prediction = adapt_scratch_predictions(ScratchAdapterRequest(dataset=dataset, prediction_runs=runs, group=group, **GROUPS))
    elif technique == "gradient_boosting":
        prediction = predict_boost(BoostPredictRequest(dataset=dataset, checkpoint=checkpoint, group=group,
                                                       max_samples=_count(dataset, group)))
    else:
        prediction = predict_model(PredictRequest(dataset=dataset, checkpoint=checkpoint, group=group,
                                                  max_samples=_count(dataset, group)))
    scoring = score_predictions(ScoreRequest(dataset=dataset, prediction_run=relative(prediction), group=group))
    record = read_run(ROOT, scoring, kind="learning_experiments.score")
    return relative(prediction), relative(scoring), record


def train(recipe, inputs, seed=None):
    """Train one recipe (optionally overriding the seed for replication); returns (checkpoint, record, owner)."""
    seed = recipe["seed"] if seed is None else seed
    from track3.features.learning_experiments.train import TrainRequest, train_model
    from track3.features.learning_experiments.boosting_operations import BoostTrainRequest, train_boost
    from track3.features.scratch_depth.train import TrainDepthRequest, train_depth
    from track3.features.depth_student.train import TrainStudentRequest, train_student

    technique, training = recipe["technique"], recipe["training"]
    if technique == "student":
        architecture = {k: v for k, v in recipe["parameters"].items() if k != "arch"}
        options = {k: v for k, v in training.items() if not k.startswith("loss_") and k != "flip_tta"}
        checkpoint = train_student(TrainStudentRequest(
            dataset=inputs["dataset"], arch=recipe["parameters"]["arch"], architecture=architecture,
            loss_weights={"log_l1": training["loss_log_l1"], "silog": training["loss_silog"], "gradient": training["loss_gradient"]},
            seed=seed, **options))
    elif technique == "scratch":
        checkpoint = train_depth(TrainDepthRequest(dataset=inputs["scratch_dataset"], seed=seed, **training))
    elif technique == "gradient_boosting":
        checkpoint = train_boost(BoostTrainRequest(dataset=inputs["dataset"], parameters=recipe["parameters"],
                                                   seed=seed, **GROUPS))
    else:
        checkpoint = train_model(TrainRequest(dataset=inputs["dataset"], technique=technique,
                                              model_parameters=recipe["parameters"], epochs=training["epochs"],
                                              batch_size=training["batch_size"], learning_rate=training["learning_rate"],
                                              weight_decay=training["weight_decay"], seed=seed,
                                              bf16_autocast=training["bf16_autocast"], **GROUPS))
    owner = checkpoint.parent.parent / "run.json" if checkpoint.suffix == ".pt" else checkpoint.parent / "run.json"
    if not owner.is_file():
        owner = checkpoint.parent / "run.json"
    return relative(checkpoint), read_json(owner), relative(owner)


def _replicate(recipe, inputs, seed):
    """One training + tuning assessment; returns a dict with status, metrics and run identities."""
    started = time.perf_counter()
    result = {"seed": seed, "status": "running"}
    checkpoint, training_record, training_run = train(recipe, inputs, seed=seed)
    result.update(checkpoint=input_file(ROOT, checkpoint), training_run=training_run,
                  training=metrics_module.training_summary(training_record))
    result["training"]["wall_seconds"] = time.perf_counter() - started
    assessed = time.perf_counter()
    prediction, scoring, score = predict_and_score(recipe["technique"], checkpoint, inputs, "tuning", "dataset",
                                                   flip_tta=recipe["training"].get("flip_tta", False))
    result.update(prediction_run=prediction, scoring_run=scoring, score=score["score_value"],
                  metrics=metrics_module.extended_metrics(inputs["dataset"], prediction, "tuning", score),
                  assessment_wall_seconds=time.perf_counter() - assessed, status="completed",
                  wall_seconds=time.perf_counter() - started)
    return result


def _aggregate(replicates):
    """Mean (and std) over completed replicates for every overall metric; representative = median-score replicate."""
    import statistics
    done = [r for r in replicates if r["status"] == "completed"]
    scores = sorted(done, key=lambda r: r["score"])
    representative = scores[len(scores) // 2]
    overall_keys = representative["metrics"]["overall"].keys()
    mean = {k: statistics.fmean(r["metrics"]["overall"][k] for r in done) for k in overall_keys}
    std = {k: (statistics.pstdev([r["metrics"]["overall"][k] for r in done]) if len(done) > 1 else 0.0) for k in overall_keys}
    episodes = {}
    for episode in representative["metrics"]["episodes"]:
        episodes[episode] = {k: statistics.fmean(r["metrics"]["episodes"][episode][k] for r in done)
                             for k in representative["metrics"]["episodes"][episode]}
    return representative, {"aggregation": f"mean over {len(done)} replicate seeds", "overall": mean, "overall_std": std,
                            "episodes": episodes, "replicate_scores": [r["score"] for r in done]}


def run_trial(recipe, inputs, directory):
    """Execute one recipe (1-3 seeds) on the tuning group; never touches holdout. Failures are recorded, not raised."""
    heavy_execution("trial")
    clear_stale_lock(ROOT)
    recipe = validate_recipe(recipe)
    outcome = {"recipe": recipe, "identity": recipe_identity(recipe), "status": "running", "started_utc": now(),
               "inputs": deepcopy(inputs), "stage": "training", "replicates": []}
    save(directory / "outcome.json", outcome)
    started = time.perf_counter()
    for index in range(recipe.get("replicates", 1)):
        seed = recipe["seed"] + index
        try:
            outcome["replicates"].append(_replicate(recipe, inputs, seed))
        except Exception as error:  # a failed replicate is an experiment outcome, not a loop failure
            entry = {"seed": seed, "status": "failed", "error": f"{type(error).__name__}: {error}",
                     "traceback": traceback.format_exc()[-4000:]}
            if getattr(error, "feature_run_path", None):
                entry["failed_run"] = error.feature_run_path
            outcome["replicates"].append(entry)
        save(directory / "outcome.json", outcome)
    done = [r for r in outcome["replicates"] if r["status"] == "completed"]
    if done:
        representative, metrics = _aggregate(outcome["replicates"])
        import statistics
        outcome.update(status="completed", stage="completed", checkpoint=representative["checkpoint"],
                       training_run=representative["training_run"], training=representative["training"],
                       prediction_run=representative["prediction_run"], scoring_run=representative["scoring_run"],
                       score=statistics.fmean(r["score"] for r in done),
                       score_std=statistics.pstdev([r["score"] for r in done]) if len(done) > 1 else 0.0,
                       metrics=metrics, representative_seed=representative["seed"],
                       completed_replicates=len(done), failed_replicates=len(outcome["replicates"]) - len(done))
    else:
        first = outcome["replicates"][0]
        outcome.update(status="failed", error=first.get("error"), traceback=first.get("traceback"),
                       failed_run=first.get("failed_run"))
    outcome["finished_utc"] = now()
    outcome["wall_seconds"] = time.perf_counter() - started
    save(directory / "outcome.json", outcome)
    return outcome
