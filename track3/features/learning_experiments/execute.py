"""Run the bounded teacher/preparation/tuning workflow through owned APIs."""

from dataclasses import dataclass

from track3.provenance import input_file, write_json
from ._common import (FEATURE, ROOT, authorize, heavy_execution, integer,
                      operation_run, read_json)


@dataclass(frozen=True)
class ExecuteRequest:
    teacher_frames: int = 20
    teacher_max_image_side: int = 768
    prepared_max_image_side: int = 128
    frame_stride: int = 5

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        integer(self.teacher_frames, "teacher_frames", 1, 20)
        integer(self.teacher_max_image_side, "teacher_max_image_side", 64, 768)
        integer(self.prepared_max_image_side, "prepared_max_image_side", 64, 128)
        integer(self.frame_stride, "frame_stride", 1, self.teacher_frames)


def execute_tuning(request: ExecuteRequest):
    """Persist progress and child lineage; never access the final holdout."""
    authorize("execute")
    for operation in ("plan", "train", "predict", "score", "tune", "boost_train", "boost_predict"):
        authorize(operation)
    heavy_execution("execute")
    from .planning import PlanRequest, write_plan
    from .tune import TuneRequest, tune_models
    from track3.features.depth.generate import GenerateDepthRequest, generate_depth
    from track3.features.scratch_depth.prepare import PrepareDepthDataRequest, prepare_depth_data

    planned = PlanRequest.from_file(FEATURE / "plan.json")
    if planned.max_trials != 12 or set(planned.techniques) != {"ann", "cnn", "transformer", "gradient_boosting"}:
        raise ValueError("This workflow requires the selected default 12-trial, four-technique plan")
    episodes = sorted(set(planned.train_episodes) | set(planned.tuning_episodes))
    with operation_run("execute", request, inputs=[FEATURE / "plan.json"]) as (directory, record):
        record.update(stage="planning", teacher_runs=[], final_holdout_used=False)

        def retain(name, path):
            entry = input_file(ROOT, path)
            record["inputs"].append(entry)
            if path.name != "run.json":
                record["inputs"].append(input_file(ROOT, path.parent / "run.json"))
            record[name] = entry
            write_json(directory / "run.json", record)
            return entry["path"]

        plan_path = retain("plan", write_plan(planned))
        record["stage"] = "teacher_generation"
        for episode in episodes:
            print(f"Generating teacher episode {episode}: frames 0..{request.teacher_frames - 1}", flush=True)
            teacher = generate_depth(GenerateDepthRequest(
                episode_index=episode, stop_frame=request.teacher_frames,
                max_image_side=request.teacher_max_image_side))
            entry = input_file(ROOT, teacher)
            record["teacher_runs"].append(entry)
            record["inputs"].append(entry)
            write_json(directory / "run.json", record)
        record["stage"] = "preparation"
        write_json(directory / "run.json", record)
        dataset_path = retain("dataset", prepare_depth_data(PrepareDepthDataRequest(
            teacher_runs=[entry["path"] for entry in record["teacher_runs"]],
            frame_stride=request.frame_stride, max_image_side=request.prepared_max_image_side)))
        record["stage"] = "tuning"
        write_json(directory / "run.json", record)
        print("Starting the 12 sequential tuning trials", flush=True)
        selection_path = retain("selection", tune_models(TuneRequest(plan=plan_path, dataset=dataset_path)))
        record["stage"] = "completed"
        write_json(directory / "result.json", {
            "plan": record["plan"], "dataset": record["dataset"],
            "selection": record["selection"], "teacher_runs": record["teacher_runs"],
            "final_holdout_used": False, "research_only": True,
        })
        print(f"Completed tuning selection: {selection_path}", flush=True)
    return directory / "result.json"
