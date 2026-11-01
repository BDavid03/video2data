"""Campaign-local metadata planning; never imports or invokes a producer."""

from .selections import inspection_tier, run_role
from .request_drafts import request_draft


def _flatten(bindings):
    """Binding values in order, expanding list-valued bindings."""
    return [item for value in bindings.values() for item in (value if isinstance(value, list) else [value])]


def episode_graph(spoon, target, budget, parked_hand_run=None, submission_mesh_method="opening_hull"):
    """A topological graph with phase-specific IDs and explicit request bindings."""
    tasks = []

    def add(kind, role="shared", bindings=None, parameters=None, after=(), needs=()):
        name = kind + ":" + role
        bindings = bindings or {}
        parents = _flatten(bindings)
        tasks.append({"stage": name, "producer_kind": kind, "phase": role,
                      "bindings": bindings, "parameters": parameters or {},
                      "dependencies": list(dict.fromkeys([*parents, *after])),
                      "needs": list(needs)})
        return name

    depth = add("depth.generate", parameters={"start_frame": 0, "stop_frame": None})
    rect = add("stereo_depth.rectify")
    stereo = add("stereo_depth.generate", bindings={"depth_run": depth, "rectification_run": rect},
                 parameters={"start_frame": 0, "stop_frame": None})
    anchor = add("stereo_depth.anchor", bindings={"depth_run": depth, "stereo_run": stereo},
                 parameters={"start_frame": 0, "stop_frame": None})
    camera = add("camera_motion.estimate", "resting", bindings={"depth_run": depth, "stereo_run": stereo,
                                                               "anchor_run": anchor},
                 parameters={"opening_frame_count": 20})
    scan_mesh = add("object_mesh.resolve", "scans" if spoon else "complete", parameters={"scans_only": spoon})

    def opening_pass(camera, phase, submission=False):
        clusters = add("object_masks.opening_clusters", phase,
                       bindings={"depth_run": depth, "anchor_run": anchor, "camera_run": camera})
        inputs = {"masks_run": clusters, "depth_run": depth, "anchor_run": anchor, "camera_run": camera}
        mesh = scan_mesh
        if spoon:
            seed = add("object_pose.initialise", "scans:" + phase, bindings={**inputs, "mesh_run": scan_mesh})
            hull = add("object_mesh.reconstruct", "spoon:" + phase, bindings={**inputs, "pose_run": seed},
                       parameters={"objects": ["wooden_spoon"], "method": "opening_hull"},
                       needs=("Reviewed spoon body-frame convention and separable opening assignments",))
            mesh = add("object_mesh.resolve", "complete:" + phase, bindings={"reconstruction_run": hull},
                       parameters={"scans_only": False})
        opening = add("object_pose.initialise", "complete:" + phase, bindings={**inputs, "mesh_run": mesh})
        if submission:
            # Every submitted object needs reconstructed geometry, including
            # scan-backed simulation objects in episodes containing a spoon.
            add("object_mesh.reconstruct", "submission:" + phase,
                bindings={**inputs, "pose_run": opening},
                parameters={"objects": [], "method": submission_mesh_method})
        return clusters, mesh, opening

    clusters, mesh, opening = opening_pass(camera, "resting", submission=target == "resting")

    def policy(scene, role, fallback=None):
        export = add("retargeting.export", role, bindings={"scene_run": scene})
        previous = export
        for operation in ("assets", "retarget", "support", "inspect"):
            previous = add("retargeting." + operation, role, bindings={"input_run": previous})
        settings = {"policy_kind": "resting" if role == "resting" else "trained",
                    "budget": "zero_update" if role == "resting" else budget}
        config = add("policy_training.compose", role, bindings={"retargeting_run": previous}, parameters=settings)
        training = add("policy_training.train", role, bindings={"compose_run": config}, parameters=settings)
        rollout = add("policy_rollout.rollout", role,
                      bindings={"training_run": training, **({"resting_inspection": fallback} if fallback else {})},
                      needs=("Explicit checkpoint path owned by the selected training run",))
        inspection = add("policy_rollout.inspect", role, bindings={"rollout_run": rollout})
        selection = add("policy_training.select", role, parameters=settings,
                        bindings={"training_run": training, "rollout_inspections": [inspection]})
        if role == "tracked":
            score = add("policy_rollout.self_score", role, bindings={"inspection_run": inspection, "scene_run": scene})
            resting_score = add("policy_rollout.self_score", "resting",
                                bindings={"inspection_run": fallback, "scene_run": scene})
        else:
            score = selection
            resting_score = selection
        add("campaign.promote", role, bindings={"run": inspection}, after=(selection, score, resting_score),
            parameters={"tier": role}, needs=("Explicit reason for selecting this inspected rollout",))
        return inspection

    world = add("object_pose.compose", "resting", bindings={"pose_run": opening, "camera_run": camera})
    resting = add("scene_assembly.assemble", "resting", parameters={"mode": "resting", "parked_hand_run": parked_hand_run},
                  bindings={"pose_run": world, "camera_run": camera, "mesh_run": mesh},
                  needs=("Sourced parked-hand template", "Explicit per-object masses and support choices"))
    fallback = policy(resting, "resting")
    if target == "resting":
        return tasks
    masks = add("object_masks.generate", bindings={"depth_run": depth, "opening_run": opening,
                                                   "opening_masks_run": clusters},
                needs=("Approved pinned object-mask backend and model assets",))
    camera = add("camera_motion.estimate", "tracked", bindings={"depth_run": depth, "stereo_run": stereo,
                                                               "anchor_run": anchor, "masks_run": masks},
                 parameters={"opening_frame_count": None})
    _, mesh, opening = opening_pass(camera, "tracked", submission=True)
    estimate = add("hand_motion.estimate", bindings={"masks_run": masks, "depth_run": depth},
                   needs=("Approved pinned WiLoR backend and licensed model assets",))
    kinematics = add("hand_motion.kinematics", bindings={"estimate_run": estimate, "masks_run": masks,
                                                        "depth_run": depth},
                     needs=("Approved pinned MANO backend and licensed left/right assets",))
    metric = {"masks_run": masks, "depth_run": depth, "stereo_run": stereo,
              "anchor_run": anchor, "camera_run": camera}
    hands = add("hand_motion.lift", bindings={**metric, "kinematics_run": kinematics})
    tracked = add("object_pose.track", bindings={**metric, "pose_run": opening, "mesh_run": mesh})
    world = add("object_pose.compose", "tracked", bindings={"pose_run": tracked, "camera_run": camera,
                                                           "hand_run": hands})
    scene = add("scene_assembly.assemble", "tracked", parameters={"mode": "full", "parked_hand_run": parked_hand_run},
                bindings={"pose_run": world, "camera_run": camera, "mesh_run": mesh, "hand_run": hands},
                needs=("Sourced parked-hand template", "Explicit per-object masses and support choices"))
    policy(scene, "tracked", fallback)
    return tasks


def environment(kind):
    if kind == "object_masks.generate":
        return ".runtime/object-masks"
    if kind == "hand_motion.estimate":
        return ".runtime/wilor"
    if kind in ("hand_motion.kinematics", "hand_motion.parked_template"):
        return ".runtime/mano"
    if kind.startswith(("retargeting.", "policy_")):
        return "linux-policy"
    if kind.startswith("campaign."):
        return "standard-library"
    return "windows-perception"


GPU_OPERATIONS = {"depth.generate", "object_masks.generate", "hand_motion.estimate",
                  "policy_training.train", "policy_rollout.rollout"}


def plan_episode(graph, split, episode, state, records, selections, pins, healthy, digest, root=None):
    """Only explicit, current pins satisfy tasks. Candidates remain suggestions."""
    resolved, pending = {}, []
    for task in graph:
        stage, kind, role = task["stage"], task["producer_kind"], task["phase"]
        dependencies = task["dependencies"]
        waiting = [name for name in dependencies if name not in resolved]
        bindings = {field: ([resolved.get(name) for name in target] if isinstance(target, list)
                            else resolved.get(target)) for field, target in task["bindings"].items()}
        # Check the producer contract even when a candidate will change the
        # eventual action to campaign.pin; unknown planner keys never disappear.
        draft = request_draft("campaign.pin" if kind == "campaign.promote" else kind,
                              split, episode, task["parameters"], bindings, root=root)
        pin = selections.get((split, episode, kind, role))
        candidate = None

        def eligible(path):
            record = records[path]
            if (waiting or record["kind"] != kind or run_role(record, records) != role or
                    record.get("split") != split or record.get("episode_index") != episode or
                    record["status"] != "completed" or record.get("blocking", False) or
                    record["provenance"] != "video_only" or not healthy(path)):
                return False
            if any(record["request"].get(key) != value for key, value in task["parameters"].items()
                   if key != "parked_hand_run"):
                return False
            parents = _flatten(bindings)
            if kind == "scene_assembly.assemble":
                template = task["parameters"].get("parked_hand_run")
                if template is None:
                    return False
                parents.append(template)
            # Producer requests may store absolute paths; canonical input hashes
            # establish binding identity without guessing from their spelling.
            return all({"path": parent, "sha256": digest(parent)} in record["inputs"] for parent in parents)

        if kind == "campaign.promote":
            selected = pins.get((split, episode, "policy_rollout.inspect"))
            if (not waiting and selected and selected["run"] == bindings["run"] and selected.get("tier") == role
                    and selected["sha256"] == digest(selected["run"])):
                resolved[stage] = selected["run"]
                continue
            # An already selected tracked entry must not be demoted just to
            # re-declare its historical resting fallback.
            if role == "resting" and state == "tracked" and not waiting:
                resolved[stage] = bindings["run"]
                continue
            action = "campaign.pin"
            draft = request_draft("campaign.pin", split, episode,
                                  {"tier": role, "artifact_name": "policy_rollout.inspect", "reason": None},
                                  bindings, root=root)
            candidate = bindings["run"]
        else:
            if pin and pin["run"] in records and pin["sha256"] == digest(pin["run"]) and eligible(pin["run"]):
                if kind != "policy_rollout.inspect" or inspection_tier(records[pin["run"]]) == role:
                    resolved[stage] = pin["run"]
                    continue
            candidates = [path for path in records if eligible(path) and
                          (kind != "policy_rollout.inspect" or inspection_tier(records[path]) == role)]
            candidate = max(candidates, key=lambda path: (records[path].get("run_id", ""), path), default=None)
            action = "campaign.pin" if candidate else kind
            if candidate:
                draft = request_draft(action, split, episode,
                                      {"run": candidate, "artifact_name": kind, "reason": None, "tier": None},
                                      root=root)
        pending.append({"split": split, "episode_index": episode, "state": state,
                        "stage": stage, "producer_kind": kind, "phase": role, "operation": action,
                        **draft, "request_bindings": task["bindings"],
                        "planned_parameters": dict(task["parameters"]),
                        "depends_on": waiting, "inputs_selected": not waiting,
                        "ready": False, "execution_authorized": False,
                        "requirements": (["Explicit reason for pinning this candidate"] if candidate and
                                         kind != "campaign.promote" else task["needs"]),
                        "candidate_run": candidate, "environment": environment(action),
                        "gpu": action in GPU_OPERATIONS,
                        "budget": task["parameters"].get("budget")})
    return pending
