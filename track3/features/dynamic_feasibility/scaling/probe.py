"""Measure simulator throughput and resident learner memory without tracking scores."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from track3.features.dynamic_feasibility import handler as d0
    from track3.features.dynamic_feasibility.scaling import handler
    from track3.provenance import write_json
    handler.environment("probe_authorized")
    queue = json.loads(handler.QUEUE.read_text())["D1"]
    base = json.loads((d0.ROOT / queue["base_queue"]).read_text())
    task = d0.load_task(base)
    budget = dict(base["L1"], world_count=args.worlds)
    budget["total_environment_steps"] = budget["total_environment_steps"] // args.worlds * args.worlds
    request = SimpleNamespace(seed=0, budget=budget)
    overrides = [value for value in d0.training_command(request, task, args.out.parent)[5:]
                 if not value.startswith(("output_dir=", "hydra.run.dir="))]
    import jax
    import jax.numpy as jnp
    import warp as wp
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.learner import create_learner
    from flash_chord.training.flash_sac.runner import resolve_jax_device

    cfg = compose_config("train_flash_sac", overrides)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    training = instantiate_typed(cfg.training, TrainingConfig)
    with wp.ScopedDevice(training.device):
        setup = setup_scene(parquet=cfg.task.parquet, control_fps=env_config.sim.fps,
                            motion_speed=cfg.task.motion_speed,
                            embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                            collision=instantiate_typed(cfg.collision, CollisionPolicy),
                            world_count=cfg.scene.world_count, include_support=cfg.scene.support,
                            decompose_objects=cfg.scene.decompose_objects,
                            object_scale_min=cfg.scene.object_scale_min,
                            object_scale_max=cfg.scene.object_scale_max,
                            object_scale_seed=cfg.scene.object_scale_seed,
                            contact_friction=cfg.scene.contact_friction,
                            object_free_joint_damping=cfg.scene.object_free_joint_damping,
                            source_frame_playback=False, motion_start_frame=0, motion_end_frame=-1)
        env = RLEnv(setup.scene, setup.reference, config=env_config)
        env.reset()
        env.capture_step()
        env.reset()
        view = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        view.set_curriculum(training.curriculum.stages[0])
        device = resolve_jax_device(training.device)
        observation = jax.device_put(jnp.array(view.reset(), copy=True), device)
        context = jax.device_put(jnp.array(view.initial_critic_context, copy=True), device)
        learner, executables = create_learner(training, observation, context,
                                            action_dim=view.action_dim,
                                            objective_term_count=len(view.objective_term_names),
                                            termination_cause_count=len(view.termination_cause_names),
                                            tracking_error_count=len(view.tracking_error_names),
                                            critic_context_names=view.critic_context_names, device=device)
        jax.block_until_ready(learner)
        action = wp.zeros(args.worlds * view.action_dim, dtype=wp.float32, device=env.device)
        for _ in range(queue["probe"]["warmup_collections"]):
            env.step(action)
        wp.synchronize_device(env.device)
        start = time.perf_counter()
        for _ in range(queue["probe"]["timed_collections"]):
            env.step(action)
        wp.synchronize_device(env.device)
        seconds = time.perf_counter() - start
        result = {"world_count": args.worlds,
                  "environment_steps": args.worlds * queue["probe"]["timed_collections"],
                  "timed_seconds": seconds,
                  "steps_per_second": args.worlds * queue["probe"]["timed_collections"] / seconds,
                  "optimizer_updates": int(jax.device_get(learner.global_update_step)),
                  "tracking_error_values_read": False, "learner_state_resident": True}
        if result["optimizer_updates"] != 0:
            raise ValueError("Sizing probe unexpectedly learned")
        write_json(args.out, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
