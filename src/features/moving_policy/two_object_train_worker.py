"""Keep B passive during training while retaining A's existing VOC curriculum."""

import argparse


def install_passive_body(active_slot: int):
    """Clear B's virtual wrench after the existing controller, before integration."""
    import warp as wp
    import flash_chord.envs.base as base

    @wp.kernel
    def clear_resting_wrench(ids: wp.array(dtype=wp.int32), active: int,
                             forces: wp.array(dtype=wp.spatial_vector)):
        index = wp.tid()
        if index % 2 != active:
            forces[ids[index]] = wp.spatial_vector()

    original = base.apply_object_control

    def active_only(model, state, control, command, scale, params=None):
        if command.layout.num_objects != 2 or command.layout.num_articulations:
            raise ValueError('Passive B requires exactly two independent rigid bodies')
        original(model, state, control, command, scale, params)
        wp.launch(clear_resting_wrench, dim=command.voc_body_ids_w.shape[0],
                  inputs=[command.voc_body_ids_w, active_slot], outputs=[state.body_f], device=model.device)

    base.apply_object_control = active_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-slot', type=int, choices=[0, 1], required=True)
    args, arguments = parser.parse_known_args()
    install_passive_body(args.active_slot)
    import sys
    from src.features.dynamic_feasibility._worker import main as run
    sys.argv = [__file__, *arguments]
    run()


if __name__ == '__main__':
    main()
