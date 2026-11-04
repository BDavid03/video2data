"""Build one joint task from an admitted moving reference and the v5 companion."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

from src.features.pose_insurance.combined_frames import D2, OUTPUT as COMBINED, task_for
from . import handler as h
from ._two_object import joint_row
from src.features.pose_insurance import pack_moving as m

PROTOCOL = h.ROOT / 'docs/chain/moving-two-object-protocol.md'
ROSTER = h.ROOT / 'data/.runtime/kit-20261102/v2d_submission_kit/data/track_3_evaluation_objects.json'


def construct(*args, companion=None, active_name=None, names=None, **kwargs):
    """Delegate the single-object path unchanged and optionally compose a companion."""
    table, grasp, acquisition, fps = h.construct(*args, **kwargs)
    if companion is not None:
        row = joint_row(table.to_pylist()[0], active_name, names, companion)
        table = pa.Table.from_pylist([row], schema=table.schema)
    return table, grasp, acquisition, fps


MOVING_PARENT=h.ROOT/'submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json'
MOVING_SHA256='b476eb1c2bc1ee6fab2df72f42f377d87e5d265c06140f3e47fc2fc67f8ff81d'


def packed_starts(episode,names):
    """Read official-slot frame-zero achieved poses from the checked uploaded moving file."""
    parent=m.read(MOVING_PARENT)
    if parent['file']['sha256']!=MOVING_SHA256:raise ValueError('Uploaded moving-v5 baseline changed')
    packed=m.checked(parent['file'])
    export=next(e for e in parent['exports'] if e['episode']==episode)
    frame=pq.read_table(m.checked(export)).to_pandas()
    first=frame[(frame.reference_frame==0)&(frame.environment_id==0)].sort_values('object_body_id')
    if first.object_body_id.tolist()!=list(range(len(names))) or first.object_body_name.tolist()!=names:
        raise ValueError('Moving-v5 frame-zero official slot order differs')
    columns=['pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w']
    rows=pq.read_table(packed,columns=['row_id',*columns]).to_pandas().set_index('row_id')
    ids=[f'track_3/e{episode:06d}/w000/f000000/o{slot}' for slot in range(len(names))]
    poses=rows.loc[ids,columns].to_numpy(float)
    achieved=first[['achieved_'+c for c in m.POSE]].to_numpy(float)
    from src.features.pose_insurance._moving import translation_errors
    errors=translation_errors(achieved[None],poses[None],np.zeros(3))
    if errors['maximum_translation_error_m']>2e-6 or errors['maximum_orientation_error_deg']>1e-4:
        raise ValueError('Uploaded moving-v5 pose differs from its checked achieved export')
    return poses[:,[0,1,2,6,3,4,5]],dict(parent=m.entry(MOVING_PARENT),packed_file=parent['file'],export=export,
                                      frame=0,world=0,slot_order=names,quaternion_conversion='xyzw -> wxyz')


def assets(episode: int, output: Path, active: int | None = None) -> tuple[list[str], dict, list[trimesh.Trimesh]]:
    """Bind v5 starts and the already traced procedural simulation meshes."""
    roster = json.loads(ROSTER.read_text())['episodes']
    if str(episode) not in roster or len(roster[str(episode)]) != 2:
        raise PermissionError('Only the sixteen scored two-object episodes are allowed')
    names = roster[str(episode)]
    parent = json.loads(D2.read_text())
    export, = [e for e in parent['exports'] if Path(e['path']).stem == f'episode_{episode:06d}']
    original_path, original = task_for(export)
    trace = json.loads((COMBINED / 'body-frames.json').read_text())
    starts,start_binding=packed_starts(episode,names) if active is not None else (None,None)
    meshes = []
    bindings = []
    for slot, name in enumerate(names):
        entry, = [r for r in trace['rows'] if r['episode'] == episode and r['slot'] == slot]
        mesh_entry = entry['combined_mesh']
        source = h.ROOT / mesh_entry['path']
        if h.digest(source) != mesh_entry['sha256'] or entry['decision'] not in ('identity_frame', 'rotated', 'v4_procedural'):
            raise ValueError('Procedural mesh or its body-frame trace changed')
        mesh = trimesh.load(source, force='mesh', process=False)
        mesh_path = output / f'{name}.obj'
        mesh.export(mesh_path)
        urdf = output / f'{name}.urdf'
        urdf.write_text(f'<robot name="{name}"><link name="{name}">\n'
            '<inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>\n'
            f'<visual><geometry><mesh filename="{mesh_path}"/></geometry></visual>\n'
            f'<collision><geometry><mesh filename="{mesh_path}"/></geometry></collision>\n'
            '</link></robot>\n')
        start = starts[slot] if active is not None and slot!=active else np.r_[original['object_body_position'][0][slot], original['object_body_wxyz'][0][slot]]
        bindings.append(dict(name=name, pose_wxyz=start.tolist(), mesh=str(mesh_path), urdf=str(urdf),
            radius=float(np.linalg.norm(mesh.vertices, axis=1).max()), source=mesh_entry,
            mesh_sha256=h.digest(mesh_path), urdf_sha256=h.digest(urdf)))
        meshes.append(mesh)
    return names, dict(objects=bindings, v5_task=str(original_path.relative_to(h.ROOT)),
                       v5_task_sha256=h.digest(original_path), protocol_sha256=h.digest(PROTOCOL),resting_start_source=start_binding), meshes


def build(episode: int, active: int, reference_path: Path, output: Path):
    """Create both free-body references without altering the episode-38 builder."""
    output = output.resolve()
    output.relative_to(h.ROOT / 'data/outputs/moving_policy/two-object')
    output.mkdir(parents=True, exist_ok=False)
    names, binding, meshes = assets(episode, output, active=active)
    with np.load(reference_path, allow_pickle=False) as arrays:
        times = arrays['time'].copy()
        poses = np.c_[arrays['position'], arrays['quaternion_xyzw'][:, [3, 0, 1, 2]]]
        gate_path=reference_path.with_name('campaign-gates-final.json')
        if not gate_path.exists():
            gate_path=reference_path.with_name('gates-final.json')
        gates = json.loads(gate_path.read_text())
        if (gates.get('passed') is not True or gates.get('reference_sha256') != h.digest(reference_path)
                or not np.array_equal(arrays['frame_index'], np.arange(len(times)))):
            raise PermissionError('A needs an admitted reference on the complete video grid')
    shift = np.asarray(binding['objects'][active]['pose_wxyz'][:3]) - poses[0, :3]
    poses[:, :3] += shift
    h.OUTPUT = output
    h.TASK = output / f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
    template_file = next(h.TEMPLATE.glob('*.parquet'))
    parquet = pq.ParquetFile(template_file)
    template = parquet.read().to_pylist()[0]
    fk_path = h.ROOT / 'data/outputs/moving_policy/ep38-v1/template-fk.npz'
    candidates = []
    for pair in ('longitudinal', 'transverse'):
        with np.load(fk_path, allow_pickle=False) as fk:
            table, grasps, acquisition, fps = construct(template, parquet.schema_arrow, poses, times,
                meshes[active], fk, True, pair, .6, body_transform=np.eye(4))
        row = table.to_pylist()[0]
        row.update(object_body_names=[names[active]], safe_object_body_names=[names[active]],
                   object_name=names[active], safe_object_name=names[active],
                   object_mesh_paths=[binding['objects'][active]['mesh']],
                   object_urdf_paths=[binding['objects'][active]['urdf']],
                   object_mesh_radius=[binding['objects'][active]['radius']],
                   raw_motion_file='Video-only moving A and constant v5 B; template hands follow A')
        row = joint_row(row, names[active], names, binding['objects'][1-active])
        target = output / pair / h.TASK.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row], schema=parquet.schema_arrow), target)
        candidates.append(dict(pair=pair, task=str(target.relative_to(h.ROOT)), grasp=grasps,
                               acquisition_end_frame=acquisition, fps=fps, frames=len(times)))
    binding.update(episode=episode, active_slot=active, resting_slot=1-active,
                   reference=dict(path=str(reference_path.relative_to(h.ROOT)), sha256=h.digest(reference_path)),
                   translation_m=shift.tolist(), candidates=candidates,
                   status='built; Stage 0 not run; training forbidden')
    first_table=pq.ParquetFile(h.ROOT/candidates[0]['task']).read().to_pylist()[0]
    pose=np.concatenate([np.asarray(first_table['object_body_position']),np.asarray(first_table['object_body_wxyz'])[...,[1,2,3,0]]],axis=-1)
    np.savez_compressed(output/'reference-task.npz',pose=pose,frame_index=np.arange(len(times)),time=times)
    h.write(output / 'joint-build.json', binding)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--active-slot', type=int, choices=[0, 1], required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.episode, args.active_slot, args.reference.resolve(), args.out)))


if __name__ == '__main__':
    main()
