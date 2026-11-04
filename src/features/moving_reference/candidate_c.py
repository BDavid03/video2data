"""Orchestrate candidate C in ordered, label-free stages without rewriting A."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from src.provenance import sha256_file,write_json
from . import support

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'data/outputs/moving_reference'


def identity(selected):
    paths=[Path(__file__),Path(__file__).with_name('model_track.py'),Path(__file__).with_name('camera_ba.py'),
           Path(__file__).with_name('camera.py'),Path(__file__).with_name('_object.py'),
           Path(__file__).with_name('support.py'),Path(__file__).with_name('gates.py')]
    return dict(inputs=selected['hashes'],sources={str(p.relative_to(ROOT)):sha256_file(p) for p in paths if p.exists()},
                label_access=False,gpu=False,thresholds_unchanged=True)


def body_constant(selected,directory):
    import trimesh
    from ._object import body_axes
    old=OUTPUT/'ep39/body-convention.json'
    previous=json.loads(old.read_text())
    scan=ROOT/previous['scan_input']['path']
    mesh=trimesh.load(scan,force='mesh',process=False)
    bounds,extents=trimesh.bounds.oriented_bounds(mesh)
    points,_=trimesh.sample.sample_surface(mesh,10000,seed=31)
    transform,details=body_axes(bounds,extents,points)
    result=dict(scan_T_asset=transform.tolist(),asset_T_scan=np.linalg.inv(transform).tolist(),
                method=previous['method'],scan_input=dict(path=str(scan.relative_to(ROOT)),sha256=sha256_file(scan)),
                details=details,previous_scan_T_asset=previous['scan_T_asset'],
                previous_constant_sha256=sha256_file(old),applies_identically_to_episodes=[39,38],trajectory_used=False)
    write_json(directory/'body-convention.json',result)
    return result


def support_run(directory,pose,mesh):
    target=directory/'support';target.mkdir(exist_ok=True)
    task,request=support.prepare(target,pose,mesh)
    urdf=target/'mini_dust_pan.urdf'
    text=urdf.read_text();start=text.index('<inertial>');end=text.index('</inertial>')+len('</inertial>')
    text=text[:start]+'<inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>'+text[end:]
    urdf.write_text(text)
    (directory/'asset/mini_dust_pan.urdf').write_text(text)
    request.update(mass_kg=.3,inertia_diagonal_kg_m2=[.001]*3,
                   cylinder_rule='S2: top 1 mm below this task own imported CoACD hulls; footprint-covering radius',
                   urdf_sha256=sha256_file(urdf))
    env={**os.environ,'PYTHONPATH':os.pathsep.join((str(ROOT),str(support.FC/'src'))),
         'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PXR_WORK_THREAD_LIMIT':'1'}
    for key in ('WARP_CACHE_PATH','FLASH_CHORD_CACHE_DIR','JAX_COMPILATION_CACHE_DIR','XDG_CACHE_HOME','MPLCONFIGDIR','TMPDIR'):
        cache=target/'cache'/key.lower();cache.mkdir(parents=True,exist_ok=True);env[key]=str(cache)
    command=[str(support.PYTHON),'-B','-m','src.features.moving_reference.candidate_c','support-worker','--directory',str(directory),'--task',str(task)]
    write_json(target/'request.json',dict(**request,argv=command))
    with (target/'worker.log').open('w') as log:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    request.update(sources={str(p.relative_to(ROOT)):sha256_file(p) for p in
                            (Path(__file__),Path(__file__).with_name('support.py'))},
                   support_sha256=sha256_file(target/'reconstructed_stage/video_reference_sharpa_wave_support.usda'),
                   worker_exit_code=process.returncode)
    write_json(target/'request.json',dict(**request,argv=command))
    path=target/'support.json'
    if not path.exists():
        result=dict(status='unmeasured',maximum_translation_m=None,maximum_rotation_deg=None,exit_code=process.returncode)
        write_json(path,result)
    return json.loads(path.read_text())


def support_worker(directory,task):
    import warp as wp
    wp.config.enable_cuda=False;wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH'];wp.set_device('cpu')
    import newton
    from flash_chord.data.loader import load_reference
    from flash_chord.scene.objects import spawn_objects
    from scipy.spatial.transform import Rotation
    from src.features.moving_policy.geometry import cylinder_support
    reference=load_reference(str(task),control_fps=20.,motion_speed=1.)
    builder=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    spawn_objects(builder,reference,decompose_collision=True)
    pieces=[]
    for i,flags in enumerate(builder.shape_flags):
        if int(flags)&int(newton.ShapeFlags.COLLIDE_SHAPES):
            vertices=np.asarray(builder.shape_source[i].vertices,float)*np.asarray(builder.shape_scale[i])
            transform=np.asarray(builder.shape_transform[i])
            pieces.append(Rotation.from_quat(transform[3:]).apply(vertices)+transform[:3])
    with np.load(directory/'asset/placement.npz') as arrays:
        pose=arrays['world_T_asset']
    center,radius,top=cylinder_support(pieces,pose[:3,3],pose[:3,:3])
    target=directory/'support'
    np.savez_compressed(target/'own-imported-hulls.npz',**{f'vertices_{i}':p for i,p in enumerate(pieces)})
    fixture=target/'reconstructed_stage/video_reference_sharpa_wave_support.usda'
    fixture.write_text(f'''#usda 1.0
(defaultPrim = "support_surfaces"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "support_surfaces" {{
 def Cylinder "table" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
  uniform token axis = "Z"
  double height = 0.01
  double radius = {radius}
  double3 xformOp:translate = ({center[0]}, {center[1]}, {top-.005})
  uniform token[] xformOpOrder = ["xformOp:translate"]
 }}
}}
''')
    write_json(target/'s2-fixture.json',dict(top_m=top,center_xy_m=center,radius_m=radius,
               hull_count=len(pieces),hull_gap_m=.001,mass_kg=.3,inertia_diagonal_kg_m2=[.001]*3,
               fixture_sha256=sha256_file(fixture),source='Own imported Newton/CoACD collision solids; no M2 pose or hull reused'))
    support.worker(task,target)


def prepare_frame0(selected,directory):
    import cv2
    from . import camera,track
    calibration=json.loads(selected['calibration'].read_text())
    maps,k,q,r1=camera.rectification(calibration)
    streams=track.captures(selected['paths'])
    try:
        images,gray=track.read_pair(streams,maps)
    finally:
        for stream in streams:
            stream.release()
    xyz,valid=camera.stereo(*gray,q)
    obj,hands,visible,static=track.masks_at(selected,0,maps,gray[0].shape)
    world,table=camera.define_world(xyz,static&valid,obj)
    np.savez_compressed(directory/'frame0.npz',image=images[0],gray=gray[0],xyz=xyz,depth_valid=valid,
                        object_mask=obj,hands=hands,visible=visible,static=static,k=k,q=q,r1=r1,
                        world_T_camera=world)
    write_json(directory/'table-initial.json',table)


def build_asset():
    import cv2
    import trimesh
    from . import camera,track
    from ._object import iou,render
    from .model_track import extrude_mask,footprint
    started=time.perf_counter();cv2.setNumThreads(2)
    directory=OUTPUT/'ep39/candidate-c';assets=directory/'asset';assets.mkdir(parents=True,exist_ok=True)
    if (assets/'parameters.json').exists():
        parameters=json.loads((assets/'parameters.json').read_text())
        previous=directory/'support/support.json'
        if previous.exists() and json.loads(previous.read_text())['status']=='measured':
            print('Asset and support already measured; preserving them');return
        mesh=trimesh.load(directory/'canonical-asset.obj',force='mesh',process=False)
        with np.load(assets/'placement.npz') as arrays:
            pose=arrays['world_T_asset']
        log=directory/'support/worker.log'
        if log.exists():
            (directory/'support/worker-import-failure.log').write_bytes(log.read_bytes())
        result=support_run(directory,pose,mesh)
        placement_status(directory,parameters,result)
        return
    selected=track.inputs(39)
    write_json(directory/'request-asset.json',identity(selected))
    if not (directory/'frame0.npz').exists():
        prepare_frame0(selected,directory)
    with np.load(directory/'frame0.npz') as arrays:
        frame={key:arrays[key] for key in arrays.files}
    pose,parameters=footprint(frame['object_mask'],frame['xyz'],frame['visible'],frame['k'],frame['world_T_camera'])
    from src.features.mesh_extents._shapes import thin_tool_profile
    profile=thin_tool_profile(parameters['dimensions_m'],parameters['profile'])
    profile_pose=pose.copy();profile_pose[2,3]=-profile.vertices[:,2].min()
    outline=extrude_mask(frame['object_mask'],frame['k'],frame['world_T_camera'],pose,parameters['thickness_m'])
    old=track.asset(selected);old_pose=pose.copy();old_pose[2,3]=-old.vertices[:,2].min()
    camera_T_world=np.linalg.inv(frame['world_T_camera'])
    candidates=[]
    for name,mesh,placement in [('old-flat',old,old_pose),('profile-refit',profile,profile_pose),('outline-extrusion',outline,pose)]:
        prediction=render(mesh.vertices,mesh.faces,camera_T_world@placement,frame['k'],frame['object_mask'].shape)
        score=iou(prediction,frame['object_mask'],frame['hands'])
        mesh.export(assets/f'{name}.obj')
        candidates.append((name,score,mesh,placement,prediction))
    winner=max(candidates[1:],key=lambda row:row[1])
    name,score,mesh,pose,prediction=winner
    mesh.export(directory/'canonical-asset.obj');mesh.export(assets/'mini_dust_pan.obj')
    np.savez_compressed(assets/'placement.npz',world_T_asset=pose)
    body=body_constant(selected,assets)
    points=camera.apply(pose,mesh.vertices)
    parameters.update(selected=name,frame0_iou_before=candidates[0][1],frame0_iou_after=score,
                      candidates={row[0]:row[1] for row in candidates},watertight=bool(mesh.is_watertight),
                      vertices=len(mesh.vertices),faces=len(mesh.faces),world_T_asset=pose.tolist(),
                      world_T_scan=(pose@np.linalg.inv(np.asarray(body['scan_T_asset']))).tolist(),
                      frame0_base_gap_m=max(0.,float(points[:,2].min())),frame0_penetration_m=max(0.,-float(points[:,2].min())),
                      placement_defect='A optimized height +/-6 cm to shrink an oversized projection. C fixes the mesh base to table z=0 and derives metric footprint from calibrated rays.',
                      wall_seconds=time.perf_counter()-started)
    write_json(assets/'parameters.json',parameters)
    view=frame['image'].copy()
    for mask,color in [(frame['object_mask'],(0,220,0)),(prediction,(0,80,255))]:
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view,contours,-1,color,1)
    cv2.imwrite(str(assets/'frame0-overlay.png'),view)
    result=support_run(directory,pose,mesh)
    placement_status(directory,parameters,result)
    print(json.dumps(dict(asset=name,iou=score,support=result['status'])),flush=True)


def placement_status(directory,parameters,result):
    gate=lambda value,limit:dict(value=value,threshold=limit,passed=value is not None and value<=limit)
    gates=dict(frame0_base_gap_m=gate(parameters['frame0_base_gap_m'],.005),
               frame0_penetration_m=gate(parameters['frame0_penetration_m'],.005),
               zero_action_support_translation_m=gate(result['maximum_translation_m'],.01),
               zero_action_support_rotation_deg=gate(result['maximum_rotation_deg'],2))
    write_json(directory/'placement-gates.json',dict(gates=gates,passed=all(row['passed'] for row in gates.values())))
    print(json.dumps(dict(placement_gates=gates)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['asset','support-worker','camera'])
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--task',type=Path)
    args=parser.parse_args()
    if args.operation=='asset':
        build_asset()
    elif args.operation=='support-worker':
        support_worker(args.directory,args.task)
    else:
        from .camera_ba import run
        run()
