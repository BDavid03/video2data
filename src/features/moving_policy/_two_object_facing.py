"""Compare tied facings and certify a procedural mesh's half-turn invariance."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def best_opposite(candidates,winner_world):
    rows=[c for c in candidates if Rotation.from_matrix(np.asarray(winner_world)[:3,:3].T@
          np.asarray(c['world_T_asset'])[:3,:3]).magnitude()>np.pi*.75]
    return max(rows,key=lambda row:row['iou']) if rows else None


def mesh_flip(vertices,faces,initial_world):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    world=np.asarray(initial_world,float)[:3,:3]
    turn=world.T@Rotation.from_rotvec([0.,0.,np.pi]).as_matrix()@world
    rotated=vertices@turn.T
    distance,indices=cKDTree(vertices).query(rotated)
    reverse,_=cKDTree(rotated).query(vertices)
    value=float(max(distance.max(),reverse.max()))
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    convex=bool(mesh.is_watertight and mesh.is_convex)
    same_faces=False
    if len(np.unique(indices))==len(vertices):
        same_faces=set(map(tuple,np.sort(indices[faces],axis=1)))==set(map(tuple,np.sort(faces,axis=1)))
    certified=convex or same_faces
    return dict(symmetric_vertex_hausdorff_m=value,surface_hausdorff_upper_bound_m=value if certified else None,
                threshold_m=.002,certified=certified,passed=bool(certified and value<=.002),
                certificate='watertight convex body' if convex else 'bijective mapped triangle set' if same_faces else 'unavailable',
                rotation_model=turn.tolist(),scan_reads=[])


def choose_clip(scores):
    medians=[float(np.median(np.asarray(s)[np.isfinite(s)])) if np.isfinite(s).any() else None for s in scores]
    resolved=all(m is not None for m in medians) and abs(medians[0]-medians[1])>.02
    return dict(opposite_resolved=bool(resolved),medians=medians,margin=abs(medians[0]-medians[1]) if all(m is not None for m in medians) else None,
                selected=int(np.argmax(medians)) if resolved else None,required_margin=.02,
                coverage=[float(np.isfinite(s).mean()) for s in scores])
