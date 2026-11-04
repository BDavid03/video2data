"""Apply declared axial symmetry to an otherwise ambiguous initial facing."""

import numpy as np

SYMMETRIES={'beige_cup':'continuous','blue_cup':'continuous','white_pot_lid':'order 2'}


def resolve_facing(name,initialization,face_normals,face_areas,table_normal=(0.,0.,1.)):
    if name not in SYMMETRIES:
        return initialization
    normals=np.asarray(face_normals,float).copy()
    areas=np.asarray(face_areas,float)
    dominant=np.argmax(np.abs(normals),axis=1)
    normals*=np.where(normals[np.arange(len(normals)),dominant]<0,-1.,1.)[:,None]
    _,inverse=np.unique(np.round(normals,6),axis=0,return_inverse=True)
    grouped=np.bincount(inverse,weights=areas)
    chosen=inverse==int(np.argmax(grouped))
    axis=np.average(normals[chosen],weights=areas[chosen],axis=0)
    axis/=np.linalg.norm(axis)
    world=np.asarray(initialization['initial_world_T_asset'],float)[:3,:3]@axis
    table=np.asarray(table_normal,float);table/=np.linalg.norm(table)
    angle=float(np.degrees(np.arccos(np.clip(abs(world@table),-1,1))))
    result=dict(initialization,declared_symmetry=SYMMETRIES[name],original_opposite_resolved=initialization['opposite_resolved'],
        symmetry_axis_model=axis.tolist(),symmetry_axis_world=world.tolist(),table_normal_world=table.tolist(),
        symmetry_axis_table_angle_deg=angle,coherent_cap_area_fraction=float(areas[chosen].sum()/areas.sum()),
        symmetry_axis_method='largest coherent unsigned face-normal area of own procedural cylinder/revolved mesh')
    if not result['opposite_resolved'] and angle<=10.:
        result.update(opposite_resolved=True,resolved_by='declared scorer symmetry')
    return result


def mesh_axis(face_normals,face_areas):
    """Return the largest coherent unsigned procedural cap normal."""
    normals=np.asarray(face_normals,float).copy()
    dominant=np.argmax(np.abs(normals),axis=1)
    normals*=np.where(normals[np.arange(len(normals)),dominant]<0,-1.,1.)[:,None]
    _,inverse=np.unique(np.round(normals,6),axis=0,return_inverse=True)
    chosen=inverse==int(np.argmax(np.bincount(inverse,weights=face_areas)))
    axis=np.average(normals[chosen],weights=np.asarray(face_areas)[chosen],axis=0)
    return axis/np.linalg.norm(axis)
