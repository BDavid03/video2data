"""Write and independently read back re-expressed TW mesh and URDF assets."""
from copy import deepcopy
import xml.etree.ElementTree as ET
from track3.paths import project_path
from track3.provenance import input_file
from track3.runs import ROOT, artifact, producer


def origin(element):
    """Read an optional URDF origin as translation and extrinsic XYZ rotation."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    node = element.find('origin')
    return (np.fromstring(node.get('xyz', '0 0 0') if node is not None else '0 0 0', sep=' '),
            Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0') if node is not None else '0 0 0', sep=' ')).as_matrix())


def tensor(element):
    """Read all six independent entries of a symmetric URDF inertia tensor."""
    import numpy as np
    node = element.find('inertia')
    return np.array([[float(node.get('i'+a+b if a <= b else 'i'+b+a)) for b in 'xyz'] for a in 'xyz'])


def write_origin(element, centre, rotation):
    """Serialize a URDF frame using the Rz(yaw) Ry(pitch) Rx(roll) convention."""
    from scipy.spatial.transform import Rotation
    node = element.find('origin')
    if node is None:
        node = ET.SubElement(element, 'origin')
    node.set('xyz', ' '.join(map(str, centre)))
    node.set('rpy', ' '.join(map(str, Rotation.from_matrix(rotation).as_euler('xyz'))))


def write_assets(task: dict, scene: list, directory, record: dict) -> list:
    """Preserve each admitted box asset's world geometry and inertia in its new body frame."""
    import numpy as np
    import trimesh
    from ._heading_v5 import asset_frame, body_inertia, collision_corners, equivalence, transform_vertices
    evidence = []
    for slot, row in enumerate(scene):
        mesh_path = project_path(ROOT, task['object_mesh_paths'][slot])
        urdf_path = project_path(ROOT, task['object_urdf_paths'][slot])
        parent = producer(mesh_path.parent / 'run.json', 'pose_insurance.build')
        record['inputs'].append(input_file(ROOT, parent[0]))
        artifact(parent, mesh_path.name, record)
        artifact(parent, urdf_path.name, record)
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        old_vertices = np.array(mesh.vertices, copy=True)
        tree = ET.parse(urdf_path)
        link = tree.getroot().find('link')
        original = deepcopy(link)
        visual, collision, inertial = (link.find(key) for key in ('visual', 'collision', 'inertial'))
        visual_centre, visual_rotation = origin(visual)
        if not np.array_equal(visual_centre, np.zeros(3)) or not np.array_equal(visual_rotation, np.eye(3)):
            raise ValueError('TW expects the admitted identity visual origin')
        if project_path(ROOT, visual.find('geometry/mesh').get('filename')) != mesh_path:
            raise ValueError('Task and URDF visual mesh paths differ')
        centre, inertia_rotation = origin(inertial)
        inertia = body_inertia(inertia_rotation, tensor(inertial))
        collision_centre, collision_rotation = origin(collision)
        dimensions = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
        if not np.array_equal(dimensions, row['dimensions_m']):
            raise ValueError('Collision box and admitted scene dimensions differ')
        old_wxyz = task['object_body_wxyz'][0][slot]
        new_wxyz = row['quaternion_wxyz']
        frame = asset_frame(old_wxyz, new_wxyz, inertia, centre, collision_rotation, collision_centre)
        mesh.vertices = transform_vertices(old_vertices, frame['geometry'])
        target_mesh = directory / f"{row['name']}.obj"
        target_mesh.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17, include_texture=False))
        visual.find('geometry/mesh').set('filename', str(target_mesh))
        write_origin(collision, frame['collision_centre'], frame['collision_rotation'])
        write_origin(inertial, frame['centre'], np.eye(3))
        node = inertial.find('inertia')
        for i, a in enumerate('xyz'):
            for j, b in enumerate('xyz'[i:], i):
                node.set('i'+a+b, str(frame['inertia'][i, j]))
        target_urdf = directory / f"{row['name']}.urdf"
        ET.indent(tree, space='  ')
        tree.write(target_urdf, encoding='unicode', xml_declaration=True)
        restored = ET.parse(target_urdf).getroot().find('link')
        new_centre, new_inertia_rotation = origin(restored.find('inertial'))
        new_collision_centre, new_collision_rotation = origin(restored.find('collision'))
        exported = trimesh.load(target_mesh, force='mesh', process=False)
        check = equivalence(old_wxyz, new_wxyz, old_vertices, np.asarray(exported.vertices), inertia,
                            body_inertia(new_inertia_rotation, tensor(restored.find('inertial'))),
                            centre, new_centre, collision_corners(dimensions, collision_rotation, collision_centre),
                            collision_corners(dimensions, new_collision_rotation, new_collision_centre))
        check.update(slot=slot, name=row['name'], geometry_rotation=frame['geometry'].tolist(),
                     mass_preserved=restored.find('inertial/mass').attrib == original.find('inertial/mass').attrib,
                     box_size_preserved=restored.find('collision/geometry/box').attrib == original.find('collision/geometry/box').attrib,
                     faces_preserved=bool(np.array_equal(mesh.faces, exported.faces)),
                     vertices=len(old_vertices), mass_kg=float(restored.find('inertial/mass').get('value')))
        if not check['passed'] or not all(check[k] for k in ('mass_preserved', 'box_size_preserved', 'faces_preserved')):
            raise ValueError(f'TW asset equivalence failed: {check}')
        evidence.append(check)
        task['object_mesh_paths'][slot] = str(target_mesh)
        task['object_urdf_paths'][slot] = str(target_urdf)
    return evidence
