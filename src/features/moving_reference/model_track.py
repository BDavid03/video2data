"""Geometry for a table-supported, silhouette-derived dust-pan asset."""

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

from .camera import apply


def plane_pixels(pixels, k, world_T_camera, height):
    rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(k).T
    rays=rays@world_T_camera[:3,:3].T
    origin=world_T_camera[:3,3]
    scale=(height-origin[2])/rays[:,2]
    return origin+rays*scale[:,None]


def footprint(mask, xyz, visible, k, world_T_camera, known_thickness=None):
    points=apply(world_T_camera,xyz[visible & np.isfinite(xyz).all(2)])
    heights=np.percentile(points[:,2],[10,25,50,75,90])
    thickness=float(heights[1]) if known_thickness is None else float(known_thickness)
    if thickness<=0:
        raise ValueError('Stereo lower-quartile floor height is nonpositive: thickness is unresolved')
    contours,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    outside=max((i for i,h in enumerate(hierarchy[0]) if h[3]<0),key=lambda i:cv2.contourArea(contours[i]))
    holes=[contours[i] for i,h in enumerate(hierarchy[0]) if h[3]==outside and cv2.contourArea(contours[i])>=4]
    if not holes:
        raise ValueError('No handle hole in the frame-0 mask; longitudinal sign is unresolved')
    hole=max(holes,key=cv2.contourArea)
    moments=cv2.moments(hole)
    ring_uv=np.array([moments['m10']/moments['m00'],moments['m01']/moments['m00']])
    yy,xx=np.nonzero(mask)
    cloud=plane_pixels(np.c_[xx,yy],k,world_T_camera,thickness)
    ring=plane_pixels(ring_uv[None],k,world_T_camera,thickness)[0]
    direction=ring[:2]-cloud[:,:2].mean(0);direction/=np.linalg.norm(direction)
    yaw=float(np.arctan2(direction[1],direction[0]))
    rotation=Rotation.from_euler('z',yaw).as_matrix()
    local=cloud@rotation
    center_local=(local.min(0)+local.max(0))/2
    local-=center_local
    center=rotation@center_local
    center[2]=thickness/2
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=center
    widths=[];edges=np.linspace(local[:,0].min(),local[:,0].max(),33)
    for lo,hi in zip(edges[:-1],edges[1:]):
        part=local[(local[:,0]>=lo)&(local[:,0]<=hi)]
        widths.append(float(np.ptp(part[:,1])) if len(part)>1 else 0.)
    widths=np.asarray(widths)
    neck=int(np.argmin(widths[12:26])+12)
    dims=np.ptp(local,axis=0);dims[2]=float(heights[3])
    head=(neck+.5)/32
    profile=dict(kind='scoop',axis='x',head_fraction=head,neck_width_fraction=float(widths[neck]/dims[1]),
                 grip_width_fraction=float(widths[26:].max()/dims[1]),grip_start_fraction=max(.4,head-.05),
                 floor_thickness_fraction=min(1.,thickness/dims[2]),grip_thickness_fraction=min(1.,thickness/dims[2]),
                 wall_width_fraction=min(.1,thickness/min(dims[:2])))
    ring_local=apply(np.linalg.inv(pose),plane_pixels(hole[:,0],k,world_T_camera,thickness))
    return pose,dict(thickness_m=thickness,thickness_rule='25th percentile of eroded stereo points above the frame-0 table; a floor-height proxy, not an independently observed underside',
                     stereo_height_percentiles_10_25_50_75_90_m=heights.tolist(),dimensions_m=dims.tolist(),
                     profile=profile,width_profile_m=widths.tolist(),
                     handle_inner_diameters_m=np.ptp(ring_local[:,:2],axis=0).tolist(),
                     handle_center_world_m=ring.tolist(),heading_rule='Mask hole center minus planar silhouette centroid; +x toward handle')


def extrude_mask(mask,k,world_T_camera,world_T_asset,thickness,stride=2):
    yy,xx=np.nonzero(mask)
    x0,x1=max(0,int(xx.min())-2),min(mask.shape[1],int(xx.max())+3)
    y0,y1=max(0,int(yy.min())-2),min(mask.shape[0],int(yy.max())+3)
    small=mask[y0:y1:stride,x0:x1:stride]
    y,x=np.nonzero(small)
    width=small.shape[1]+1
    quads=np.c_[y*width+x,y*width+x+1,(y+1)*width+x+1,(y+1)*width+x]
    ids,inverse=np.unique(quads,return_inverse=True)
    quads=inverse.reshape(-1,4)
    pixels=np.c_[x0+(ids%width)*stride-stride/2,y0+(ids//width)*stride-stride/2]
    top=plane_pixels(pixels,k,world_T_camera,thickness)
    vertices=apply(np.linalg.inv(world_T_asset),top)
    bottom=vertices.copy();bottom[:,2]-=thickness
    count=len(vertices);vertices=np.vstack([bottom,vertices])
    faces=[]
    for a,b,c,d in quads:
        faces.extend([(a,c,b),(a,d,c),(a+count,b+count,c+count),(a+count,c+count,d+count)])
    for (row,col),quad in zip(zip(y,x),quads):
        for dy,dx,first,second in ((-1,0,0,1),(0,1,1,2),(1,0,2,3),(0,-1,3,0)):
            ny,nx=row+dy,col+dx
            if not (0<=ny<small.shape[0] and 0<=nx<small.shape[1] and small[ny,nx]):
                a,b=quad[first],quad[second]
                faces.extend([(a,b,b+count),(a,b+count,a+count)])
    mesh=trimesh.Trimesh(vertices,np.asarray(faces),process=True)
    mesh.fix_normals(multibody=False)
    return mesh


def tracking_geometry(mask,k,world_T_camera,world_T_asset,thickness,vertices,faces):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    loops=[]
    for contour in contours:
        if cv2.contourArea(contour)<4:
            continue
        contour=cv2.approxPolyDP(contour,.65,True)[:,0]
        points=apply(np.linalg.inv(world_T_asset),plane_pixels(contour,k,world_T_camera,thickness))
        points[:,2]=thickness/2
        loops.append(points)
    dense=[]
    for loop in loops:
        for a,b in zip(loop,np.roll(loop,-1,axis=0)):
            dense.extend(a+(b-a)*np.linspace(0,1,max(2,int(np.linalg.norm(b-a)/.002)))[:,None])
    outline=np.asarray(dense);bottom=outline.copy();bottom[:,2]-=thickness
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    samples,_=trimesh.sample.sample_surface(mesh,5000,seed=31)
    return dict(loops=loops,outline=np.vstack([outline,bottom]),surface_tree=cKDTree(samples),thickness=thickness)


def fast_render(geometry,camera_T_asset,k,shape):
    from .camera import project
    layers=[]
    for offset in (0.,-geometry['thickness']):
        polygons=[]
        for loop in geometry['loops']:
            points=loop.copy();points[:,2]+=offset
            points=apply(camera_T_asset,points)
            polygons.append(np.rint(np.clip(project(points,k),-10000,10000)).astype(np.int32))
        mask=np.zeros(shape,np.uint8)
        cv2.fillPoly(mask,polygons,1)
        layers.append(mask)
    return (layers[0]|layers[1]).astype(bool)


def image_evidence(mask,hands,xyz,visible):
    edge=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)&~hands
    yy,xx=np.nonzero(edge)
    pixels=np.c_[xx,yy].astype(float)
    if len(pixels)>180:
        pixels=pixels[np.linspace(0,len(pixels)-1,180).astype(int)]
    distance=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,3)
    cloud=xyz[visible & np.isfinite(xyz).all(2)]
    count=len(cloud)
    if len(cloud)>160:
        cloud=cloud[np.linspace(0,len(cloud)-1,160).astype(int)]
    return dict(mask=mask,hands=hands,distance=distance,pixels=pixels,cloud=cloud,stereo_pixels=count)


def fit_pose(base,evidence,geometry,k,maximum_evaluations=22):
    from .camera import project
    points=evidence['cloud'];outline=geometry['outline'];target=evidence['pixels']
    factor=k[0,0]/max(float(np.median(points[:,2])) if len(points) else 1.,.1)
    def pose_at(values):
        pose=base.copy();pose[:3,:3]=Rotation.from_rotvec(values[:3]).as_matrix()@base[:3,:3]
        pose[:3,3]=base[:3,3]+values[3:]
        return pose
    def residual(values):
        pose=pose_at(values);seen=apply(pose,outline);uv=project(seen,k)
        uv=np.nan_to_num(uv,nan=-100,posinf=10000,neginf=-10000)
        x=uv[:,0].astype(np.float32).reshape(-1,1);y=uv[:,1].astype(np.float32).reshape(-1,1)
        first=cv2.remap(evidence['distance'],x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=40).ravel()
        hand=cv2.remap(evidence['hands'].astype(np.uint8),x,y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).ravel()
        first*=1-hand
        second=cKDTree(uv).query(target)[0] if len(target) else np.zeros(0)
        local=apply(np.linalg.inv(pose),points)
        surface=geometry['surface_tree'].query(local)[0]*factor if len(points) else np.zeros(0)
        return np.r_[first,second,surface]*640/960
    steps=np.array([.002,.002,.002,.0005,.0005,.0005])
    def jacobian(values):
        initial=residual(values)
        return np.column_stack([(residual(values+np.eye(6)[i]*steps[i])-initial)/steps[i] for i in range(6)])
    result=least_squares(residual,np.zeros(6),jac=jacobian,loss='huber',f_scale=2.,
                         x_scale=np.array([.1,.1,.1,.02,.02,.02]),max_nfev=maximum_evaluations,
                         ftol=1e-3,xtol=1e-3,gtol=1e-3)
    pose=pose_at(result.x)
    return pose,float(np.mean(np.minimum(residual(result.x)**2,100)))


def select_pose(starts,evidence,geometry,k):
    from ._object import iou
    candidates=[]
    for base in starts:
        pose,cost=fit_pose(base,evidence,geometry,k)
        score=iou(fast_render(geometry,pose,k,evidence['mask'].shape),evidence['mask'],evidence['hands'])
        candidates.append((score,cost,pose))
    best=max(candidates,key=lambda row:row[0])
    weak=best[0]<.60
    if weak:
        for axis in ('x','z'):
            alternate=best[2].copy();alternate[:3,:3]=alternate[:3,:3]@Rotation.from_euler(axis,np.pi).as_matrix()
            pose,cost=fit_pose(alternate,evidence,geometry,k)
            score=iou(fast_render(geometry,pose,k,evidence['mask'].shape),evidence['mask'],evidence['hands'])
            candidates.append((score,cost,pose))
        best=max(candidates,key=lambda row:row[0])
    opposite=[row for row in candidates if Rotation.from_matrix(best[2][:3,:3].T@row[2][:3,:3]).magnitude()>np.pi*.75]
    margin=best[0]-max([row[0] for row in opposite],default=0.)
    return best[2],dict(iou=best[0],cost=best[1],starts=len(candidates),weak_fit=weak,
                       opposite_resolved=bool(not opposite or margin>.02),opposite_iou_margin=float(margin))
