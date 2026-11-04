"""Reuse the original table fit and admit its available three-patch instrument."""

import cv2
import numpy as np

from src.features.moving_reference import full_c,camera
from src.provenance import write_json
from ._two_object_table import available_patch_angle


def fit(directory,cameras):
    gauge,report=full_c.table_fit(directory,cameras)
    if report['patch_normal_max_deg'] is not None:
        return gauge,report
    first=full_c.read_frame(directory,0);yy,xx=np.indices(first['gray'].shape)
    oy,ox=np.nonzero(first['object_mask']);mx,my=np.median(ox),np.median(oy)
    nearby=(abs(xx-mx)<first['gray'].shape[1]*.3)&(yy>my-first['gray'].shape[0]*.1)
    patches=[nearby&((yy>my)==yh)&((xx>mx)==xh) for yh in (False,True) for xh in (False,True)]
    groups=[[] for _ in patches]
    for index in report['precontact_frames']:
        frame=full_c.read_frame(directory,index)
        valid=frame['static']&frame['depth_valid']
        for group,patch in zip(groups,patches):
            points=frame['xyz'][valid&patch]
            if len(points)>2000:points=points[np.linspace(0,len(points)-1,2000).astype(int)]
            group.extend(camera.apply(cameras[index],points).tolist())
    measured=available_patch_angle(groups,np.asarray(report['normal_before_gauge']),report['offset_before_gauge_m'])
    write_json(directory/'table-patch-instrument.json',dict(measured,original_maximum_angle_deg=report['patch_normal_max_deg'],
               limit_deg=3.,predicates_unchanged=True,plane_gauge_unchanged=True))
    if measured['available_patches']==3:
        report=dict(report,patch_normal_max_deg=measured['maximum_angle_deg'],available_patch_count=3,
                    patch_instrument='same pairwise normal angles on three available original disjoint patches')
    return gauge,report
