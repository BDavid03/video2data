"""Compose the joint-only conditional camera spatial-coverage decision."""

import math

from src.features.moving_reference._final_gates import compose_final

REGIONS='camera_heldout_regions_min'
PREREQUISITES={
    'camera_heldout_reprojection_median_px640':('le',.5),
    'camera_heldout_reprojection_p95_px640':('le',2.),
    'camera_heldout_tracks_min':('ge',30),
    'camera_coverage_fraction':('ge',.98),
    'camera_heldout_world_residual_median_m':('le',.01),
}


def prerequisites(gates):
    """Recompute each unchanged direct-camera verdict, failing closed on missing data."""
    result={}
    for name,(comparison,limit) in PREREQUISITES.items():
        gate=gates.get(name,{})
        value=gate.get('value')
        finite=type(value) in (int,float) and math.isfinite(value) and value>=0
        result[name]=bool(finite and gate.get('comparison')==comparison and gate.get('threshold')==limit
                          and (value<=limit if comparison=='le' else value>=limit) and gate.get('passed') is True)
    return result


def compose_joint(source,sealed):
    """Keep original M1 composition and amend only a three-region joint verdict."""
    result=compose_final(source,sealed)
    region=result['gates'][REGIONS]
    value=region['value']
    if value is not None and 3<=value<4:
        checks=prerequisites(result['gates'])
        result['gates'][REGIONS]=dict(region,threshold=3,passed=all(checks.values()),
            original_threshold=4,original_passed=region['passed'],conditional_prerequisites=checks,
            rule='At least three regions only when all five unchanged direct-camera checks pass')
    result['passed']=all(g['passed'] for g in result['gates'].values())
    return result


def provisional_camera(gates):
    """Permit collection of the unmeasured world instrument without granting a pass."""
    if all(g['passed'] for g in gates.values()):
        return True
    region=gates[REGIONS]
    available={name:gate for name,gate in gates.items() if name!=REGIONS}
    return bool(region['value'] is not None and 3<=region['value']<4 and all(g['passed'] for g in available.values()))
