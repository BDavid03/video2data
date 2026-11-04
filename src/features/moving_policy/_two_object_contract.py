"""Validate joint Stage0, source-reference and frozen-recipe identities."""

import numpy as np
from src.features.pose_insurance._moving import translation_errors

STAGE0_CHECKS={'1_schema','2_closure','3_penetration','4a_finger_limits','4b_reachability','5_support','6_truth','6_joint_truth'}


def require_stage_checks(stage,episode):
    checks=stage.get('checks',{})
    if (stage.get('status')!='PASS' or stage.get('episode')!=episode or stage.get('plumbing_only')
            or not STAGE0_CHECKS<=checks.keys() or not all(c.get('passed') is True for c in checks.values())):
        raise ValueError('Every named joint Stage-0 check must be present and passed is True')


def require_source_reference(task,source,frames,times,fps,shift,active):
    task=np.asarray(task,float);source=np.asarray(source,float);times=np.asarray(times,float)
    if (source.shape!=(len(task),7) or not np.array_equal(frames,np.arange(len(task)))
            or times.shape!=(len(task),) or not np.isfinite(fps) or fps<=0
            or not np.allclose(times-times[0],np.arange(len(task))/fps,atol=1e-6,rtol=0)):
        raise ValueError('A source frame grid or clock differs from the actual task')
    errors=translation_errors(source[:,None,:],task[:,active:active+1,:],np.asarray(shift,float))
    if errors['maximum_translation_error_m']>1e-6 or errors['maximum_orientation_error_deg']>1e-4:
        raise ValueError('A full pose sequence differs from its translated source reference')
    return errors


def recipe_values(config):
    return dict(seed=config['reset']['seed'],world_count=config['scene']['world_count'],
                update_ratio=config['training']['updates_per_collection'],curriculum=config['curriculum'])


def require_recipe(config,episode38):
    try:
        actual=recipe_values(config);expected=recipe_values(episode38)
    except KeyError as error:
        raise ValueError('Checkpoint resolved configuration is missing frozen recipe fields') from error
    if expected['seed']!=0 or expected['world_count']!=1024 or expected['update_ratio']!=2:
        raise ValueError('Episode38 saved configuration differs from the frozen recipe')
    if actual!=expected:
        raise ValueError('Checkpoint saved seed/worlds/update ratio/curriculum differs from episode38')
    return actual
