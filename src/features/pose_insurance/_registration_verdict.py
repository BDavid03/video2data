"""Pure panel verdicts and selected-pose diagnostics for saved HR2 predictions."""

import numpy as np
from scipy.spatial.transform import Rotation

from ._placement import transform_errors


def selected_pairs(predictions: list, references: dict, pairs: list) -> list:
    """Measure each configured pair without overriding saved acceptance decisions."""
    by_slot = {(r['episode'], r['slot']): r for r in predictions}
    results = []
    for pair in pairs:
        episode = pair['episode']
        rows = [by_slot[episode, slot] for slot in pair['slots']]
        selected = np.concatenate((np.array([r['translation_m'] for r in rows]),
            Rotation.from_matrix(np.array([r['rotation'] for r in rows])).as_quat()), axis=1)
        measured = transform_errors(selected, references[episode])
        predicted, reference = measured['achieved_offset_m'], measured['reference_offset_m']
        accepted = all(r['accepted'] for r in rows)
        results.append(dict(episode=episode, accepted=accepted,
            b_error_deg=measured['direction_error_deg'],
            C_error_deg=measured['relative_rotation_error_deg'],
            predicted_b_m=predicted, reference_b_m=reference, descriptive_only=not accepted,
            predicted_offset_cm=float(np.linalg.norm(predicted) * 100),
            reference_offset_cm=float(np.linalg.norm(reference) * 100)))
    return results


def panel_verdict(up: list, pairs: list, spec: dict) -> dict:
    """Apply strict angle bounds and inclusive coverage minima from the admitted spec."""
    pair_limits, up_limits = spec['pair_thresholds'], spec['up_thresholds']
    by_episode = {r['episode']: r for r in pairs}
    angle_pass = {r['episode']: r['b_error_deg'] is not None
        and r['b_error_deg'] < pair_limits['b_error_deg']
        and r['C_error_deg'] < pair_limits['C_error_deg'] for r in pairs}
    pair_checks = {}
    for episode in spec['mandatory_pairs']:
        accepted = by_episode[episode]['accepted']
        pair_checks[f'mandatory_{episode}_accepted'] = accepted
        pair_checks[f'mandatory_{episode}_angles'] = accepted and angle_pass[episode]
    pair_checks['every_accepted_pair_angles'] = all(
        angle_pass[r['episode']] for r in pairs if r['accepted'])
    accepted = [r for r in up if r['accepted']]
    episodes = sorted({r['episode'] for r in accepted})
    types = sorted({r['name'] for r in accepted})
    up_checks = {
        f"at_least_{up_limits['minimum_accepted_slots']}_accepted":
            len(accepted) >= up_limits['minimum_accepted_slots'],
        f"at_least_{up_limits['minimum_episodes']}_episodes":
            len(episodes) >= up_limits['minimum_episodes'],
        f"at_least_{up_limits['minimum_object_types']}_object_types":
            len(types) >= up_limits['minimum_object_types'],
        f"every_accepted_up_below_{up_limits['up_error_deg']}": all(
            r['up_error_deg'] is not None and r['up_error_deg'] < up_limits['up_error_deg']
            for r in accepted)}
    pair_panel = dict(passed=all(pair_checks.values()),
        accepted_pairs=sum(r['accepted'] for r in pairs), checks=pair_checks)
    up_panel = dict(passed=all(up_checks.values()), accepted_slots=len(accepted),
        accepted_episodes=episodes, accepted_object_types=types,
        max_accepted_up_error_deg=max((r['up_error_deg'] for r in accepted
                                      if r['up_error_deg'] is not None), default=None),
        checks=up_checks)
    return dict(pair_panel=pair_panel, up_panel=up_panel,
                overall_passed=pair_panel['passed'] and up_panel['passed'])
