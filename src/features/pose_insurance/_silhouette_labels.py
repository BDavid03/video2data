"""Pure S1 episode label rules, correctness classification and panel verdict."""

import numpy as np

from ._silhouette import label_errors


def assigned_clusters(assigned: dict, trial: dict) -> list:
    """Select the frozen assignment format for an S1 object."""
    if trial['episode'] == 19:
        return assigned['observations']['cluster_ids'][trial['slot']]
    if trial['episode'] in (24, 27):
        return assigned['episodes'][str(trial['episode'])]['objects'][trial['name']]
    return assigned['episodes'][f"public:{trial['episode']}"]['objects'][trial['name']]


def partner_name(episode: int) -> str | None:
    """Return the relative-orientation label partner, if the case requires one."""
    return 'mini_dust_pan' if episode == 18 else None


def classify_case(trial: dict, prediction: dict, rotations: np.ndarray,
                  normal: np.ndarray, reference: dict, thresholds: dict,
                  partner: np.ndarray | None = None) -> dict:
    """Apply the frozen correctness and signed-identity agreement rules."""
    if trial['episode'] == 18:
        errors = label_errors(rotations, normal, reference[trial['name']],
                              partner=partner, partner_reference=reference[partner_name(trial['episode'])])
        correct = np.array(errors['C_error_deg']) < thresholds['C_error_deg']
        testable = int(correct.sum()) == 1
        reason = None if testable else 'requires_exactly_one_correct_candidate'
    else:
        canonical = np.array([1. if trial['name'].endswith('_1') else -1., 0, 0])
        errors = label_errors(rotations, normal, reference[trial['name']], canonical)
        correct = np.array(errors['up_error_deg']) < thresholds['up_error_deg']
        signed = np.array(errors['canonical_up_error_deg']) < thresholds['up_error_deg']
        testable = bool(np.array_equal(correct, signed) and correct.any())
        reason = 'identity_uncertain' if not np.array_equal(correct, signed) else (
            'no_correct_candidate' if not correct.any() else None)
    selected = prediction['decision']['selected']
    selected_correct = None if selected is None or not testable else bool(correct[selected])
    return dict(episode=trial['episode'], name=trial['name'], testable=testable, reason=reason,
                correct_candidates=np.flatnonzero(correct).tolist(), selected=selected,
                selected_correct=selected_correct, verdict='not_testable' if not testable else (
                    'abstain' if selected is None else 'correct' if selected_correct else 'WRONG'), **errors)


def assess_panel(cases: list, rows: list) -> dict:
    """Combine known-answer correctness, control completeness and application veto."""
    sweeper = next(c for c in cases if c['episode'] == 18)
    known_passed = sweeper['testable'] and sweeper['selected'] == 0 and all(
        c['selected_correct'] is not False for c in cases if c['testable'])
    controls = []
    for row in rows:
        if row['role'] != 'control':
            continue
        failures = [r for r in row['abstain_reasons'] if r != 'no_common_winner_with_margin']
        selections = [d['selected'] for d in row['search_decisions']]
        controls.append(dict(episode=row['episode'], name=row['name'], complete=not failures,
                             search_selections=selections, veto=1 in selections, reasons=failures))
    complete = all(c['complete'] for c in controls)
    veto = any(c['veto'] for c in controls)
    passed = bool(known_passed and complete and not veto)
    application = next(r for r in rows if r['role'] == 'application')
    selected = application['decision']['selected']
    outcome = 'application withheld' if not passed else 'abstain' if selected is None else (
        'agree' if selected == 0 else 'disagree')
    return dict(passed=passed, known_passed=bool(known_passed), known_cases=cases, controls=controls,
                controls_complete=complete, control_veto=veto, episode_19=outcome)
