"""Named contact-objective recipes; the simulator and observations stay upstream."""

from copy import deepcopy


CONTACT_TERMS = (
    ("contact_wrench_support", True, 10.0),
    ("missed_contact", True, -1.0),
    ("unintended_contact", True, -10.0),
    ("force_closure", False, 0.0),
    ("contact_force_l2", False, 0.0),
)
RECIPE_VARIANTS = ("contact_aware", "motion_only")


def recipe_overrides(variant, curriculum_stages):
    """Override contact terms and replace the whole (alias-free) curriculum so no later stage restores them."""
    if variant not in RECIPE_VARIANTS or len(curriculum_stages) != 11:
        raise ValueError("Expected a named recipe and all eleven pinned stages")
    stages = [deepcopy(stage) for stage in curriculum_stages]
    values = []
    for name, enabled, weight in CONTACT_TERMS:
        active, amount = (enabled, weight) if variant == "contact_aware" else (False, 0.0)
        for suffix, value in (("enabled", active), ("weight", amount)):
            values.append({"key": f"objective.{name}.{suffix}", "value": value,
                           "reason": f"Immutable {variant} contact-objective recipe"})
        for stage in stages:
            if name not in stage["objective_weights"]:
                raise ValueError(f"Pinned curriculum lacks {name}")
            stage["objective_weights"][name] = amount
    values.append({"key": "curriculum.stages", "value": stages,
                   "reason": "Explicit contact weights in every stage; preserve assistance, reset and motion terms"})
    return values
