"""Standalone, disabled research components; no model imports at package import."""

from .planning import PlanRequest, plan_trials

__all__ = ["PlanRequest", "plan_trials"]
