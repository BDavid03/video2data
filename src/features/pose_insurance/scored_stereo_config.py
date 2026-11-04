"""Paths and fixed budgets for the scored stereo and heading stages."""
from pathlib import Path
from src.runs import ROOT

FEATURE = Path(__file__).parent
KIT = ROOT / 'upstream/v2d_submission_kit/v2d_submission_kit'
FC = ROOT / 'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord'
BUDGET = dict(environment_steps=512, worlds=1, replay_minimum=32)
EPISODES = (3, 6, 8, 10, 14, 15, 16, 19, 20, 25, 30, 32, 33)
UNTOUCHED = (1, 5, 28, 35, 36, 37, 38)
POSE_PROVENANCE = ('video_only', 'public_asset_diagnostic')
