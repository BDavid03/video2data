"""Inputs for one pre-contact object feasibility measurement."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageARequest:
    episode_index: int
    object_name: str
    cluster_ids: list[int]
    depth_run: str
    measure_run: str
    stereo_reference_run: str
    first_contact_frame: int
    contact_annotation: dict
    source_split: str = 'public'
    registration: str = 'pnp_icp'

    def __post_init__(self):
        if self.registration not in ('pnp_icp', 'pnp'):
            raise ValueError('Registration must be pnp_icp (F1) or pnp (F1b)')
        if self.source_split != 'public' or self.episode_index not in (12, 42):
            raise PermissionError('F1 Stage A admits public 12/42 only; 41/43/44 are excluded')
        expected = {12: ('plastic_dish_rack', [0, 2, 3, 4, 5, 6]), 42: ('white_pot', [0])}
        if (self.object_name, self.cluster_ids) != expected[self.episode_index]:
            raise ValueError('Object and opening clusters must match the frozen F1 protocol')
        if not 1 <= self.first_contact_frame <= 200:
            raise ValueError('Supply a visually annotated first-contact boundary (1..200)')
        if not self.contact_annotation.get('method') or not self.contact_annotation.get('refined_frames'):
            raise ValueError('Contact annotation needs its method and refined frame identities')

    @property
    def candidate_frames(self):
        return list(range(0, self.first_contact_frame, 5))

    @property
    def variant(self):
        return 'F1b' if self.registration == 'pnp' else 'F1'
