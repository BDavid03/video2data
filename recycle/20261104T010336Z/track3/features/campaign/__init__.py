"""Disabled standard-library campaign records; never schedules or executes stages."""

from .operations import CampaignRequest, campaign
from .evidence import EvidenceRequest, campaign_evidence

__all__ = ["CampaignRequest", "campaign", "EvidenceRequest", "campaign_evidence"]
