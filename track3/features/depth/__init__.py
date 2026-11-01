"""Depth generation feature; importing it starts no processing or model load."""

from .generate import GenerateDepthRequest, generate_depth

__all__ = ["GenerateDepthRequest", "generate_depth"]
