"""Feature-owned deterministic partition and bounded robust scale location."""

from hashlib import sha256

PARTITION = "sha256-episode-frame-query-v1"


def reserved(episode, frame, query_id, modulus=5):
    """The ID is stable before outcomes; zero residue is never fit support."""
    identity = f"{episode}/{frame}/{query_id}".encode("utf-8")
    return int.from_bytes(sha256(identity).digest()[:8], "big") % modulus == 0


def reserved_grid(episode, frame, shape, np):
    height, width = shape
    return np.fromiter((reserved(episode, frame, f"a:{v}:{u}")
                        for v in range(height) for u in range(width)),
                       dtype=bool, count=height * width).reshape(shape)


def episode_location(scales, method, np):
    if method == "median":
        return float(np.median(scales))
    values = np.log(np.asarray(scales, dtype=np.float64))
    centre = float(np.median(values))
    if method == "huber_log":
        # Fixed iteration count and tuning constant: no outcome-driven search.
        width = max(float(1.4826 * np.median(abs(values - centre))), 1e-6)
        for _ in range(20):
            weights = np.minimum(1., 1.345 * width / np.maximum(abs(values - centre), 1e-12))
            updated = float(np.sum(values * weights) / np.sum(weights))
            if abs(updated - centre) < 1e-10:
                break
            centre = updated
    return float(np.exp(centre))
