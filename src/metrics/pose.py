"""Dependency-free SE(3) primitives, in metres and scalar-first quaternions.

Pose(A_T_B) maps column-vector coordinates in B into A. No alignment is
estimated here. All returned angular errors are radians, with no symmetry
reduction. Invalid/missing rows must be accounted for by the caller.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

QUATERNION_NORM_TOLERANCE = 1e-3


def _finite_vector(values: Sequence[float], length: int) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"expected {length} components")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise ValueError("components must be real numbers")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("components must be finite")
    return result


def _multiply(a: tuple, b: tuple) -> tuple:
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def _conjugate(q: tuple) -> tuple:
    return (q[0], -q[1], -q[2], -q[3])


def _rotate(q: tuple, point: tuple) -> tuple:
    return _multiply(_multiply(q, (0.0, *point)), _conjugate(q))[1:]


@dataclass(frozen=True)
class Pose:
    """Validated transform; small quaternion roundoff is normalized once."""

    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        translation = _finite_vector(self.translation, 3)
        quaternion = _finite_vector(self.quaternion, 4)
        norm = math.hypot(*quaternion)
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise ValueError("quaternion norm outside 1 +/- 1e-3")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "quaternion", tuple(v / norm for v in quaternion))

    @classmethod
    def from_row(cls, pose: Sequence[float], *, valid: bool) -> "Pose | None":
        """Filter visibility/validity before reading a pose, including zero rows.

        For public labels pass visible as valid. For predictions pass the
        explicit track-validity flag. A flagged-valid malformed pose raises;
        it must not silently become an identity or disappear from coverage.
        """
        if not isinstance(valid, bool):
            raise ValueError("valid must be a boolean")
        if not valid:
            return None
        values = _finite_vector(pose, 7)
        return cls(values[:3], values[3:])

    def compose(self, other: "Pose") -> "Pose":
        """A_T_B.compose(B_T_C) returns A_T_C."""
        rotated = _rotate(self.quaternion, other.translation)
        return Pose(tuple(a + b for a, b in zip(self.translation, rotated)),
                    _multiply(self.quaternion, other.quaternion))

    def inverse(self) -> "Pose":
        q = _conjugate(self.quaternion)
        return Pose(_rotate(q, tuple(-v for v in self.translation)), q)


def pose_error(reference: Pose, prediction: Pose) -> tuple[float, float]:
    """Return translation L2 in metres and SO(3) angle in [0, pi] radians.

    Inputs must already share a coordinate frame. q and -q are equivalent.
    atan2 preserves small-angle accuracy better than acos(dot).
    """
    translation = math.dist(reference.translation, prediction.translation)
    delta = _multiply(_conjugate(reference.quaternion), prediction.quaternion)
    angle = 2.0 * math.atan2(math.hypot(*delta[1:]), abs(delta[0]))
    return translation, angle


def relative_pose(start: Pose, end: Pose) -> Pose:
    """Return start_T_end; invariant to one common left SE(3) transform."""
    return start.inverse().compose(end)


def coverage(matched: int, valid_reference: int) -> float | None:
    """Matched valid predictions / valid labels; None when unsupported.

    The denominator is independent of predictions. Missing/invalid predicted
    rows count against coverage. A zero-coverage run has no pose-error mean.
    """
    if any(isinstance(v, bool) or not isinstance(v, int)
           for v in (matched, valid_reference)):
        raise ValueError("counts must be integers")
    if not 0 <= matched <= valid_reference:
        raise ValueError("require 0 <= matched <= valid_reference")
    return matched / valid_reference if valid_reference else None
