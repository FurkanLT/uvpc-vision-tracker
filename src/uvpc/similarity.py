"""Pure vector similarity helpers; not an appearance/Re-ID model."""

from math import isfinite, sqrt
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or len(a) != len(b):
        raise ValueError("Vectors must be nonempty and have equal lengths")
    if not all(isfinite(v) for v in (*a, *b)):
        raise ValueError("Vector entries must be finite")
    na = sqrt(sum(v * v for v in a))
    nb = sqrt(sum(v * v for v in b))
    if na == 0 or nb == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - cosine_similarity(a, b)
