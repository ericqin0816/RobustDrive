import numpy as np


def temporal_split(
    n: int, train: float = 0.70, validation: float = 0.15, test: float = 0.15, gap: int = 30
) -> dict[str, list[int]]:
    """Three chronological blocks, discarding gap frames at each boundary.

    Split row indices BEFORE expanding cameras or augmenting. Never shuffle the
    source CSV. Training may shuffle its own block after membership is fixed.
    """
    if min(train, validation, test) <= 0 or abs(train + validation + test - 1) > 1e-6:
        raise ValueError("Split fractions must be positive and sum to one.")
    if gap < 0:
        raise ValueError("gap must be nonnegative")
    first, second = int(n * train), int(n * (train + validation))
    left, right = gap // 2, gap - gap // 2
    splits = {
        "train": list(range(0, first - left)),
        "validation": list(range(first + right, second - left)),
        "test": list(range(second + right, n)),
    }
    if any(len(v) < 2 for v in splits.values()):
        raise ValueError(
            f"{n} rows are insufficient for these splits and gap={gap}; "
            "collect more data or reduce gap_frames for smoke tests."
        )
    retained = set(np.concatenate([np.asarray(x) for x in splits.values()]).tolist())
    splits["discarded"] = [i for i in range(n) if i not in retained]
    return splits
