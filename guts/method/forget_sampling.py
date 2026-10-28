"""Stratified sampling of forget-set targets, shared by the dataset forget-set builders.

Strata are quartiles of a per-target measure (degree, trajectory length, ...): stratum i holds
bounds[i-1] < value <= bounds[i]. Targets are drawn from the strata in proportion to their size."""

from __future__ import annotations

import bisect
import random
from typing import Dict, Hashable, List

import numpy as np

NUM_STRATA = 4


def quartile_bounds(values: List[int]) -> List[int]:
    """Upper bounds of the lower NUM_STRATA - 1 strata (quantiles of the values; equal ones merged,
    and none at the maximum, which would leave the top stratum empty)."""
    values = np.asarray(values)
    qs = np.quantile(values, [i / NUM_STRATA for i in range(1, NUM_STRATA)], method="lower")
    return sorted({int(q) for q in qs if q < values.max()})


def stratum_of(value: int, bounds: List[int]) -> int:
    """Stratum i holds bounds[i-1] < value <= bounds[i]; the last one holds value > bounds[-1]."""
    return bisect.bisect_left(bounds, value)


def stratum_labels(bounds: List[int]) -> List[str]:
    labels = []
    for i in range(len(bounds) + 1):
        if i == 0:
            labels.append(f"<={bounds[0]}" if bounds else "all")
        elif i == len(bounds):
            labels.append(f">{bounds[-1]}")
        else:
            lo, hi = bounds[i - 1] + 1, bounds[i]
            labels.append(str(hi) if lo == hi else f"{lo}-{hi}")
    return labels


def sample_from_strata(buckets: Dict[Hashable, list], pool_size: int, n: int, rng: random.Random) -> list:
    """n items, each stratum getting a share proportional to its size (at least one).

    Strata are visited in sorted key order; any shortfall is filled from the strata in
    random order."""
    result: list = []
    result_set: set = set()
    remaining = n
    for _, bpool in sorted(buckets.items()):
        quota = max(1, round(n * len(bpool) / pool_size))
        quota = min(quota, len(bpool), remaining)
        sampled = rng.sample(bpool, quota)
        result.extend(sampled)
        result_set.update(sampled)
        remaining -= quota

    if remaining > 0:
        bucket_list = list(buckets.values())
        rng.shuffle(bucket_list)
        for bpool in bucket_list:
            if remaining <= 0:
                break
            k = min(len(bpool), remaining * 3 + 5)
            for cand in rng.sample(bpool, k):
                if cand not in result_set:
                    result.append(cand)
                    result_set.add(cand)
                    remaining -= 1
                    if remaining <= 0:
                        break
    return result[:n]
