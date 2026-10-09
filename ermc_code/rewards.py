"""First-hit rewards are supplied by the wrapper; cap them per collection."""
import numpy as np


def combine_rewards(base_rewards, raw_auxiliary, mean_cap=0.00005):
    """Combine an already-preprocessed base pathway with bounded raw bonuses.

    Call once on a complete collection batch, before returns/advantages are
    computed. Do not call on resampled replay minibatches. The host is
    responsible for forming base_rewards from external and RM rewards using
    the same preprocessing and coefficients in all compared methods.
    Returns combined rewards and framework-neutral scalar diagnostic keys.
    """
    base = np.asarray(base_rewards, dtype=np.float32)
    auxiliary = np.asarray(raw_auxiliary, dtype=np.float32)
    if base.shape != auxiliary.shape or not auxiliary.size:
        raise ValueError('Expected equally shaped nonempty reward arrays')
    if not np.isfinite(base).all() or not np.isfinite(auxiliary).all():
        raise ValueError('Rewards must be finite')
    if (auxiliary < 0).any() or not np.isfinite(mean_cap) or mean_cap < 0:
        raise ValueError('Invalid auxiliary reward or mean budget')
    raw_sum = float(auxiliary.sum(dtype=np.float64))
    scale = min(1., mean_cap*auxiliary.size/raw_sum) if raw_sum > 0 else 1.
    paid = auxiliary * scale
    return base + paid, {
        'auxiliary/raw_sum': raw_sum,
        'auxiliary/paid_sum': float(paid.sum(dtype=np.float64)),
        'auxiliary/scale': scale,
    }
