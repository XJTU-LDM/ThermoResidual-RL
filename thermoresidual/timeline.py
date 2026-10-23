"""Timestamp audit utilities; no interpolation and no implicit gap propagation."""
import numpy as np


def checked_times(times_s):
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("At least two finite timestamps required")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Duplicate or nonmonotone timestamps require explicit resolution")
    return times


def segments(times_s, gap_multiple=2.0):
    times = checked_times(times_s)
    if not np.isfinite(gap_multiple) or gap_multiple <= 1:
        raise ValueError("Gap multiple must exceed one")
    cadence = float(np.median(np.diff(times)))
    breaks = np.flatnonzero(np.diff(times) > gap_multiple * cadence) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(times)]
    return [{"segment_id": i, "start_row": int(start), "end_row_exclusive": int(end),
             "start_s": float(times[start]), "last_sample_s": float(times[end-1]),
             "samples": int(end-start)}
            for i, (start, end) in enumerate(zip(starts, ends))]


def backward_match(source_s, decisions_s, max_age_s):
    """Return diagnostic source indices; -1 means no eligible observed sample.

    Assumes zero latency ONLY for timestamp audit. This is not a sensor adapter.
    Matches never reach across an identified source gap, even with large max_age.
    """
    source = checked_times(source_s)
    decisions = np.asarray(decisions_s, dtype=float)
    if decisions.ndim != 1 or not np.isfinite(decisions).all():
        raise ValueError("Finite one-dimensional decision timestamps required")
    if not np.isfinite(max_age_s) or max_age_s < 0:
        raise ValueError("Nonnegative finite maximum age required")
    indices = np.searchsorted(source, decisions, side="right") - 1
    clipped = np.maximum(indices, 0)
    valid = (indices >= 0) & (decisions - source[clipped] <= max_age_s)
    for segment in segments(source)[:-1]:
        gap_left = segment["last_sample_s"]
        gap_right = source[segment["end_row_exclusive"]]
        valid &= ~((decisions > gap_left) & (decisions < gap_right))
    return np.where(valid, indices, -1)


def split_windows(times_s, blocks):
    """Intersect episodes with development/test blocks without crossing either."""
    times = checked_times(times_s)
    ordered = list(blocks.items())
    previous_end = 0
    for name, (start, end) in ordered:
        if type(start) is not int or type(end) is not int or start != previous_end or not start < end <= len(times):
            raise ValueError(f"Invalid contiguous half-open block: {name}")
        previous_end = end
    if previous_end != len(times):
        raise ValueError("Blocks must cover the full record")
    rows = []
    for name, (start, end) in ordered:
        for segment in segments(times):
            lo, hi = max(start, segment["start_row"]), min(end, segment["end_row_exclusive"])
            if lo < hi:
                rows.append({"block": name, "segment_id": segment["segment_id"],
                             "start_row": lo, "end_row_exclusive": hi,
                             "start_s": float(times[lo]), "last_sample_s": float(times[hi-1]),
                             "samples": hi-lo})
    return rows
