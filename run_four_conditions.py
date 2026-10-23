"""Target waveform definitions only; no field-boundary replay environment."""
import math
import torch

SCENARIOS = {
    "C1_low_range_slow": {
        "label": "Near-rated low-amplitude / slow switching",
        "power_low": 24.0,
        "power_high": 32.0,
        "cooling_low": 36.0,
        "cooling_high": 44.0,
        "plateau": 600,
        "transition": 200,
    },
    "C2_high_range_slow": {
        "label": "Part-load to bounded transient / slow switching",
        "power_low": 15.0,
        "power_high": 36.0,
        "cooling_low": 30.0,
        "cooling_high": 50.0,
        "plateau": 600,
        "transition": 200,
    },
    "C3_low_range_fast": {
        "label": "Near-rated low-amplitude / fast switching",
        "power_low": 24.0,
        "power_high": 32.0,
        "cooling_low": 36.0,
        "cooling_high": 44.0,
        "plateau": 150,
        "transition": 50,
    },
    "C4_high_range_fast": {
        "label": "Part-load to bounded transient / fast switching",
        "power_low": 15.0,
        "power_high": 36.0,
        "cooling_low": 30.0,
        "cooling_high": 50.0,
        "plateau": 150,
        "transition": 50,
    },
}


def cosine_target(step, low, high, plateau, transition):
    cycle = 2 * (plateau + transition)
    pos = torch.remainder(step, cycle)
    high_mask = pos < plateau
    down_mask = (pos >= plateau) & (pos < plateau + transition)
    low_mask = (pos >= plateau + transition) & (pos < 2 * plateau + transition)
    up_pos = (pos - (2 * plateau + transition)).clamp(min=0)
    down_mu = ((pos - plateau) / max(1, transition)).clamp(0, 1)
    up_mu = (up_pos / max(1, transition)).clamp(0, 1)
    down = high + (low - high) * (1 - torch.cos(math.pi * down_mu)) / 2
    up = low + (high - low) * (1 - torch.cos(math.pi * up_mu)) / 2
    return torch.where(high_mask, high, torch.where(down_mask, down, torch.where(low_mask, low, up)))


def _pseudo_random_level(idx: torch.Tensor, scenario_id: int, channel_offset: float) -> torch.Tensor:
    value = torch.sin((idx.float() + 1.0 + 17.0 * scenario_id + channel_offset) * 12.9898) * 43758.5453
    return torch.remainder(value, 1.0)


def mixed_step_ramp_target(step: torch.Tensor, low: float, high: float, scenario_id: int, channel_offset: float) -> torch.Tensor:
    segment = 180.0 if scenario_id in (0, 1) else 120.0
    ramp = 45.0 if scenario_id in (0, 1) else 30.0
    local = torch.remainder(step + 31.0 * channel_offset, segment)
    idx = torch.floor((step + 31.0 * channel_offset) / segment)
    level0 = low + (high - low) * _pseudo_random_level(idx, scenario_id, channel_offset)
    level1 = low + (high - low) * _pseudo_random_level(idx + 1.0, scenario_id, channel_offset)
    mu = ((local - (segment - ramp)) / max(ramp, 1.0)).clamp(0.0, 1.0)
    smooth = mu * mu * (3.0 - 2.0 * mu)
    return level0 + (level1 - level0) * smooth


def target_for_profile(step: torch.Tensor, spec: dict, scenario_id: int, cooling: bool, profile: str) -> torch.Tensor:
    phase_offset = (spec["plateau"] + spec["transition"]) / 2 if cooling else 0.0
    s = step + phase_offset
    if profile == "standard":
        return cosine_target(
            s,
            spec["cooling_low"] if cooling else spec["power_low"],
            spec["cooling_high"] if cooling else spec["power_high"],
            spec["plateau"],
            spec["transition"],
        )
    if profile in {"medium_unseen", "medium_interpolation"}:
        return cosine_target(
            s,
            spec["cooling_low"] if cooling else spec["power_low"],
            spec["cooling_high"] if cooling else spec["power_high"],
            300.0,
            100.0,
        )
    if profile in {"very_fast_unseen", "very_fast_extrapolation"}:
        return cosine_target(
            s,
            spec["cooling_low"] if cooling else spec["power_low"],
            spec["cooling_high"] if cooling else spec["power_high"],
            75.0,
            25.0,
        )
    if profile in {"mixed_step_ramp", "mixed_shape_transfer"}:
        if cooling:
            low = 28.0 if scenario_id in (1, 3) else 32.0
            high = 52.0 if scenario_id in (1, 3) else 48.0
        else:
            low = 15.0 if scenario_id in (1, 3) else 24.0
            high = 36.0 if scenario_id in (1, 3) else 32.0
        return mixed_step_ramp_target(s, low, high, scenario_id, 9.0 if cooling else 0.0)
    raise ValueError(f"Unknown target_profile: {profile}")

