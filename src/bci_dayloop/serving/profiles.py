from __future__ import annotations

from dataclasses import dataclass, replace

from bci_dayloop.realtime.runtime_mapping import APPROVED_NEURACLE_59_TO_STANDARD64

PROTOCOL_VERSION = 1
WINDOW_SEC = 4.0
STEP_SEC = 0.5
REVE_WINDOW_SEC = 2.0
REVE_STEP_SEC = 0.5
REVE_TARGET_HZ = 200.0

BCIGO_32_CHANNEL_NAMES: tuple[str, ...] = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "F7",
    "F8",
    "Fz",
    "C3",
    "C4",
    "Cz",
    "P3",
    "P4",
    "P7",
    "P8",
    "Pz",
    "O1",
    "O2",
    "T7",
    "T8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "FT9",
    "FT10",
    "TP9",
    "TP10",
    "IO",
)

MOTOR_IMAGERY_CLASS_NAMES: tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
)


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    id: str
    device: str
    channel_names: tuple[str, ...]
    sample_rate: float
    window_sec: float = WINDOW_SEC

    @property
    def channels(self) -> int:
        return len(self.channel_names)

    @property
    def samples(self) -> int:
        return int(round(self.window_sec * self.sample_rate))


NEURACLE59_PROFILE = DeviceProfile(
    id="neuracle59",
    device="neuracle",
    channel_names=APPROVED_NEURACLE_59_TO_STANDARD64.source_channel_names,
    sample_rate=1000.0,
)

BCIGO32_PROFILE = DeviceProfile(
    id="bcigo32",
    device="bcigo",
    channel_names=BCIGO_32_CHANNEL_NAMES,
    sample_rate=250.0,
)

DEVICE_PROFILES: dict[str, DeviceProfile] = {
    NEURACLE59_PROFILE.id: NEURACLE59_PROFILE,
    BCIGO32_PROFILE.id: BCIGO32_PROFILE,
}


def profile_with_window(profile: DeviceProfile, window_sec: float) -> DeviceProfile:
    if not (window_sec > 0):
        raise ValueError(f"window_sec must be positive, got {window_sec}")
    return replace(profile, window_sec=float(window_sec))


def get_device_profile(profile_id: str) -> DeviceProfile:
    try:
        return DEVICE_PROFILES[profile_id]
    except KeyError as error:
        known = ", ".join(sorted(DEVICE_PROFILES))
        raise ValueError(f"Unknown device profile {profile_id!r}. Known: {known}") from error


def match_device_profile(
    *,
    channel_names: tuple[str, ...],
    sample_rate: float,
    samples: int | None = None,
    require_samples: bool = True,
) -> DeviceProfile | None:
    for profile in DEVICE_PROFILES.values():
        if channel_names != profile.channel_names:
            continue
        if abs(sample_rate - profile.sample_rate) > 1e-6:
            continue
        if require_samples and samples is not None and samples != profile.samples:
            continue
        return profile
    return None
