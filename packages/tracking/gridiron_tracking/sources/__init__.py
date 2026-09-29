"""Where tracking comes from when it is not coming from a camera.

Public NGS releases are the benchmark's ground truth and the play model's training
set, so they are loaded through the same contract everything else speaks.
"""

from .ngs import (
    check_angle_convention,
    infer_play_direction,
    load_tracking,
    ngs_angle_to_contract,
    plays,
    to_play_track,
)

__all__ = [
    "check_angle_convention", "infer_play_direction", "load_tracking",
    "ngs_angle_to_contract", "plays", "to_play_track",
]
