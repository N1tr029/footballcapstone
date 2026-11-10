"""Camera motion must follow grass rather than screen-fixed broadcast overlays."""
import cv2
import numpy as np
import pytest

from gridiron_tracking.registration.homography import Homography
from gridiron_tracking.registration.propagate import HopQuality, PropagatedRegistration


@pytest.mark.parametrize("opaque", [True, False])
def test_camera_motion_ignores_a_stationary_scoreboard(opaque):
    rng = np.random.default_rng(8)
    field = np.full((240, 400, 3), (35, 125, 45), dtype=np.uint8)
    noise = rng.integers(-25, 26, (240, 400, 1), dtype=np.int16)
    before = np.clip(field.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    after = cv2.warpAffine(before, np.float32([[1, 0, 8], [0, 1, 3]]), (400, 240))
    for image in (before, after):
        if opaque:
            cv2.rectangle(image, (20, 155), (380, 239), (15, 15, 15), -1)
        for y in (200, 225):
            cv2.putText(image, "SCORE 123456789", (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
    reg = PropagatedRegistration(Homography(np.eye(3)), min_inliers=15,
                                 horizon_frac=0, max_features=400)
    hop, quality = reg._hop(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), 1,
                            prev_image=before, prev_idx=0)
    assert quality.ok
    assert np.allclose(hop[:2, 2], [8, 3], atol=0.5)


def test_a_failed_hop_keeps_the_last_registered_reference(monkeypatch):
    class Source:
        def frames(self, start, stop, step):
            for i in range(3):
                yield i, np.full((20, 20, 3), i * 50, dtype=np.uint8)

    reg = PropagatedRegistration(Homography(np.eye(3)))
    seen = []

    def hop(previous, current, i, **kwargs):
        seen.append(int(previous[0, 0]))
        return (None if i == 1 else np.eye(3)), HopQuality(i, 30, 30, i != 1)

    monkeypatch.setattr(reg, "_hop", hop)
    reg.build(Source(), 0, 3)
    assert seen == [0, 0]
    assert reg.at(1) is None
    assert reg.at(2) is not None
