"""Paint detection must retain oblique yard lines in a sideline broadcast."""
import cv2
import numpy as np

from gridiron_tracking.registration.lines import find_yard_lines


def test_diagonal_yard_lines_survive_the_angle_filter():
    image = np.full((480, 960, 3), (40, 130, 55), dtype=np.uint8)
    for x in (100, 350):
        cv2.line(image, (x, 80), (x + 480, 400), (245, 245, 245), 5)
    cv2.line(image, (30, 40), (920, 40), (245, 245, 245), 5)
    lines = find_yard_lines(image, min_support=150)
    for expected in (340, 590):
        assert any(abs(line.x_at(240) - expected) < 8 for line in lines)
    assert all(abs(line.x_at(400) - line.x_at(80)) > 400 for line in lines)
