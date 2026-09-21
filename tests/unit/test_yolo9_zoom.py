"""The box-aware zoom knob of the YOLO9 train transform.

``zoom_to_boxes`` itself is pinned in ``test_augment_geometry.py``. These tests
cover the knob around it: off by default and silent on the RNG, a real
magnification of the labelled object when on, cut from the unresized image the
dataset hands over while the knob is on, and skipped for samples the zoom does
not handle.
"""

from __future__ import annotations

import logging
import random

import cv2
import numpy as np
import pytest

from libreyolo.data.augment.yolo9 import YOLO9MosaicMixupDataset, YOLO9TrainTransform
from libreyolo.data.dataset import YOLODataset

pytestmark = pytest.mark.unit

_INPUT = (64, 64)


def _transform(**kwargs):
    """A transform whose only random op is the one under test."""
    kwargs.setdefault("flip_prob", 0.0)
    kwargs.setdefault("hsv_prob", 0.0)
    return YOLO9TrainTransform(**kwargs)


def _scene(height=360, width=200, box=(80, 150, 110, 190)):
    """A dark frame with one bright object, and its ``[x1, y1, x2, y2, class]`` row."""
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = 220
    return img, np.array([[x1, y1, x2, y2, 0]], dtype=np.float32)


def _rows(padded):
    """The label rows that are not padding, as ``[class, x1, y1, x2, y2]``."""
    return padded[padded[:, 0] >= 0]


def _count_draws(monkeypatch):
    calls = []
    real = random.random

    def counting():
        calls.append(1)
        return real()

    monkeypatch.setattr(random, "random", counting)
    return calls


def test_zoom_is_off_by_default_and_asks_for_nothing():
    transform = YOLO9TrainTransform()
    assert transform.zoom_prob == 0.0
    assert transform.wants_unresized_image is False


def test_zoom_off_draws_no_random_number(monkeypatch):
    """hsv, flip and vertical flip are the three draws of the detection path."""
    img, targets = _scene()
    calls = _count_draws(monkeypatch)
    _transform(zoom_prob=0.0)(img.copy(), targets.copy(), _INPUT)
    assert len(calls) == 3
    calls.clear()
    _transform(zoom_prob=0.0)(img.copy(), np.zeros((0, 5), dtype=np.float32), _INPUT)
    assert len(calls) == 3


def test_zoom_on_asks_the_dataset_for_the_unresized_image():
    assert _transform(zoom_prob=0.3).wants_unresized_image is True


@pytest.mark.parametrize("zoom_range", [(0.5, 2.0), (3.0, 2.0)])
def test_a_range_that_is_not_a_zoom_in_is_refused_at_construction(zoom_range):
    with pytest.raises(ValueError):
        _transform(zoom_prob=0.5, zoom_range=zoom_range)


def test_zoom_magnifies_the_object_and_keeps_it_whole():
    img, targets = _scene()
    _, plain = _transform()(img.copy(), targets.copy(), _INPUT)
    plain_w = (_rows(plain)[0, 3] - _rows(plain)[0, 1]) * _INPUT[1]
    for seed in range(25):
        random.seed(seed)
        out, padded = _transform(zoom_prob=1.0, zoom_range=(3.0, 3.0))(
            img.copy(), targets.copy(), _INPUT
        )
        rows = _rows(padded)
        assert len(rows) == 1
        _cls, x1, y1, x2, y2 = rows[0]
        assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0
        assert (x2 - x1) * _INPUT[1] == pytest.approx(3 * plain_w, rel=0.06)
        # The label sits on the bright object in the image that is returned.
        cx, cy = int((x1 + x2) / 2 * _INPUT[1]), int((y1 + y2) / 2 * _INPUT[0])
        assert out[:, cy, cx].min() > 0.7


def _padded_columns(image):
    """Trailing columns of a CHW sample that are nothing but letterbox padding."""
    padding = np.isclose(image, 114 / 255, atol=1e-3).all(axis=(0, 1))
    return int(padding[::-1].argmin()) if not padding.all() else len(padding)


def test_a_zoomed_sample_fills_the_input_unless_told_to_keep_the_image_s_shape():
    """The 360 x 200 scene letterboxes into 64 x 64 with 29 columns of padding."""
    img, targets = _scene()
    plain, _ = _transform()(img.copy(), targets.copy(), _INPUT)
    band = _padded_columns(plain)
    assert band == 64 - int(200 * 64 / 360)
    random.seed(5)
    filled, _ = _transform(zoom_prob=1.0, zoom_range=(3.0, 3.0))(
        img.copy(), targets.copy(), _INPUT
    )
    assert _padded_columns(filled) == 0
    random.seed(5)
    kept, _ = _transform(zoom_prob=1.0, zoom_range=(3.0, 3.0), zoom_fill=False)(
        img.copy(), targets.copy(), _INPUT
    )
    assert _padded_columns(kept) == band


def test_zoom_fill_magnifies_as_much_as_the_image_s_own_shape_does():
    """Filling the input changes what surrounds the object, not how large it comes out."""
    img, targets = _scene()
    widths = []
    for fill in (True, False):
        random.seed(9)
        _, padded = _transform(zoom_prob=1.0, zoom_range=(3.0, 3.0), zoom_fill=fill)(
            img.copy(), targets.copy(), _INPUT
        )
        widths.append(_rows(padded)[0, 3] - _rows(padded)[0, 1])
    assert widths[0] == pytest.approx(widths[1], rel=0.02)


def test_the_largest_zoom_leaves_the_margin_around_the_object():
    """A zoom far past what fits: the object takes 1 / (1 + 2 * margin) of the sample."""
    img, targets = _scene(box=(80, 150, 120, 190))  # 40 x 40
    for margin, share in [(0.0, 1.0), (0.1, 1 / 1.2), (0.5, 0.5)]:
        random.seed(2)
        _, padded = _transform(
            zoom_prob=1.0, zoom_range=(60.0, 60.0), zoom_margin=margin
        )(img.copy(), targets.copy(), _INPUT)
        _cls, x1, y1, x2, y2 = _rows(padded)[0]
        assert x2 - x1 == pytest.approx(share, abs=0.04)
        assert y2 - y1 == pytest.approx(share, abs=0.04)


def test_the_margin_defaults_to_a_tenth_and_a_negative_one_is_refused():
    assert _transform(zoom_prob=0.5).zoom_margin == 0.1
    with pytest.raises(ValueError):
        _transform(zoom_prob=0.5, zoom_margin=-0.1)


def test_zoom_probability_is_a_probability():
    img, targets = _scene()
    _, plain = _transform()(img.copy(), targets.copy(), _INPUT)
    random.seed(0)
    transform = _transform(zoom_prob=0.5, zoom_range=(3.0, 3.0))
    zoomed = sum(
        not np.allclose(transform(img.copy(), targets.copy(), _INPUT)[1], plain)
        for _ in range(200)
    )
    assert 70 <= zoomed <= 130


def test_zoom_reaches_a_background_image():
    img = np.zeros((360, 200, 3), dtype=np.uint8)
    img[:, 100:] = 200  # a vertical edge: a crop narrower than the frame moves it
    empty = np.zeros((0, 5), dtype=np.float32)
    plain, _ = _transform()(img.copy(), empty, _INPUT)
    random.seed(1)
    outs = [
        _transform(zoom_prob=1.0, zoom_range=(4.0, 4.0))(img.copy(), empty, _INPUT)
        for _ in range(10)
    ]
    assert any(not np.array_equal(out, plain) for out, _ in outs)
    assert all((labels[:, 0] == -1).all() for _, labels in outs)


def test_zoom_skips_angle_targets_with_one_warning(caplog):
    img, targets = _scene()
    obb = np.hstack([targets, np.array([[0.3]], dtype=np.float32)])
    _, plain = _transform()(img.copy(), obb.copy(), _INPUT)
    transform = _transform(zoom_prob=1.0, zoom_range=(3.0, 3.0))
    with caplog.at_level(logging.WARNING, logger="libreyolo.data.augment.yolo9"):
        _, first = transform(img.copy(), obb.copy(), _INPUT)
        _, second = transform(img.copy(), obb.copy(), _INPUT)
    assert np.allclose(first, plain) and np.allclose(second, plain)
    assert sum("zoom is set" in record.message for record in caplog.records) == 1


def test_zoom_skips_segments(caplog):
    img, targets = _scene()
    ring = np.array([[80, 150], [110, 150], [110, 190], [80, 190]], dtype=np.float32)
    plain = _transform()(img.copy(), targets.copy(), _INPUT, segments=[[ring.copy()]])
    with caplog.at_level(logging.WARNING, logger="libreyolo.data.augment.yolo9"):
        zoomed = _transform(zoom_prob=1.0, zoom_range=(3.0, 3.0))(
            img.copy(), targets.copy(), _INPUT, segments=[[ring.copy()]]
        )
    assert np.allclose(zoomed[1], plain[1])
    assert np.array_equal(zoomed[2], plain[2])


def _dataset(tmp_path, transform):
    """One 360x200 frame on disk, read through the real dataset and wrapper."""
    img, targets = _scene()
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "frame.png"), img)
    x1, y1, x2, y2 = targets[0, :4]
    height, width = img.shape[:2]
    (tmp_path / "labels" / "frame.txt").write_text(
        f"0 {(x1 + x2) / 2 / width} {(y1 + y2) / 2 / height} "
        f"{(x2 - x1) / width} {(y2 - y1) / height}\n"
    )
    frames = YOLODataset(
        img_files=[str(tmp_path / "images" / "frame.png")],
        label_files=[str(tmp_path / "labels" / "frame.txt")],
        img_size=_INPUT,
        preproc=transform,
    )
    return YOLO9MosaicMixupDataset(frames, _INPUT, mosaic=False, preproc=transform)


def test_the_dataset_hands_the_zoom_the_source_pixels(tmp_path):
    """The point of the knob: a 4x zoom of a 360 px frame into 64 px is a
    downscale of real pixels, not a blow-up of the 64 px resize."""
    plain_dir, zoom_dir = tmp_path / "plain", tmp_path / "zoom"
    plain_dir.mkdir()
    zoom_dir.mkdir()
    _, plain = _dataset(plain_dir, _transform())[0][:2]
    plain_w = (_rows(plain)[0, 3] - _rows(plain)[0, 1]) * _INPUT[1]

    zoomed_set = _dataset(zoom_dir, _transform(zoom_prob=1.0, zoom_range=(4.0, 4.0)))
    seen = zoomed_set.dataset.pull_item(0)[0]
    assert seen.shape[:2] == (360, 200), "the transform was handed the resized frame"
    random.seed(3)
    _, zoomed = zoomed_set[0][:2]
    rows = _rows(zoomed)
    assert len(rows) == 1
    assert (rows[0, 3] - rows[0, 1]) * _INPUT[1] == pytest.approx(4 * plain_w, rel=0.06)
