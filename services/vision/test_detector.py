import numpy as np
import pytest
from PIL import Image

from detector import PAD_VALUE, letterbox, nms, postprocess


def test_letterbox_keeps_aspect_ratio_and_pads_evenly():
    x, lb = letterbox(Image.new("RGB", (200, 100), (255, 0, 0)), 64)
    assert x.shape == (1, 3, 64, 64)
    assert x.dtype == np.float32
    assert lb.ratio == pytest.approx(0.32)
    assert lb.resized == (64, 32)
    assert lb.pad == (0, 16)
    # top padding is grey, centre is the (red) image
    assert x[0, :, 0, 0] == pytest.approx([PAD_VALUE / 255] * 3)
    assert x[0, :, 32, 32] == pytest.approx([1.0, 0.0, 0.0])


def test_nms_drops_overlaps_and_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, iou_thres=0.5) == [0, 2]


def make_outputs(rows, num_classes=2, size=64):
    """Build fake raw outputs. rows: (cx, cy, w, h, class scores, mask coefs[0])."""
    n_coefs = 32
    pred = np.zeros((4 + num_classes + n_coefs, len(rows)), dtype=np.float32)
    for j, (cx, cy, w, h, scores, coef) in enumerate(rows):
        pred[:4, j] = [cx, cy, w, h]
        pred[4 : 4 + num_classes, j] = scores
        pred[4 + num_classes, j] = coef
    protos = np.zeros((n_coefs, size // 4, size // 4), dtype=np.float32)
    protos[0] = 10.0  # prototype 0 "fires" everywhere: coef +1 -> mask on, -1 -> off
    return pred, protos


@pytest.fixture
def lb():
    # 200x100 photo letterboxed to 64: ratio 0.32, image occupies rows 16-48
    return letterbox(Image.new("RGB", (200, 100)), 64)[1]


def test_postprocess_decodes_boxes_masks_and_class_aware_nms(lb):
    pred, protos = make_outputs(
        [
            (32, 32, 20, 10, [0.9, 0.0], 1.0),  # A: kept
            # B: same class, overlaps A -> suppressed
            (33, 32, 20, 10, [0.8, 0.0], 1.0),
            (32, 32, 20, 10, [0.0, 0.7], -1.0),  # C: other class, same spot -> kept
            (10, 20, 4, 4, [0.1, 0.0], 1.0),  # D: below confidence threshold
        ]
    )
    dets = postprocess(pred, protos, lb, (200, 100), num_classes=2, conf_thres=0.25)

    assert [d.class_id for d in dets] == [0, 1]
    assert [round(d.confidence, 2) for d in dets] == [0.9, 0.7]

    # Input box (22, 27, 42, 37): remove 16px top padding, divide by 0.32
    assert dets[0].box == pytest.approx((68.75, 34.375, 131.25, 65.625))

    # Mask A is "on" everywhere, so after clipping to its box it fills the box exactly
    mask = dets[0].mask
    assert mask.shape == (100, 200)
    assert mask.sum() == mask[34:66, 68:132].sum() == 32 * 64
    # Mask C has negative logits -> empty
    assert dets[1].mask.sum() == 0


def test_postprocess_returns_nothing_below_threshold(lb):
    pred, protos = make_outputs([(32, 32, 20, 10, [0.1, 0.2], 1.0)])
    assert postprocess(pred, protos, lb, (200, 100), num_classes=2) == []
