"""YOLO-seg inference on onnxruntime, with the post-processing written out in numpy.

Ultralytics would do all of this for us, but it pulls in PyTorch (~2 GB). The
ONNX model only gives raw tensors, so this module does the three steps
Ultralytics normally hides:

1. letterbox: resize keeping aspect ratio, pad to a square of side `imgsz`
2. decode + NMS: turn the raw prediction tensor into boxes, classes, scores
3. masks: combine each box's 32 mask coefficients with the 32 prototype masks

Model outputs for a seg model with `nc` classes at imgsz 640:
    output0: (1, 4 + nc + 32, 8400)  per anchor: cx, cy, w, h, nc class scores, 32 mask coefs
    output1: (1, 32, 160, 160)       prototype masks at 1/4 of the input resolution
"""

import ast
from dataclasses import dataclass

import numpy as np
from PIL import Image

PAD_VALUE = 114  # grey padding, same as Ultralytics training


@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    mask: np.ndarray  # bool (H, W) at original image size


@dataclass
class Letterbox:
    size: int  # side of the square model input
    ratio: float  # scale applied to the original image
    pad: tuple[int, int]  # left, top padding in model-input pixels
    resized: tuple[int, int]  # width, height of the image inside the padding


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize to fit a size x size square without distortion, padding the rest.

    Returns a (1, 3, size, size) float32 tensor in 0-1 and the transform, which
    we need later to map boxes and masks back onto the original image.
    """
    w, h = image.size
    r = min(size / w, size / h)
    new_w, new_h = round(w * r), round(h * r)
    # The -0.1 matches Ultralytics' rounding so an odd padding splits the same way.
    left = round((size - new_w) / 2 - 0.1)
    top = round((size - new_h) / 2 - 0.1)

    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    # Pillow antialiases when downscaling; Ultralytics' predict uses OpenCV
    # without it. Verified against Ultralytics on the same ONNX: same detections,
    # box IoU > 0.95 and mask IoU > 0.9 on most objects, confidences within a few points.
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], Letterbox(size, r, (left, top), (new_w, new_h))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy non-maximum suppression. Keeps the best box, drops boxes overlapping it, repeats."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(
    pred: np.ndarray,
    protos: np.ndarray,
    lb: Letterbox,
    orig_size: tuple[int, int],
    num_classes: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 100,
) -> list[Detection]:
    """Turn raw model outputs into detections on the original image.

    pred: (4 + nc + 32, N) for one image, protos: (32, mh, mw).
    """
    pred = pred.T  # (N, 4 + nc + 32): one row per anchor
    scores = pred[:, 4 : 4 + num_classes]
    class_ids = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= conf_thres
    pred, class_ids, conf = pred[keep], class_ids[keep], conf[keep]
    if not len(pred):
        return []

    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    # Class-aware NMS in one pass: shift each class's boxes far apart so boxes
    # of different classes never overlap (a dent and a scratch can share a spot).
    offsets = class_ids[:, None] * 10_000.0
    idx = nms(boxes + offsets, conf, iou_thres)[:max_det]
    boxes, class_ids, conf = boxes[idx], class_ids[idx], conf[idx]
    coefs = pred[idx, 4 + num_classes :]

    masks = _decode_masks(coefs, protos, lb, orig_size)

    # Undo the letterbox on boxes: remove padding, then undo the resize.
    w, h = orig_size
    boxes = (boxes - [lb.pad[0], lb.pad[1], lb.pad[0], lb.pad[1]]) / lb.ratio
    boxes = boxes.clip(0, [w, h, w, h])

    detections = []
    for box, cls, score, mask in zip(boxes, class_ids, conf, masks):
        x1, y1, x2, y2 = box
        # Masks come from a low-res blend, so they can bleed past the box. Clip them to it.
        cropped = np.zeros_like(mask)
        ys, xs = slice(int(y1), int(np.ceil(y2))), slice(int(x1), int(np.ceil(x2)))
        cropped[ys, xs] = mask[ys, xs]
        detections.append(
            Detection(
                int(cls),
                float(score),
                (float(x1), float(y1), float(x2), float(y2)),
                cropped,
            )
        )
    return detections


def _decode_masks(coefs, protos, lb: Letterbox, orig_size) -> list[np.ndarray]:
    """Each mask = sigmoid(coefs . prototypes), then mapped back to the original image."""
    c, mh, mw = protos.shape
    logits = (coefs @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    probs = 1 / (1 + np.exp(-logits))

    # Prototypes cover the whole letterboxed input at lower resolution (160 for 640):
    # cut out the region holding the real image, then stretch it to the original size.
    s = mw / lb.size
    left, top = lb.pad[0] * s, lb.pad[1] * s
    right, bottom = (lb.pad[0] + lb.resized[0]) * s, (lb.pad[1] + lb.resized[1]) * s
    out = []
    for p in probs:
        m = Image.fromarray(p.astype(np.float32), mode="F")
        m = m.resize(orig_size, Image.BILINEAR, box=(left, top, right, bottom))
        out.append(np.asarray(m) > 0.5)
    return out


class Detector:
    def __init__(self, model_path: str, providers: list[str] | None = None):
        import onnxruntime as ort  # imported here so tests of the pure functions don't need it

        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        # Ultralytics writes names as a Python dict literal, e.g. "{0: 'dent', 1: 'scratch'}".
        self.names: dict[int, str] = ast.literal_eval(meta["names"])
        self.imgsz = int(ast.literal_eval(meta["imgsz"])[0])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image: Image.Image, conf: float = 0.25, iou: float = 0.7):
        x, lb = letterbox(image, self.imgsz)
        pred, protos = self.session.run(None, {self.input_name: x})
        return postprocess(
            pred[0], protos[0], lb, image.size, len(self.names), conf, iou
        )
