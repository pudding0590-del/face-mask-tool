"""Pose engine: RTMO (one-stage, OpenMMLab) through rtmlib + onnxruntime on CPU.

Only the head/shoulder keypoints are kept.  RTMO's per-keypoint scores drop to ~0 for a face seen from
behind, which is what the "skip pure back views" mode relies on (measured 2026-09-23 on 20 clips:
93% agreement with the macOS Vision reference).

Person boxes overlapping by up to 70% are kept (rtmlib's default of 45% dropped a person standing right
behind another one - 对话1, 2.3-3.7 s); duplicate heads are merged later in masks.py.  The person-box
confidence is returned too: statues tend to get lower values than people (one of the statue cues).
"""
import shutil
from pathlib import Path

import numpy as np

from .media import bundle_root

ENGINES = {
    "rtmo-m": {
        "file": "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.onnx",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
        "input": (640, 640),
        "label": "标准（RTMO-m）",
        "mask_shape": {},
    },
    "rtmo-s": {
        "file": "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
        "input": (640, 640),
        "label": "快速（RTMO-s）",
        # less precise keypoints: a slightly larger ellipse keeps eye coverage level with RTMO-m (26 test clips:
        # 98.6% of Vision eye points fully covered at 0.44, 98.9% at 0.46; RTMO-m at 0.44: 98.8%)
        "mask_shape": {"face_a": .46, "face_b": .52},
    },
}
COCO = {0: "nose", 1: "leftEye", 2: "rightEye", 3: "leftEar", 4: "rightEar", 5: "leftShoulder", 6: "rightShoulder"}
NMS_IOU = 0.7
SCORE_MIN = 0.7


def models_dir():
    return bundle_root() / "models"


def ensure_model(name, download=True):
    """Return the local .onnx path for an engine; download it into models/ (via rtmlib's mirror-aware
    downloader) only when allowed, so the packaged app never touches the network."""
    spec = ENGINES[name]
    path = models_dir() / spec["file"]
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(f"缺少模型文件 {path.name}（应在程序目录的 models 文件夹）。")
    from rtmlib.tools.file import download_checkpoint
    cached = Path(download_checkpoint(spec["url"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, path)
    return path


class PoseEngine:
    def __init__(self, name="rtmo-m", download=True):
        from rtmlib import RTMO
        spec = ENGINES[name]
        self.name = name
        self.mask_shape = spec.get("mask_shape", {})
        self.model = RTMO(str(ensure_model(name, download)), model_input_size=spec["input"],
                          nms_thr=NMS_IOU, score_thr=SCORE_MIN, backend="onnxruntime", device="cpu")

    def __call__(self, frame_bgr):
        from rtmlib.tools.object_detection.post_processings import multiclass_nms
        image, ratio = self.model.preprocess(frame_bgr)
        det, pose = self.model.inference(image)
        boxes, box_scores = det[0, :, :4] / ratio, det[0, :, 4]
        keypoints, scores = pose[0, :, :, :2] / ratio, pose[0, :, :, 2]
        _, keep = multiclass_nms(boxes, box_scores[:, None], nms_thr=NMS_IOU, score_thr=SCORE_MIN)
        people = []
        for j in ([] if keep is None else keep):
            k, s = keypoints[j], scores[j]
            people.append({"confidence": float(np.mean(s)), "box_score": float(box_scores[j]),
                           "box": [float(x) for x in boxes[j]],
                           "keypoints": {n: [float(k[i][0]), float(k[i][1]), float(s[i])] for i, n in COCO.items()}})
        return people
