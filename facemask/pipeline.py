"""One video end to end: probe -> decode -> pose -> masks -> render -> verify -> contact sheet -> report."""
import json
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import media, masks

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}


def find_videos(inputs):
    found = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            found.extend(p for p in item.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file() and not p.name.startswith("._"))
        elif item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES:
            found.append(item)
    return sorted(set(p.resolve() for p in found))


class ContactSheet:
    """One tile per second from the rendered frames, so a reviewer can eyeball a clip in one image."""

    def __init__(self, fps, tile_height=320, columns=6):
        self.every = max(1, int(round(fps)))
        self.fps = fps
        self.h = tile_height
        self.cols = columns
        self.tiles = []

    def add(self, index, frame):
        if index % self.every:
            return
        s = self.h / frame.shape[0]
        tile = cv2.resize(frame, None, fx=s, fy=s)
        cv2.rectangle(tile, (0, 0), (90, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"{index / self.fps:5.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        self.tiles.append(tile)

    def save(self, path):
        if not self.tiles:
            return None
        w = max(t.shape[1] for t in self.tiles)
        rows = []
        for i in range(0, len(self.tiles), self.cols):
            row = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in self.tiles[i:i + self.cols]]
            while len(row) < self.cols:
                row.append(np.zeros((self.h, w, 3), np.uint8))
            rows.append(np.hstack(row))
        cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(str(path))
        return path


def review_notes(segments, people_per_frame, fps):
    """Plain-language list of what to look at: mask spans, head-seen-but-not-masked spans, no-person spans."""
    lines = []
    t = lambda n: f"{n / fps:.1f}s"
    for s in segments:
        who = f"人物{s['track'] + 1}"
        for a, b in s["drawn"]:
            lines.append(f"{t(a)}–{t(b + 1)}  {who}：已遮")
        for a, b in s["seen_not_drawn"]:
            if b - a + 1 >= fps * .3:
                lines.append(f"{t(a)}–{t(b + 1)}  {who}：看到头但判定为背对/脸不可见，未遮 ← 请确认")
        for a, b in s["filled_gaps"]:
            if b - a + 1 >= fps * .2:   # single dropped frames are routine; only flag gaps a viewer could notice
                lines.append(f"{t(a)}–{t(b + 1)}  {who}：识别短暂丢失，按前后位置补上 ← 请确认")
    empty = masks._runs([len(p) == 0 for p in people_per_frame])
    for a, b in empty:
        if b - a + 1 >= fps * .5:
            lines.append(f"{t(a)}–{t(b + 1)}  画面中未识别到人 ← 如有人请确认")
    return sorted(lines, key=lambda l: float(l.split("s")[0]))


def process_video(source, out_dir, engine, mode="smart", progress=None):
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{source.stem}_遮脸.mp4"
    if out_video.exists():
        raise FileExistsError(f"已存在，跳过：{out_video.name}")
    say = progress or (lambda *a: None)
    t0 = time.perf_counter()
    meta, times = media.preflight(source)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    w, h = media.display_size(v)
    fps = float(Fraction(v["r_frame_rate"]))
    people_per_frame = []
    for i, frame in enumerate(media.decode_frames(source, w, h)):
        people_per_frame.append(engine(frame))
        if i % 30 == 0:
            say("识别", i, len(times))
    if len(people_per_frame) != len(times):
        raise RuntimeError(f"解码帧数 {len(people_per_frame)} 与时间戳数 {len(times)} 不符。")
    t1 = time.perf_counter()
    frames, segments = masks.build_masks(people_per_frame, fps, mode)
    sheet = ContactSheet(fps)
    verification = media.render(source, out_video, frames, meta, times, on_frame=sheet.add,
                                progress=lambda n, total: say("导出", n, total))
    sheet_path = sheet.save(out_dir / f"{source.stem}_对照图.jpg")
    notes = review_notes(segments, people_per_frame, fps)
    report = {
        "source": str(source), "source_sha256": media.sha256(source), "output": str(out_video),
        "engine": engine.name, "mode": mode, "fps": fps, "width": w, "height": h, "frames": len(times),
        "frames_with_mask": sum(1 for f in frames if f["bars"]), "tracks": len(segments),
        "seconds_pose": round(t1 - t0, 1), "seconds_total": round(time.perf_counter() - t0, 1),
        "verification": verification, "contact_sheet": str(sheet_path) if sheet_path else None,
        "review_notes": notes, "masks": frames, "segments": segments,
    }
    (out_dir / f"{source.stem}_检查记录.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{source.stem}_可疑片段.txt").write_text(
        "自动遮罩不是验收，请回看成片。重点看下面这些时间段：\n\n" + ("\n".join(notes) if notes else "（无提示）") + "\n", encoding="utf-8")
    return report
