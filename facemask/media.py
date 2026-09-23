"""FFmpeg-based probe / decode / render / verify.

Ported from the earlier prototype's media.py (same timestamp-preserving export and the same
post-export checks), with two changes: videos with a rotation tag are accepted (frames are
handled in display orientation and the export is written upright), and any audio codec is
copied as-is instead of AAC only.
"""
import hashlib
import json
import math
import shutil
import subprocess
import sys
import uuid
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def bundle_root():
    """Folder that holds bin/ and models/: the PyInstaller bundle dir, or the project root when run from source."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def executable(name):
    root = bundle_root() / "bin"
    for candidate in (root / (name + ".exe"), root / name):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"未找到 {name}：请把 ffmpeg.exe / ffprobe.exe 放进程序目录的 bin 文件夹，或安装 FFmpeg。")
    return found


def _run(cmd, **kw):
    return subprocess.run(cmd, creationflags=_NO_WINDOW, **kw)


def probe(path, *args):
    result = _run([executable("ffprobe"), "-v", "error", *args, "-of", "json", str(path)],
                  capture_output=True, check=True)
    return json.loads(result.stdout)


def metadata(path):
    return probe(path, "-show_streams", "-show_format")


def frame_times(path):
    return probe(path, "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=pts,duration")["frames"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rotation(video_stream):
    """Rotation tag in degrees (0/90/180/270) from the display matrix or the legacy 'rotate' tag."""
    value = 0
    for side in video_stream.get("side_data_list", []):
        if "rotation" in side:
            value = float(side["rotation"])
    if not value:
        value = float(video_stream.get("tags", {}).get("rotate", 0) or 0)
    return int(round(value)) % 360


def display_size(video_stream):
    """Width/height as displayed (ffmpeg auto-rotates decoded frames to this size)."""
    w, h = video_stream["width"], video_stream["height"]
    return (h, w) if rotation(video_stream) in (90, 270) else (w, h)


def preflight(path):
    meta = metadata(path)
    video = [s for s in meta["streams"] if s["codec_type"] == "video"]
    audio = [s for s in meta["streams"] if s["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) > 1:
        raise ValueError("只支持 1 路视频和至多 1 路音频的普通成片。")
    v = video[0]
    if v.get("color_transfer") in ("smpte2084", "arib-std-b67") or "10" in v.get("pix_fmt", ""):
        raise ValueError("这是 HDR / 10 位视频，本版本暂不处理：请先导出为普通 SDR 1080p 再试。")
    w, h = display_size(v)
    if w % 2 or h % 2:
        raise ValueError("视频宽高必须是偶数。")
    if "duration_ts" not in v or any("duration_ts" not in a for a in audio):
        raise ValueError("视频缺少可靠时长，无法做保时长导出。")
    times = frame_times(path)
    if not times or any("pts" not in x for x in times):
        raise ValueError("视频缺少画面时间戳。")
    return meta, times


def decode_frames(path, width, height, log=None):
    """Yield BGR frames in display orientation, one per source frame (timestamps passed through)."""
    cmd = [executable("ffmpeg"), "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    size = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL, creationflags=_NO_WINDOW)
    try:
        while True:
            raw = proc.stdout.read(size)
            if not raw:
                break
            if len(raw) != size:
                raise RuntimeError("解码帧不完整。")
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        if proc.wait() != 0:
            raise RuntimeError("视频解码失败。")
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def timing_expression(times, step):
    expression = f"{step}*N"
    previous = 0
    for n, frame in enumerate(times):
        offset = frame["pts"] - step * n
        if offset != previous:
            expression += f"+{offset-previous}*gte(N\\,{n})"
            previous = offset
    if len(expression) > 20000:
        raise ValueError("此视频的可变帧率过于复杂，请先转换为固定帧率。")
    return expression


def render(source, output, frames, meta, times, on_frame=None, progress=None):
    """Decode the source, paint every polygon in frames[n]['bars'] black on frame n, and re-encode with the
    original timestamps, audio copied bit-for-bit. Never overwrites; verifies the result before renaming it."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError("输出文件已存在；程序不会覆盖原片或已有结果。")
    if len(frames) != len(times):
        raise ValueError("分析帧数与源视频不符，停止导出。")
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
    w, h = display_size(v)
    tb = Fraction(v["time_base"])
    if tb.numerator != 1:
        raise ValueError("当前时间基尚未验证。")
    scale = tb.denominator
    fps = Fraction(v["r_frame_rate"])
    step = round(scale / fps)
    expression = timing_expression(times, step)
    last = v.get("start_pts", 0) + v["duration_ts"] - times[-1]["pts"]
    if last <= 0:
        raise ValueError("源视频尾帧时长无效。")
    duration = f"setts=duration=if(eq(PTS\\,{times[-1]['pts']})\\,{last}\\,{step})"
    audio_args, movie_scale = [], scale
    if a:
        packets = probe(source, "-select_streams", "a:0", "-show_packets", "-show_entries", "packet=pts,duration")["packets"]
        last_pts = packets[-1]["pts"]
        last_duration = a.get("start_pts", 0) + a["duration_ts"] - last_pts
        if last_duration <= 0:
            raise ValueError("源音频尾包时长无效。")
        audio_args = ["-bsf:a", f"setts=duration=if(eq(PTS\\,{last_pts})\\,{last_duration}\\,DURATION)"]
        movie_scale = math.lcm(scale, Fraction(a["time_base"]).denominator)
    if movie_scale > 2_000_000_000:
        raise ValueError("当前音视频时间基组合尚未验证。")
    color = []
    for argument, key in (("-color_primaries", "color_primaries"), ("-color_trc", "color_transfer"), ("-colorspace", "color_space")):
        if v.get(key) and v[key] != "unknown":
            color.extend([argument, v[key]])
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.stem + ".pending-" + uuid.uuid4().hex[:8] + ".mp4")
    encode = [executable("ffmpeg"), "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
              "-r", str(fps), "-i", "pipe:0", "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
              "-vf", f"settb=1/{scale},setpts=" + expression, "-fps_mode", "passthrough", "-enc_time_base", f"1/{scale}",
              "-video_track_timescale", str(scale), "-c:v", "libx264", "-bf", "0", "-bsf:v", duration,
              "-preset", "fast", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p", *color,
              "-c:a", "copy", *audio_args, "-movie_timescale", str(movie_scale), "-map_metadata", "-1",
              "-movflags", "+faststart", str(pending)]
    with output.with_suffix(".encode.log").open("wb") as log:
        encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=log, creationflags=_NO_WINDOW)
        count = 0
        try:
            for frame in decode_frames(source, w, h, log):
                if count >= len(frames):
                    raise RuntimeError("视频解码帧数多于分析帧数。")
                frame = frame.copy()
                for bar in frames[count]["bars"]:
                    cv2.fillConvexPoly(frame, np.asarray(bar["polygon"], np.int32), (0, 0, 0))
                if on_frame:
                    on_frame(count, frame)
                encoder.stdin.write(frame.tobytes())
                count += 1
                if progress and count % 60 == 0:
                    progress(count, len(frames))
            encoder.stdin.close()
            if encoder.wait() or count != len(frames):
                raise RuntimeError("编码失败，请查看 .encode.log。未作为成功结果输出。")
        except BaseException:
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
            raise
        finally:
            if not encoder.stdin.closed:
                encoder.stdin.close()
    verification = verify(source, pending, times, frames)
    if output.exists():
        raise FileExistsError(output)
    pending.rename(output)
    verification["output"] = str(output)
    verification["output_sha256"] = sha256(output)
    return verification


def audio_hash(path):
    result = _run([executable("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "data", "pipe:1"],
                  capture_output=True, check=True)
    return hashlib.sha256(result.stdout).hexdigest()


def verify(source, output, times, masks):
    src, dst = metadata(source), metadata(output)
    v = next(s for s in src["streams"] if s["codec_type"] == "video")
    ov = next(s for s in dst["streams"] if s["codec_type"] == "video")
    out_times = frame_times(output)
    checks = {
        "dimensions_match": display_size(v) == (ov["width"], ov["height"]),
        "frame_timestamps_match": [f["pts"] for f in times] == [f["pts"] for f in out_times] and v["time_base"] == ov["time_base"],
        "video_duration_match": v["duration_ts"] == ov["duration_ts"],
        "container_duration_match": src["format"]["duration"] == dst["format"]["duration"],
    }
    audios = [s for s in src["streams"] if s["codec_type"] == "audio"]
    out_audios = [s for s in dst["streams"] if s["codec_type"] == "audio"]
    checks["audio_stream_count_match"] = len(audios) == len(out_audios)
    if audios and out_audios:
        checks["audio_packets_unchanged"] = audio_hash(source) == audio_hash(output)
        checks["audio_duration_match"] = audios[0]["duration_ts"] == out_audios[0]["duration_ts"] and audios[0]["time_base"] == out_audios[0]["time_base"]
    bar_check = check_black_pixels(output, masks, ov["width"], ov["height"])
    checks["decodes_without_error"] = bar_check["decoded_frames"] == len(times)
    checks["rendered_masks_are_black"] = bar_check["max_inner_pixel_mean"] < 8
    if not all(checks.values()):
        raise RuntimeError(f"输出验证未通过：{checks}")
    return {"checks": checks, "frames": len(times), "mask_pixels": bar_check,
            "notice": "媒体检查只证明遮罩已按数据写入成片，不证明每一帧的脸都被遮住；请回看成片。"}


def check_black_pixels(path, masks, width, height):
    count, checked, worst = 0, 0, 0.0
    for frame in decode_frames(path, width, height):
        if count >= len(masks):
            raise RuntimeError("成片解码帧数异常")
        for bar in masks[count]["bars"]:
            points = np.array(bar["polygon"], np.int32)
            x, y, w, h = cv2.boundingRect(points)
            x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mask, points - [x0, y0], 255)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if np.any(mask):
                worst = max(worst, float(frame[y0:y1, x0:x1][mask > 0].mean()))
                checked += 1
        count += 1
    return {"decoded_frames": count, "mask_regions_checked": checked, "max_inner_pixel_mean": worst}
