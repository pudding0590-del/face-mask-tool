"""Build-machine self test: make a short synthetic clip with ffmpeg, run the whole pipeline on it
(no people -> no masks), and check that export + verification pass. Exercises ffmpeg, ffprobe,
onnxruntime and the model files exactly as the packaged app will use them."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facemask import media
from facemask.engine import PoseEngine
from facemask.pipeline import process_video

tmp = Path(tempfile.mkdtemp())
clip = tmp / "测试 中文 路径.mp4"
subprocess.run([media.executable("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(clip)], check=True)
engine = PoseEngine("rtmo-m", download=False)
report = process_video(clip, tmp / "out", engine, "smart")
assert all(report["verification"]["checks"].values()), report["verification"]["checks"]
assert report["frames"] == 60, report["frames"]
print("smoke test ok:", report["output"], "frames", report["frames"], "masks", report["frames_with_mask"],
      "pose s", report["seconds_pose"], "total s", report["seconds_total"])
