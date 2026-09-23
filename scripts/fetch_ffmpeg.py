"""Windows build helper: download a static FFmpeg build and place ffmpeg.exe / ffprobe.exe into bin/.
Source: BtbN/FFmpeg-Builds (GPL build; FFmpeg is called as a separate process). The license file is kept next to the binaries."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
root = Path(__file__).resolve().parents[1]
bin_dir = root / "bin"
bin_dir.mkdir(exist_ok=True)
if (bin_dir / "ffmpeg.exe").exists() and (bin_dir / "ffprobe.exe").exists():
    print("ffmpeg already present")
    sys.exit(0)
print("downloading", URL, flush=True)
data = urllib.request.urlopen(URL, timeout=600).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for member in z.namelist():
        base = member.rsplit("/", 1)[-1]
        if base in ("ffmpeg.exe", "ffprobe.exe", "LICENSE.txt"):
            (bin_dir / base).write_bytes(z.read(member))
            print("extracted", base, flush=True)
assert (bin_dir / "ffmpeg.exe").exists() and (bin_dir / "ffprobe.exe").exists()
