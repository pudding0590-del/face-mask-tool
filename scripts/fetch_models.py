"""Download the pose models into models/ (used by the Windows build; run once on a dev machine too)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facemask.engine import ENGINES, ensure_model

for name in ENGINES:
    path = ensure_model(name, download=True)
    print(name, "->", path, f"{path.stat().st_size/1048576:.0f} MB")
