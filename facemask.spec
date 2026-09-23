# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-folder build with two entry points sharing one folder:
#   遮脸工具.exe  (window, no console)   facemask-cli.exe  (console, for batch scripts)
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH)
datas = [(str(p), "models") for p in (root / "models").glob("*.onnx")]
datas += [(str(p), "bin") for p in (root / "bin").glob("*") if p.is_file()]
datas += collect_data_files("rtmlib")
hidden = collect_submodules("rtmlib") + ["onnxruntime", "cv2", "numpy"]

a = Analysis(["run.py"], pathex=[str(root)], datas=datas, hiddenimports=hidden, excludes=["torch", "torchvision", "matplotlib", "PIL.ImageQt"])
pyz = PYZ(a.pure)
exe_gui = EXE(pyz, a.scripts, exclude_binaries=True, name="遮脸工具", console=False, icon=None)
exe_cli = EXE(pyz, a.scripts, exclude_binaries=True, name="facemask-cli", console=True, icon=None)
coll = COLLECT(exe_gui, exe_cli, a.binaries, a.datas, name="facemask")
