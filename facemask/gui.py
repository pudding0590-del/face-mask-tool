"""Minimal window for colleagues: pick folders, pick mode, press start. Tkinter only (ships with Python)."""
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, masks
from .cli import run_batch
from .engine import ENGINES


def open_folder(path):
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"遮脸工具 {__version__}")
        root.geometry("640x520")
        root.minsize(560, 460)
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="smart")
        self.engine_var = tk.StringVar(value="rtmo-m")
        self.running = False
        self.stop_flag = False
        pad = {"padx": 10, "pady": 4}
        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="视频文件夹（含子文件夹，原片只读）").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.input_var).grid(row=1, column=0, sticky="we", **pad)
        ttk.Button(frame, text="选择…", command=self.pick_input).grid(row=1, column=1, **pad)
        ttk.Label(frame, text="输出文件夹（默认：视频文件夹旁边的“黑条版”）").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=3, column=0, sticky="we", **pad)
        ttk.Button(frame, text="选择…", command=self.pick_output).grid(row=3, column=1, **pad)
        opts = ttk.Frame(frame)
        opts.grid(row=4, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(opts, text="遮挡范围：").pack(side="left")
        for key, label in masks.MODES.items():
            ttk.Radiobutton(opts, text=label, value=key, variable=self.mode_var).pack(side="left", padx=4)
        eng = ttk.Frame(frame)
        eng.grid(row=5, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(eng, text="速度：").pack(side="left")
        for key, spec in ENGINES.items():
            ttk.Radiobutton(eng, text=spec["label"], value=key, variable=self.engine_var).pack(side="left", padx=4)
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="w", **pad)
        self.start_btn = ttk.Button(buttons, text="开始处理", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="停止", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(buttons, text="打开输出文件夹", command=self.open_output).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=1000)
        self.progress.grid(row=7, column=0, columnspan=2, sticky="we", **pad)
        self.status = ttk.Label(frame, text="就绪。自动遮罩不是验收，处理完请回看成片。")
        self.status.grid(row=8, column=0, columnspan=2, sticky="w", **pad)
        self.log = tk.Text(frame, height=12, wrap="word")
        self.log.grid(row=9, column=0, columnspan=2, sticky="nsew", **pad)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(9, weight=1)

    def pick_input(self):
        d = filedialog.askdirectory(title="选择视频文件夹")
        if d:
            self.input_var.set(d)
            if not self.output_var.get():
                self.output_var.set(str(Path(d).parent / "黑条版"))

    def pick_output(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.output_var.set(d)

    def open_output(self):
        out = self.output_var.get()
        if out and Path(out).exists():
            open_folder(out)
        else:
            messagebox.showinfo("提示", "输出文件夹还不存在。")

    def write(self, text):
        self.root.after(0, lambda: (self.log.insert("end", text + "\n"), self.log.see("end")))

    def on_progress(self, n, total, stage, i, frames):
        frac = (n - 1 + (i / max(1, frames)) * (0.7 if stage == "识别" else 0.3) + (0.7 if stage == "导出" else 0)) / total
        self.root.after(0, lambda: (self.progress.configure(value=int(frac * 1000)),
                                    self.status.configure(text=f"第 {n}/{total} 条  {stage} {i}/{frames} 帧")))

    def start(self):
        if self.running:
            return
        src, out = self.input_var.get().strip(), self.output_var.get().strip()
        if not src or not Path(src).exists():
            messagebox.showerror("错误", "请先选择视频文件夹。")
            return
        if not out:
            out = str(Path(src).parent / "黑条版")
            self.output_var.set(out)
        self.running, self.stop_flag = True, False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        threading.Thread(target=self.worker, args=(src, out), daemon=True).start()

    def worker(self, src, out):
        try:
            run_batch([src], out, self.mode_var.get(), self.engine_var.get(), download=False,
                      log=self.write, progress=self.on_progress, stop=lambda: self.stop_flag)
        except SystemExit as error:
            self.write(f"未开始：{error}")
        except Exception as error:
            self.write(f"出错：{type(error).__name__}: {error}")
        finally:
            self.root.after(0, self.done)

    def done(self):
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status.configure(text="处理结束。请打开输出文件夹回看成片，并核对每条的“可疑片段.txt”。")

    def stop(self):
        self.stop_flag = True
        self.status.configure(text="将在当前这条处理完后停止…")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
