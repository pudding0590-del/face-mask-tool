"""Command line batch: python -m facemask <video or folder>... [--output DIR] [--mode smart|head|face] [--engine rtmo-m|rtmo-s]"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, masks
from .engine import ENGINES, PoseEngine
from .pipeline import find_videos, process_video


def run_batch(inputs, output, mode="smart", engine_name="rtmo-m", download=True, log=print, progress=None, stop=None):
    """Process every video under inputs into output/. Returns the job list (also written to output/任务清单.json)."""
    sources = find_videos(inputs)
    if not sources:
        raise SystemExit("未找到视频（支持 mp4 / mov / m4v）。")
    output = Path(output).resolve()
    if any(output == s.parent or output in s.parents for s in sources):
        raise SystemExit("输出目录不能是待处理视频所在的目录，请另选一个文件夹。")
    output.mkdir(parents=True, exist_ok=True)
    logfile = (output / "处理日志.txt").open("a", encoding="utf-8")
    _log = log

    def log(text):   # every line goes to the screen and to 处理日志.txt in the output folder
        _log(text)
        logfile.write(time.strftime("%H:%M:%S ") + text + "\n")
        logfile.flush()
    log(f"facemask {__version__}  引擎 {ENGINES[engine_name]['label']}  模式 {masks.MODES[mode]}  共 {len(sources)} 条")
    engine = PoseEngine(engine_name, download=download)
    jobs, failed = [], 0
    for n, source in enumerate(sources, 1):
        if stop and stop():
            log("已停止。")
            break
        job = {"source": str(source), "status": "running"}
        t = time.perf_counter()
        try:
            def say(stage, i, total):
                if progress:
                    progress(n, len(sources), stage, i, total)
            report = process_video(source, output, engine, mode, progress=say)
            job.update(status="done", output=report["output"], seconds=report["seconds_total"], notes=len(report["review_notes"]))
            log(f"[{n}/{len(sources)}] 完成 {source.name} → {Path(report['output']).name}  用时 {report['seconds_total']:.0f}s  待确认提示 {len(report['review_notes'])} 条")
        except FileExistsError as error:
            job.update(status="skipped", error=str(error))
            log(f"[{n}/{len(sources)}] 跳过 {source.name}：{error}")
        except Exception as error:   # keep going with the rest of the batch
            failed += 1
            job.update(status="failed", error=f"{type(error).__name__}: {error}", seconds=round(time.perf_counter() - t, 1))
            log(f"[{n}/{len(sources)}] 失败 {source.name}：{error}")
        jobs.append(job)
        (output / "任务清单.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"结束：完成 {sum(j['status'] == 'done' for j in jobs)}，跳过 {sum(j['status'] == 'skipped' for j in jobs)}，失败 {failed}。"
        f"  成片在 {output}。自动遮罩不是验收，请回看成片并核对“可疑片段.txt”。")
    logfile.close()
    return jobs


def diagnose():
    """One screen of environment facts to paste back when something goes wrong."""
    import os
    import platform
    import subprocess
    from .media import bundle_root, executable
    import onnxruntime
    lines = [f"facemask {__version__}", f"python {platform.python_version()} {platform.architecture()[0]}",
             f"os {platform.platform()}", f"cpu {platform.processor() or '?'}  逻辑核 {os.cpu_count()}",
             f"onnxruntime {onnxruntime.__version__} providers {onnxruntime.get_available_providers()}",
             f"bundle_root {bundle_root()}"]
    for name, spec in ENGINES.items():
        path = bundle_root() / "models" / spec["file"]
        lines.append(f"model {name}: {'ok' if path.is_file() else '缺失'} {path}")
    for tool in ("ffmpeg", "ffprobe"):
        try:
            exe = executable(tool)
            ver = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
            lines.append(f"{tool}: {exe}  {ver}")
        except Exception as error:
            lines.append(f"{tool}: 不可用 {error}")
    return "\n".join(lines)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):   # Windows consoles may not be UTF-8; never crash on a Chinese file name
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    p = argparse.ArgumentParser(prog="facemask", description="批量给视频人物脸部打黑色椭圆遮罩（离线、本机处理，原片只读）。")
    p.add_argument("inputs", nargs="+", help="视频文件或文件夹（可多个，含子文件夹；支持中文路径）")
    p.add_argument("--output", "-o", help="输出文件夹（默认：第一个输入旁边的“黑条版”）")
    p.add_argument("--mode", choices=list(masks.MODES), default="smart", help="smart=除纯背面（默认） head=含背面 face=仅正侧脸")
    p.add_argument("--engine", choices=list(ENGINES), default="rtmo-m", help="rtmo-m=标准（默认） rtmo-s=快速")
    p.add_argument("--no-download", action="store_true", help="缺模型时直接报错，不联网下载")
    p.add_argument("--diagnose", action="store_true", help="只打印环境自检信息，不处理视频")
    if argv is None and "--diagnose" in sys.argv:
        print(diagnose())
        return 0
    a = p.parse_args(argv)
    first = Path(a.inputs[0]).resolve()
    output = Path(a.output) if a.output else (first.parent / "黑条版")
    jobs = run_batch(a.inputs, output, a.mode, a.engine, download=not a.no_download)
    return 1 if any(j["status"] == "failed" for j in jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
