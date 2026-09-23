# 遮脸工具（face-mask-tool）

给短视频里的人物脸部自动打黑色椭圆遮罩。离线、本机处理，原片只读，成片另存。
方法：人体姿态模型找到每个人的头 → 按人整段平滑头部大小和位置 → 画椭圆（约 1.6 倍头宽，转头前后各多留 0.25 秒）。
验收标准（2026-09-22 定）：把脸大体挡住、**稳定挡住眼睛**；背影不遮。

## 同事怎么用（Windows）

1. 解压 `facemask-windows-x64.zip`，双击 `遮脸工具.exe`（首次运行如弹"Windows 已保护你的电脑"，点"更多信息 → 仍要运行"）。
2. 选视频文件夹、选输出文件夹，点"开始处理"。
3. 输出：`原名_遮脸.mp4`（成片）、`原名_对照图.jpg`（每秒一帧的拼图，先看这个）、`原名_可疑片段.txt`（需要重点回看的时间段）、`原名_检查记录.json`。
4. **自动遮罩不是验收**：请回看成片，重点看"可疑片段"里的时间段。

遮挡范围三选一：除纯背面（默认，看不到任何脸部关键点才不遮）／含背面／仅正侧脸。速度两档：标准 RTMO-m、快速 RTMO-s。

命令行：`facemask-cli.exe 视频或文件夹 [--output 输出目录] [--mode smart|head|face] [--engine rtmo-m|rtmo-s]`

## 输入要求

MP4/MOV（H.264/HEVC 8 位 SDR），竖屏横屏均可，带旋转标记也可；可变帧率可；音轨原样保留。HDR/10 位视频暂不处理（先导出为 SDR）。中文路径可。

## 开发

```
pip install -r requirements.txt
python scripts/fetch_models.py     # 下载模型到 models/
python -m facemask 视频或文件夹 --output 输出目录
python scripts/smoke_test.py       # 自测
```

Windows 安装包由 GitHub Actions 自动构建（`.github/workflows/build-windows.yml`）：装依赖 → 下模型 → 下 FFmpeg → 自测 → PyInstaller 打包 → 产物 `facemask-windows-x64.zip`。

## 第三方组件与许可证

Python（PSF）、numpy（BSD）、OpenCV（Apache-2.0）、onnxruntime（MIT）、rtmlib 与 RTMO 模型（OpenMMLab，Apache-2.0；模型训练数据集各有研究用途条款）、FFmpeg（作为独立程序调用，使用 BtbN 的 GPL 构建，许可证随 bin/ 附带）、PyInstaller（GPL 带例外条款，打出的程序不受约束）。
