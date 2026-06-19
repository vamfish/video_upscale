# =============================================================================
# 全部引擎基准测试
# 用法: uv run vspipe benchmark/bench_all.vpy -c y4m - --progress 2>&1 1>/dev/null
# 输出: 每个引擎单独计时，最后汇总对比
# =============================================================================
import vapoursynth as vs
import sys
import time

core = vs.core
core.std.LoadPlugin("../video_upscale_project/lib/libvstrt.so")
core.std.LoadPlugin("../video_upscale_project/lib/libffms2.so")

# --- 配置 ---
VIDEO_PATH = "test_input.mp4"
GPU = 0
STREAMS = 1

ENGINES = [
    {
        "name":     "RealESRGAN x4plus",
        "engine":   "../video_upscale_project/models/RealESRGAN_x4plus.engine",
        "scale":    4,
    },
    {
        "name":     "Waifu2x cunet 2x (降噪 1)",
        "engine":   "../video_upscale_project/models/Waifu_cunet_x2n1.engine",
        "scale":    2,
    },
]

# --- 加载输入 ---
try:
    src = core.ffms2.Source(VIDEO_PATH)
    src_label = VIDEO_PATH
except:
    src = core.std.BlankClip(width=960, height=540, format=vs.RGBS,
                             length=100, fpsnum=24, fpsden=1, color=[0.5, 0.5, 0.5])
    src_label = "(540p BlankClip)"

in_w, in_h = src.width, src.height
in_frames = src.num_frames

if src.format.id != vs.RGBS:
    src = core.resize.Bicubic(src, format=vs.RGBS)

# --- 逐个引擎测试 ---
results = []
for i, eng in enumerate(ENGINES):
    clip = src
    t0 = time.time()
    clip = core.trt.Model(clip, engine_path=eng["engine"],
                          num_streams=STREAMS, device_id=GPU)
    dt = time.time() - t0

    out_w, out_h = clip.width, clip.height
    results.append((eng["name"], eng["scale"], dt, in_w, in_h, out_w, out_h))

    # 只输出最后一个引擎的视频
    if i == len(ENGINES) - 1:
        if clip.format.color_family != vs.YUV:
            clip = core.resize.Bicubic(clip, format=vs.YUV444P8, matrix_s="709")
        clip.set_output()

# --- 打印汇总 ---
print(f"", file=sys.stderr)
print(f"╔══════════════════════════════════════════════════════╗", file=sys.stderr)
print(f"║              引擎基准测试汇总                         ║", file=sys.stderr)
print(f"╠══════════════════════════════════════════════════════╣", file=sys.stderr)
print(f"║  输入: {src_label}", file=sys.stderr)
print(f"║  分辨率: {in_w}x{in_h}  |  帧数: {in_frames}", file=sys.stderr)
print(f"╠══════════════════════════════════════════════════════╣", file=sys.stderr)
for name, scale, dt, iw, ih, ow, oh in results:
    perf = in_frames / dt if dt > 0 else 0
    print(f"║  {name}", file=sys.stderr)
    print(f"║    {iw}x{ih} → {ow}x{oh} ({scale}x)  |  init: {dt:.2f}s", file=sys.stderr)
print(f"╚══════════════════════════════════════════════════════╝", file=sys.stderr)
