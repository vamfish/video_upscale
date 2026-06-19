# =============================================================================
# Waifu2x cunet 2x 降噪 + 超分 基准测试
# 用法: uv run vspipe benchmark/bench_waifu.vpy -c y4m - --progress 2>&1 1>/dev/null
# =============================================================================
import vapoursynth as vs
import sys

core = vs.core
core.std.LoadPlugin("../video_upscale_project/lib/libvstrt.so")
core.std.LoadPlugin("../video_upscale_project/lib/libffms2.so")

# --- 配置 ---
VIDEO_PATH  = "test_input.mp4"
ENGINE_PATH = "../video_upscale_project/models/Waifu_cunet_x2n1.engine"
ENGINE_NAME = "Waifu2x cunet 2x (降噪 level 1)"
SCALE       = 2
GPU         = 0
STREAMS     = 1

# --- 加载输入 ---
try:
    clip = core.ffms2.Source(VIDEO_PATH)
    src_label = VIDEO_PATH
except:
    clip = core.std.BlankClip(width=960, height=540, format=vs.RGBS,
                              length=100, fpsnum=24, fpsden=1, color=[0.5, 0.5, 0.5])
    src_label = "(540p BlankClip)"

in_w, in_h = clip.width, clip.height
in_frames  = clip.num_frames

# --- RGB float32 ---
if clip.format.id != vs.RGBS:
    clip = core.resize.Bicubic(clip, format=vs.RGBS)

# --- 超分 ---
clip = core.trt.Model(clip, engine_path=ENGINE_PATH, num_streams=STREAMS, device_id=GPU)
out_w, out_h = clip.width, clip.height

# --- 打印信息 ---
print(f"═══════════════════════════════════════════", file=sys.stderr)
print(f"  引擎:       {ENGINE_NAME} ({SCALE}x)", file=sys.stderr)
print(f"  输入:       {src_label}", file=sys.stderr)
print(f"  分辨率:     {in_w}x{in_h} → {out_w}x{out_h}", file=sys.stderr)
print(f"  帧数:       {in_frames}", file=sys.stderr)
print(f"  GPU:        {GPU}  |  Streams: {STREAMS}", file=sys.stderr)
print(f"═══════════════════════════════════════════", file=sys.stderr)

# --- 输出 ---
if clip.format.color_family != vs.YUV:
    clip = core.resize.Bicubic(clip, format=vs.YUV444P8, matrix_s="709")
clip.set_output()
