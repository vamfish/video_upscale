# =============================================================================
# 视频超分基准测试脚本 (VapourSynth)
# 用法:
#   环境验证:  uv run vspipe benchmark.vpy -c y4m --progress . | ffmpeg -i - -f null -
#   性能测试:  放入 test_input.mp4 后运行同上
#   输出: 显示输入/输出分辨率、帧率、处理性能
# =============================================================================
import vapoursynth as vs
import sys

core = vs.core

# --- 配置 ---
VIDEO_PATH = "test_input.mp4"                                   # 测试视频路径
ENGINE_PATH = "video_upscale_project/models/RealESRGAN_x4plus.engine"
USE_GPU = 0                                                     # GPU 设备编号
NUM_STREAMS = 3                                                 # CUDA stream 数量

# --- 加载视频或生成测试素材 ---
HAS_VIDEO = True
try:
    clip = core.ffms2.Source(VIDEO_PATH)
except:
    HAS_VIDEO = False
    # 无视频时: 用 540p BlankClip 测试引擎 (4x → 2160p)
    clip = core.std.BlankClip(
        width=960, height=540, format=vs.YUV420P8,
        length=100, fpsnum=24, fpsden=1, color=[128, 128, 128]
    )

# --- 打印输入信息 ---
in_w = clip.width
in_h = clip.height
in_fmt = clip.format.name if clip.format else "unknown"
in_fps = clip.fps_num / clip.fps_den if clip.fps_den > 0 else 0
in_frames = clip.num_frames

print(f"═══════════════════════════════════════════", file=sys.stderr)
print(f"  输入文件:   {VIDEO_PATH if HAS_VIDEO else '(无视频, 540p BlankClip 测试)'}", file=sys.stderr)
print(f"  输入分辨率: {in_w}x{in_h}  {in_fmt}", file=sys.stderr)
print(f"  输入帧率:   {in_fps:.2f} fps  ({in_frames} 帧)", file=sys.stderr)
print(f"  超分引擎:   4x (RealESRGAN_x4plus)", file=sys.stderr)

# --- 超分处理 (始终运行，包括 BlankClip) ---
clip = core.trt.Model(
    clip,
    engine_path=ENGINE_PATH,
    num_streams=NUM_STREAMS,
    device_id=USE_GPU,
)

# --- 打印输出信息 ---
out_w = clip.width
out_h = clip.height
out_fmt = clip.format.name if clip.format else "unknown"
print(f"  输出分辨率: {out_w}x{out_h}  {out_fmt}", file=sys.stderr)
print(f"  引擎文件:   {ENGINE_PATH}", file=sys.stderr)
print(f"  GPU 设备:   {USE_GPU}  |  CUDA Streams: {NUM_STREAMS}", file=sys.stderr)
print(f"═══════════════════════════════════════════", file=sys.stderr)

# --- 输出 ---
clip.set_output()
