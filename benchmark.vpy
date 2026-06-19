# =============================================================================
# 视频超分基准测试脚本 (VapourSynth)
# 用法: uv run vspipe benchmark.vpy -c y4m --progress .
#       或指定输入文件: vspipe benchmark.vpy -c y4m --progress . -a "video=test.mp4"
# 输出: 终端显示每帧处理时间和总 fps
# =============================================================================
import vapoursynth as vs
import sys

core = vs.core

# --- 配置 ---
VIDEO_PATH = "test_input.mp4"     # 测试视频路径
ENGINE_PATH = "video_upscale_project/models/RealESRGAN_x4plus.engine"
USE_GPU = 0                       # GPU 设备编号
NUM_STREAMS = 3                   # CUDA stream 数量 (影响并行度)

# --- 加载视频 ---
HAS_VIDEO = True
try:
    clip = core.ffms2.Source(VIDEO_PATH)
except:
    HAS_VIDEO = False
    # 无测试视频: 生成彩条代替 (用于验证环境)
    core.std.BlankClip(
        width=1920, height=1080, format=vs.YUV420P8,
        length=100, fpsnum=24, fpsden=1, color=[128, 128, 128]
    ).set_output()

if HAS_VIDEO:
    # --- 超分处理 ---
    clip = core.trt.Model(
        clip,
        engine_path=ENGINE_PATH,
        num_streams=NUM_STREAMS,
        device_id=USE_GPU,
    )

    # --- 输出 ---
    clip.set_output()

# 性能数据由 vspipe --progress 参数输出:
# "Output N frames in X.XX seconds (Y.YY fps)"
# 无视频时仅输出 BlankClip 用于环境验证
