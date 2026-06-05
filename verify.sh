#!/bin/bash
# =============================================================================
# 视频超分环境验证脚本
# 检查部署是否完成、所有组件是否正常工作
# 用法: ./verify.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/video_upscale_project"
TENSORRT_DIR="$SCRIPT_DIR/ai_libs/TensorRT-8.6.1.6"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

PASS=0; FAIL=0; SKIP=0

check() {
    local name="$1"; shift
    echo -n "  $name ... "
    if "$@" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

check_cmd() {
    local name="$1"; local cmd="$2"
    echo -n "  $name ... "
    if command -v "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC} ($(which "$cmd"))"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC} (命令未找到)"
        FAIL=$((FAIL + 1))
    fi
}

check_file() {
    local name="$1"; local file="$2"
    echo -n "  $name ... "
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}✅ PASS${NC} ($size)"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC} (文件不存在: $file)"
        FAIL=$((FAIL + 1))
    fi
}

check_lib() {
    local name="$1"; local lib="$2"
    echo -n "  $name ... "
    if ldconfig -p 2>/dev/null | grep -q "$lib"; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       视频超分环境验证                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
echo "━━━ 1. NVIDIA GPU & CUDA ━━━"
check_cmd "nvidia-smi"       nvidia-smi
check_cmd "nvcc (CUDA 12.2)" /usr/local/cuda-12.2/bin/nvcc
check_file "cudnn_version.h" /usr/local/cuda-12.2/include/cudnn_version.h
check_lib  "libcudnn.so"     libcudnn
echo ""

# ------------------------------------------------------------------
echo "━━━ 2. TensorRT ━━━"
check_file "libnvinfer.so" "$TENSORRT_DIR/lib/libnvinfer.so"
check_cmd  "trtexec"        "$TENSORRT_DIR/bin/trtexec"
echo ""

# ------------------------------------------------------------------
echo "━━━ 3. VapourSynth ━━━"
check_lib  "libvapoursynth.so" libvapoursynth
check_cmd  "vspipe"            vspipe
echo ""

# ------------------------------------------------------------------
echo "━━━ 4. Python 环境 ━━━"
check_cmd  "uv"     uv
check_cmd  "python3" "$PROJECT_DIR/.venv/bin/python3"

# 激活 venv 测试 Python 依赖
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
fi

PYTHON="${VIRTUAL_ENV:-$PROJECT_DIR/.venv}/bin/python3"
if [ -x "$PYTHON" ]; then
    check "torch + CUDA" "$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
    check "onnx"          "$PYTHON" -c "import onnx; print('OK')"
    check "spandrel"      "$PYTHON" -c "import spandrel; print('OK')"
    check "vapoursynth"   "$PYTHON" -c "import vapoursynth; core = vapoursynth.core; print(f'VapourSynth {core.version()}')"
else
    echo -e "  ${YELLOW}⚠ SKIP${NC}  Python 虚拟环境不可用"
    SKIP=$((SKIP + 4))
fi
echo ""

# ------------------------------------------------------------------
echo "━━━ 5. VapourSynth 插件 ━━━"
check_file "libvstrt.so (TensorRT)" "$PROJECT_DIR/lib/libvstrt.so"
check_file "libffms2.so (视频源)"   "$PROJECT_DIR/lib/libffms2.so"
echo ""

# ------------------------------------------------------------------
echo "━━━ 6. AI 模型文件 ━━━"
check_file "RealESRGAN_x4plus.pth"   "$PROJECT_DIR/models/RealESRGAN_x4plus.pth"
check_file "RealESRGAN_x4plus.onnx"  "$PROJECT_DIR/models/RealESRGAN_x4plus.onnx"
check_file "RealESRGAN_x4plus.engine" "$PROJECT_DIR/models/RealESRGAN_x4plus.engine"

# 计数预下载的 waifu2x 模型
WAIFU_DIR="$PROJECT_DIR/models/downloaded/waifu2x/cunet"
if [ -d "$WAIFU_DIR" ]; then
    COUNT=$(ls "$WAIFU_DIR"/*.onnx 2>/dev/null | wc -l)
    echo -e "  waifu2x cunet 模型 ... ${GREEN}✅ PASS${NC} ($COUNT 个)"
    PASS=$((PASS + 1))
else
    echo -e "  waifu2x cunet 模型 ... ${YELLOW}⚠ SKIP${NC} (目录不存在)"
    SKIP=$((SKIP + 1))
fi
echo ""

# ------------------------------------------------------------------
echo "━━━ 7. 环境变量 ━━━"
check "LD_LIBRARY_PATH 含 TensorRT" grep -q "TensorRT" <<< "${LD_LIBRARY_PATH:-}"
check "PATH 含 CUDA 12.2"           grep -q "cuda-12.2" <<< "${PATH:-}"
echo ""

# ------------------------------------------------------------------
echo "━━━ 8. 功能自检 (VapourSynth 脚本测试) ━━━"
# 创建一个最小测试脚本
TEST_VPY=$(mktemp /tmp/vs_test_XXXXXX.vpy)
cat > "$TEST_VPY" << 'EOF'
import vapoursynth as vs
core = vs.core
# 生成测试彩条 (BlankClip)
clip = core.std.BlankClip(width=320, height=240, format=vs.RGB24, length=1, color=[255, 0, 0])
clip.set_output()
EOF

if vspipe --version > /dev/null 2>&1; then
    echo -n "  vspipe 执行测试 ... "
    if vspipe "$TEST_VPY" -c y4m --progress . 2>/dev/null | head -c 100 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC} (VapourSynth 脚本可正常执行)"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
else
    echo -e "  vspipe 执行测试 ... ${YELLOW}⚠ SKIP${NC} (vspipe 不可用)"
    SKIP=$((SKIP + 1))
fi
rm -f "$TEST_VPY"
echo ""

# ------------------------------------------------------------------
# 总结
# ------------------------------------------------------------------
TOTAL=$((PASS + FAIL + SKIP))
echo "╔══════════════════════════════════════════════════════════════╗"
printf "║  结果: ${GREEN}%2d 通过${NC}  ${RED}%2d 失败${NC}  ${YELLOW}%2d 跳过${NC}  (共 %2d 项)         ║\n" $PASS $FAIL $SKIP $TOTAL
echo "╚══════════════════════════════════════════════════════════════╝"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 环境验证全部通过！可以开始视频超分处理。${NC}"
    echo ""
    echo "  用法示例:"
    echo "    source $SCRIPT_DIR/env.sh"
    echo "    vspipe your_script.vpy -c y4m - | ffmpeg -i - output.mp4"
    exit 0
else
    echo ""
    echo -e "${RED}⚠ 有 $FAIL 项检查未通过，请检查上述 FAIL 项。${NC}"
    echo "  可能需要重新运行: ./setup_video_upscale.sh"
    exit 1
fi
