#!/bin/bash
# =============================================================================
# 视频超分环境验证脚本 v2.0
# 自动检测 CUDA/TensorRT/cuDNN 版本并验证所有组件
# 用法: ./verify.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/video_upscale_project"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

PASS=0; FAIL=0; SKIP=0

check() {
    local name="$1"; shift
    printf "  %-45s " "$name"
    if "$@" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

check_file() {
    local name="$1"; local file="$2"
    printf "  %-45s " "$name"
    if [ -f "$file" ]; then
        local size; size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}✅ PASS${NC} ($size)"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

# --- 自动检测版本 ---
detect_cuda_ver() {
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version 2>&1 | grep -oP 'release \K[\d.]+' | head -1
    else
        # 从 /usr/local/ 下的 LTS cuda 目录推断
        ls -d /usr/local/cuda-* 2>/dev/null | grep -v '/cuda$' | sort -V | tail -1 | grep -oP 'cuda-\K[\d.]+' || echo "???"
    fi
}
detect_trt_ver() {
    if command -v trtexec >/dev/null 2>&1; then
        trtexec --version 2>&1 | grep -oP 'TensorRT v?\K[\d.]+' | head -1 || echo "???"
    else
        echo "???"
    fi
}

CUDA_VER=$(detect_cuda_ver)
TRT_VER=$(detect_trt_ver)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       视频超分环境验证 v2.0                                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  检测到: CUDA ${CUDA_VER} | TensorRT ${TRT_VER}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
echo "━━━ 1. NVIDIA GPU & CUDA ━━━"
check "nvidia-smi 可用"         command -v nvidia-smi
check "nvcc (CUDA ${CUDA_VER})"  command -v nvcc

# cuDNN 检测: 可能在任何 CUDA 版本目录下的 include/lib64 中
CUDNN_H_FOUND=false
for cuda_dir in /usr/local/cuda-*; do
    [ -d "$cuda_dir" ] || continue
    if [ -f "$cuda_dir/include/cudnn_version.h" ]; then
        CUDNN_H_FOUND=true
        echo -e "  cudnn_version.h             ${GREEN}✅ PASS${NC} ($cuda_dir/include)"
        PASS=$((PASS + 1))
        break
    fi
done
$CUDNN_H_FOUND || { echo -e "  cudnn_version.h             ${RED}❌ FAIL${NC}"; FAIL=$((FAIL + 1)); }

check "libcudnn.so"               sh -c '/sbin/ldconfig -p 2>/dev/null | grep -q libcudnn'
echo ""

# ------------------------------------------------------------------
echo "━━━ 2. TensorRT ━━━"
check "libnvinfer.so"             sh -c '/sbin/ldconfig -p 2>/dev/null | grep -q libnvinfer'
check "trtexec 可用"              command -v trtexec

# 检测 trtexec 使用的是哪个版本
TRTEXEC_BIN=$(command -v trtexec 2>/dev/null || echo "")
if [ -n "$TRTEXEC_BIN" ] && [ "$TRTEXEC_BIN" != "" ]; then
    TRTEXEC_OUT=$("$TRTEXEC_BIN" --version 2>&1 | grep -oP 'TensorRT[^)]+' | head -1 || echo "")
    echo -e "  trtexec: $TRTEXEC_BIN (${TRTEXEC_OUT:-unknown})"
fi
echo ""

# ------------------------------------------------------------------
echo "━━━ 3. VapourSynth ━━━"
# libvapoursynth.so 可能装在标准路径或 /usr/local/lib/python3 下面
VS_LIB_OK=false
if ldconfig -p 2>/dev/null | grep -q libvapoursynth; then
    VS_LIB_OK=true
elif find /usr/local/lib -name 'libvapoursynth.so*' 2>/dev/null | grep -q .; then
    VS_LIB_OK=true
    VS_LIB_PATH=$(find /usr/local/lib -name 'libvapoursynth.so*' 2>/dev/null | head -1)
    echo -e "  libvapoursynth.so* ${GREEN}找到${NC} ($VS_LIB_PATH)"
fi
if $VS_LIB_OK; then
    echo -e "  libvapoursynth.so           ${GREEN}✅ PASS${NC}"
    PASS=$((PASS + 1))
else
    echo -e "  libvapoursynth.so           ${RED}❌ FAIL${NC}"
    FAIL=$((FAIL + 1))
fi

check "vspipe 可用" command -v vspipe
echo ""

# ------------------------------------------------------------------
echo "━━━ 4. Python 环境 ━━━"
check "uv"     command -v uv
check "python3" command -v python3

# 尝试激活 venv
PYTHON=""
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
fi
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python3"
elif [ -x "$PROJECT_DIR/.venv/bin/python3" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python3"
else
    PYTHON="python3"
fi

check "torch + CUDA" "$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'ok')"
check "onnx"          "$PYTHON" -c "import onnx; print('ok')"
check "spandrel"      "$PYTHON" -c "import spandrel; print('ok')"
check "vapoursynth"   "$PYTHON" -c "import vapoursynth; core=vapoursynth.core; print(f'VS r{core.version()}')"
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
ENGINE_FILE="$PROJECT_DIR/models/RealESRGAN_x4plus.engine"
if [ -f "$ENGINE_FILE" ]; then
    check_file "RealESRGAN_x4plus.engine" "$ENGINE_FILE"
else
    echo -e "  RealESRGAN_x4plus.engine   ${YELLOW}⚠ SKIP${NC} (未编译, 可手动运行 trtexec)"
    SKIP=$((SKIP + 1))
fi

WAIFU_DIR="$PROJECT_DIR/models/downloaded/waifu2x/cunet"
if [ -d "$WAIFU_DIR" ]; then
    COUNT=$(ls "$WAIFU_DIR"/*.onnx 2>/dev/null | wc -l)
    echo -e "  waifu2x cunet 模型          ${GREEN}✅ PASS${NC} ($COUNT 个)"
    PASS=$((PASS + 1))
fi
echo ""

# ------------------------------------------------------------------
echo "━━━ 7. 环境变量 ━━━"
# TensorRT 11 通过 deb 安装到系统路径，LD_LIBRARY_PATH 中无痕是正常的
if /sbin/ldconfig -p 2>/dev/null | grep -q libnvinfer; then
    echo -e "  TensorRT 库 (系统路径)       ${GREEN}✅ PASS${NC}"
    PASS=$((PASS + 1))
elif echo "${LD_LIBRARY_PATH:-}" | grep -q TensorRT; then
    echo -e "  LD_LIBRARY_PATH 含 TensorRT ${GREEN}✅ PASS${NC}"
    PASS=$((PASS + 1))
else
    echo -e "  TensorRT 库                  ${RED}❌ FAIL${NC}"
    FAIL=$((FAIL + 1))
fi

# CUDA 路径检测
CUDA_IN_PATH=false
for cuda_dir in /usr/local/cuda-*/bin; do
    if echo "${PATH:-}" | grep -q "$cuda_dir"; then
        CUDA_IN_PATH=true
        echo -e "  PATH 含 CUDA                ${GREEN}✅ PASS${NC} ($cuda_dir)"
        PASS=$((PASS + 1))
        break
    fi
done
$CUDA_IN_PATH || { echo -e "  PATH 含 CUDA                ${YELLOW}⚠ SKIP${NC} (source env.sh 后生效)"; SKIP=$((SKIP + 1)); }
echo ""

# ------------------------------------------------------------------
echo "━━━ 8. 功能自检 ━━━"
TEST_VPY=$(mktemp /tmp/vs_test_XXXXXX.vpy)
cat > "$TEST_VPY" << 'VSEOF'
import vapoursynth as vs
core = vs.core
clip = core.std.BlankClip(width=320, height=240, format=vs.RGB24, length=1, color=[255, 0, 0])
clip.set_output()
VSEOF

if command -v uv >/dev/null 2>&1 && [ -f "$PROJECT_DIR/.venv/bin/python3" ]; then
    printf "  %-45s " "uv run vspipe 执行测试"
    VS_LIB_PATHS="/usr/local/lib/python3/dist-packages/vapoursynth:/usr/local/lib"
    export LD_LIBRARY_PATH="$VS_LIB_PATHS:${LD_LIBRARY_PATH:-}"
    export VAPOURSYNTH_CONF="${VAPOURSYNTH_CONF:-$HOME/.config/vapoursynth/vapoursynth.toml}"
    if cd "$PROJECT_DIR" && uv run vspipe "$TEST_VPY" -c y4m --progress . 2>/dev/null > /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASS=$((PASS + 1))
    else
        VS_ERR=$(cd "$PROJECT_DIR" && uv run vspipe "$TEST_VPY" -c y4m --progress . 2>&1 | head -3 || true)
        echo -e "${RED}❌ FAIL${NC}"
        echo -e "                  ${RED}$VS_ERR${NC}"
        FAIL=$((FAIL + 1))
    fi
else
    echo -e "  vspipe 执行测试             ${YELLOW}⚠ SKIP${NC} (需要 uv + venv)"
    SKIP=$((SKIP + 1))
fi
rm -f "$TEST_VPY"
echo ""

# ------------------------------------------------------------------
TOTAL=$((PASS + FAIL + SKIP))
echo "╔══════════════════════════════════════════════════════════════╗"
printf "║  结果: ${GREEN}%2d 通过${NC}  ${RED}%2d 失败${NC}  ${YELLOW}%2d 跳过${NC}  (共 %2d 项)         ║\n" $PASS $FAIL $SKIP $TOTAL
echo "╚══════════════════════════════════════════════════════════════╝"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 环境验证全部通过！${NC}"
    echo ""
    echo "  用法示例:"
    echo "    source $SCRIPT_DIR/env.sh"
    echo "    vspipe your_script.vpy -c y4m - | ffmpeg -i - output.mp4"
    exit 0
else
    echo ""
    echo -e "${RED}⚠ 有 $FAIL 项检查未通过。${NC}"
    echo "  建议: source $SCRIPT_DIR/env.sh && ./verify.sh"
    exit 1
fi
