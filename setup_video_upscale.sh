#!/bin/bash
# =============================================================================
# 视频超分环境一键部署脚本 v2.0 (Ubuntu LTS / WSL2)
#
# 包含: CUDA 12.2, TensorRT 8.6, cuDNN 8.9, VapourSynth, vs-mlrt, FFMS2
#       以及自动下载/转换/编译 RealESRGAN_x4plus 模型
#
# 用法:
#   ./setup_video_upscale.sh              # 完整部署
#   ./setup_video_upscale.sh --help       # 查看帮助
#   ./setup_video_upscale.sh --dry-run    # 预览将执行的步骤
#   ./setup_video_upscale.sh --skip-cuda --skip-tensorrt --skip-cudnn
#
# 可移植性:
#   将整个项目目录复制到新机器，放入 Downloads/ 中的离线包，
#   直接运行此脚本即可完成部署。
# =============================================================================

set -euo pipefail

# ============================================================================
# 路径配置 — 所有路径基于脚本所在目录，实现可移植
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="$SCRIPT_DIR/Downloads"
PROJECT_DIR="$SCRIPT_DIR/video_upscale_project"
AI_LIBS_DIR="$SCRIPT_DIR/ai_libs"
LOG_FILE="$SCRIPT_DIR/setup.log"

# ============================================================================
# 默认版本配置 — 适用于 RTX 20/30/40 系列 (SM 75-99)
# 脚本运行时会根据实际 GPU 自动检测是否需要升级
# 如果 Downloads/ 中有其他版本，会自动匹配
# ============================================================================
# 注意: TENSORRT_CUDA_COMPAT 是 TensorRT 包名中的 CUDA 版本号，
#       可能与 CUDA_VERSION 不完全一致（如 12.2 对应 compat 12.0）
CUDA_VERSION="12.2"
TENSORRT_VERSION="8.6.1.6"
TENSORRT_CUDA_COMPAT="12.0"
CUDNN_VERSION="8.9.7.29"
CUDNN_CUDA_COMPAT="12"
PYTHON_VERSION="3.12"
FFMS2_VERSION="5.0"

# GPU 架构 → TensorRT 最低版本映射
# SM >= 100 (Blackwell RTX 50xx): 需要 TensorRT 10.x + CUDA 12.6
# SM 80-99  (Ampere/Ada): 需要 TensorRT 8.0+
# SM 75-79  (Turing): 需要 TensorRT 7.0+
declare -A GPU_TRT_REQUIREMENT=(
    [cuda]=""
    [trt]=""
    [cudnn]=""
)

# 离线包文件名 — 由 _rebuild_package_names() 根据当前版本变量生成
CUDA_DEB=""
CUDA_PIN="cuda-wsl-ubuntu.pin"
TENSORRT_TAR=""
CUDNN_TAR=""
LIBTINFO5_DEB="libtinfo5_6.3-2ubuntu0.1_amd64.deb"

_rebuild_package_names() {
    # CUDA deb 文件名: cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
    #                 (文件夹部分用连字符, 版本部分用点号)
    local cuda_dash="${CUDA_VERSION//\./-}"
    CUDA_DEB="cuda-repo-wsl-ubuntu-${cuda_dash}-local_${CUDA_VERSION}.0-1_amd64.deb"
    TENSORRT_TAR="TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${TENSORRT_CUDA_COMPAT}.tar.gz"
    CUDNN_TAR="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${CUDNN_CUDA_COMPAT}-archive.tar.xz"
}
_rebuild_package_names  # 初始构建

# ============================================================================
# 命令行参数解析
# ============================================================================
SKIP_CUDA=false; SKIP_TENSORRT=false; SKIP_CUDNN=false
SKIP_VAPOURSYNTH=false; SKIP_PYTHON=false; SKIP_DEPS=false
SKIP_VSMLRT=false; SKIP_FFMS2=false; SKIP_MODELS=false
DRY_RUN=false

usage() {
    cat << 'EOF'
用法: ./setup_video_upscale.sh [选项]

选项:
  --help                 显示此帮助信息
  --skip-cuda            跳过 CUDA 安装
  --skip-tensorrt        跳过 TensorRT 安装
  --skip-cudnn           跳过 cuDNN 安装
  --skip-vapoursynth     跳过 VapourSynth 编译安装
  --skip-python          跳过 Python 环境初始化 (uv + venv)
  --skip-deps            跳过 Python 依赖安装
  --skip-vsmlrt          跳过 vs-mlrt 插件编译
  --skip-ffms2           跳过 FFMS2 插件编译
  --skip-models          跳过模型下载与转换
  --dry-run              仅显示将执行的步骤，不实际执行

目录结构要求 (脚本同级目录):
  setup_video_upscale.sh          # 本脚本
  Downloads/                      # 离线安装包 (可选，缺文件则需联网)
  video_upscale_project/          # 项目工作目录 (自动创建)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help) usage ;;
        --skip-cuda) SKIP_CUDA=true; shift ;;
        --skip-tensorrt) SKIP_TENSORRT=true; shift ;;
        --skip-cudnn) SKIP_CUDNN=true; shift ;;
        --skip-vapoursynth) SKIP_VAPOURSYNTH=true; shift ;;
        --skip-python) SKIP_PYTHON=true; shift ;;
        --skip-deps) SKIP_DEPS=true; shift ;;
        --skip-vsmlrt) SKIP_VSMLRT=true; shift ;;
        --skip-ffms2) SKIP_FFMS2=true; shift ;;
        --skip-models) SKIP_MODELS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "未知选项: $1"; usage ;;
    esac
done

# ============================================================================
# 初始化
# ============================================================================
export LD_LIBRARY_PATH=""
export PATH="$HOME/.local/bin:$PATH"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# 日志: 同时输出到终端和日志文件
# 内部的 trap '' PIPE 防止 tee 因 SIGPIPE 退出
exec > >(trap '' PIPE; tee -a "$LOG_FILE") 2>&1

info()    { echo -e "${BLUE}[INFO]${NC}    $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}    $1"; }
error()   { echo -e "${RED}[ERROR]${NC}   $1"; exit 1; }
step()    {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}▶ $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}
dry()     { if $DRY_RUN; then echo -e "${YELLOW}[DRY-RUN]${NC}  ${1:-}"; fi; }

# dry-run 模式下假设目标命令已安装，避免因缺少尚未安装的命令而提前终止
has_cmd() { $DRY_RUN || command -v "$1" &> /dev/null; }
has_deb() { dpkg -l "$1" 2>/dev/null | grep -q '^ii'; }

# run: 在 DRY_RUN 模式下仅打印命令，否则实际执行
# 注意: 不支持带管道的命令，管道请用 if/else 处理
run() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# ============================================================================
# 打印配置摘要
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       视频超分环境部署脚本 v2.0                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  项目目录:   $PROJECT_DIR"
echo "║  离线包目录: $DOWNLOAD_DIR"
echo "║  运行库目录: $AI_LIBS_DIR"
echo "║  日志文件:   $LOG_FILE"
echo "║  CUDA:       $CUDA_VERSION"
echo "║  TensorRT:   $TENSORRT_VERSION"
echo "║  cuDNN:      $CUDNN_VERSION"
echo "║  Python:     $PYTHON_VERSION"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if $DRY_RUN; then
    warn "Dry-run 模式，仅显示将执行的步骤，不实际执行"
    echo ""
fi

# ============================================================================
# 0. 系统环境检查
# ============================================================================
step "0. 系统环境检查"

IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
    info "检测到 WSL 环境"
else
    info "检测到原生 Linux 环境"
fi

if [ -f /etc/os-release ]; then
    . /etc/os-release
    info "操作系统: $NAME $VERSION"
    [[ "$NAME" != *"Ubuntu"* ]] && warn "非 Ubuntu 系统，部分步骤可能不兼容"
else
    warn "无法检测操作系统版本"
fi

if has_cmd nvidia-smi; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '未知')
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo '0.0')
    # 计算 SM 架构号: compute_cap 12.0 → 120, 8.9 → 89
    GPU_SM_MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f1)
    GPU_SM_MINOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f2)
    GPU_SM="${GPU_SM_MAJOR}${GPU_SM_MINOR}"
    info "GPU: $GPU_NAME (Compute Capability: $GPU_COMPUTE_CAP, SM: $GPU_SM)"
else
    error "未找到 nvidia-smi，请确认 NVIDIA 驱动已安装"
fi

# --------------------------------------------------------------------
# GPU → 离线包兼容性检测 (在所有安装之前执行)
# --------------------------------------------------------------------

# --- 根据 GPU SM 确定需要的版本 ---
# Blackwell (SM >= 100): 需要 TensorRT >= 10.0 → CUDA >= 12.4
# 以下为推荐的最低版本，脚本接受 Downloads/ 中任意更新版本

_recommend_min_cuda_for_sm() {
    local sm_major="${1:-0}"
    if   [ "$sm_major" -ge 10 ]; then echo "12.4";    # Blackwell → CUDA >= 12.4
    elif [ "$sm_major" -ge 9 ];  then echo "12.2";    # Ada
    else                              echo "12.2";    # Ampere/Turing
    fi
}
_recommend_min_trt_for_sm() {
    local sm_major="${1:-0}"
    if   [ "$sm_major" -ge 10 ]; then echo "10.0.0";   # Blackwell → TRT >= 10
    elif [ "$sm_major" -ge 9 ];  then echo "8.6.0";    # Ada
    elif [ "$sm_major" -ge 8 ];  then echo "8.0.0";    # Ampere
    else                              echo "7.0.0";    # Turing
    fi
}
_recommend_min_cudnn_for_cuda() {
    local cuda_major="${1:-12}"
    if   [ "$cuda_major" -ge 13 ]; then echo "9.0.0";
    elif [ "$cuda_major" -ge 12 ]; then echo "8.9.0";
    else                                 echo "8.5.0"; fi
}

REC_MIN_CUDA=$(_recommend_min_cuda_for_sm "$GPU_SM_MAJOR")
REC_MIN_TRT=$(_recommend_min_trt_for_sm "$GPU_SM_MAJOR")
REC_MIN_TRT_MAJOR=$(echo "$REC_MIN_TRT" | cut -d. -f1)

# --- 扫描 Downloads/ 中实际存在的离线包 ---
_find_download() {
    local pattern="$1"
    find "$DOWNLOAD_DIR" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | sort -V | tail -1
}

EXISTING_TRT_TAR=$(_find_download "TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz")
EXISTING_CUDA_DEB=$(_find_download "cuda-repo-wsl-ubuntu-*-local_*_amd64.deb")
EXISTING_CUDNN_TAR=$(_find_download "cudnn-linux-x86_64-*_cuda*-archive.tar.xz")

# 解析文件名的版本信息
_parse_trt_ver()   { basename "$1" | sed -E 's/TensorRT-([0-9.]+)\.Linux.*/\1/'; }
_parse_trt_cuda()  { basename "$1" | sed -E 's/.*cuda-([0-9.]+)\.tar\.gz/\1/'; }
# CUDA WSL deb 文件名如 cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
# 提取版本: 12.2 (文件夹部分用 - 替换 .)
_parse_cuda_ver()  { basename "$1" | sed -E 's/cuda-repo-wsl-ubuntu-([0-9]+)-([0-9]+)-local.*/\1.\2/'; }
_parse_cudnn_ver() { basename "$1" | sed -E 's/cudnn-linux-x86_64-([0-9.]+)_cuda.*/\1/'; }
_version_ge() {  # $1 >= $2 ?  (语义版本比较)
    local a="${1:-0}"; local b="${2:-0}"
    [ "$(printf '%s\n' "$b" "$a" | sort -V | tail -1)" = "$a" ]
}

# --- 兼容性判断: Downloads 中的包版本 >= 最低要求 ---
NEED_CUDA_UPGRADE=false; NEED_TRT_UPGRADE=false; NEED_CUDNN_UPGRADE=false

CURRENT_TRT_MAJOR=$(echo "$TENSORRT_VERSION" | cut -d. -f1)
CURRENT_CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

# Check if current (hardcoded default) meets minimum
if [ "$CURRENT_TRT_MAJOR" -lt "$REC_MIN_TRT_MAJOR" ]; then
    NEED_TRT_UPGRADE=true
    NEED_CUDA_UPGRADE=true   # TensorRT 10.x requires CUDA >= 12.4
    NEED_CUDNN_UPGRADE=true
fi

# Auto-detect compatible packages from Downloads/
HAS_COMPAT_TRT=false; HAS_COMPAT_CUDA=false; HAS_COMPAT_CUDNN=false

if [ -n "$EXISTING_TRT_TAR" ]; then
    EX_TRT_VER=$(_parse_trt_ver "$EXISTING_TRT_TAR")
    EX_TRT_MAJOR=$(echo "$EX_TRT_VER" | cut -d. -f1)
    if [ "$EX_TRT_MAJOR" -ge "$REC_MIN_TRT_MAJOR" ]; then
        HAS_COMPAT_TRT=true
        TENSORRT_VERSION="$EX_TRT_VER"
        TENSORRT_CUDA_COMPAT=$(_parse_trt_cuda "$EXISTING_TRT_TAR")
    fi
fi

if [ -n "$EXISTING_CUDA_DEB" ]; then
    EX_CUDA_VER=$(_parse_cuda_ver "$EXISTING_CUDA_DEB")
    EX_CUDA_MAJOR=$(echo "$EX_CUDA_VER" | cut -d. -f1)
    # 接受 >= REC_MIN_CUDA 的版本（12.4, 12.6, 13.0, 13.3 等）
    if _version_ge "$EX_CUDA_VER" "$REC_MIN_CUDA"; then
        HAS_COMPAT_CUDA=true
        CUDA_VERSION="$EX_CUDA_VER"
        # 更新 CUDNN_CUDA_COMPAT 以匹配实际 CUDA 大版本
        CUDNN_CUDA_COMPAT="$EX_CUDA_MAJOR"
        REC_MIN_CUDNN=$(_recommend_min_cudnn_for_cuda "$EX_CUDA_MAJOR")
    elif [ "$HAS_COMPAT_TRT" = true ]; then
        # TensorRT 已兼容但 CUDA 不满足 → 以 TensorRT 包的 CUDA compat 为准
        CUDA_VERSION="$TENSORRT_CUDA_COMPAT"
        CUDNN_CUDA_COMPAT="$(echo "$TENSORRT_CUDA_COMPAT" | cut -d. -f1)"
        HAS_COMPAT_CUDA=true  # 假设用户会下载匹配的 CUDA
        REC_MIN_CUDNN=$(_recommend_min_cudnn_for_cuda "$CUDNN_CUDA_COMPAT")
    fi
fi

REC_MIN_CUDNN="${REC_MIN_CUDNN:-$(_recommend_min_cudnn_for_cuda "$CURRENT_CUDA_MAJOR")}"

if [ -n "$EXISTING_CUDNN_TAR" ]; then
    EX_CUDNN_VER=$(_parse_cudnn_ver "$EXISTING_CUDNN_TAR")
    if _version_ge "$EX_CUDNN_VER" "${REC_MIN_CUDNN:-8.9.0}"; then
        HAS_COMPAT_CUDNN=true
        CUDNN_VERSION="$EX_CUDNN_VER"
    fi
fi

_rebuild_package_names  # 用最终确定的版本重新生成包名

# --- 输出兼容性报告 ---
info "━━━ GPU 与离线包兼容性检测 ━━━"
info "  GPU:         $GPU_NAME"
info "  Compute Cap: $GPU_COMPUTE_CAP (SM $GPU_SM)"
info ""
info "  Downloads/ 中的离线包:"
    if [ -n "$EXISTING_CUDA_DEB" ]; then
        info "    CUDA:     $(basename "$EXISTING_CUDA_DEB") (v$(_parse_cuda_ver "$EXISTING_CUDA_DEB"))"
    else
        info "    CUDA:     ❌ 未找到"
    fi
    if [ -n "$EXISTING_TRT_TAR" ]; then
        info "    TensorRT: $(basename "$EXISTING_TRT_TAR") (v$(_parse_trt_ver "$EXISTING_TRT_TAR"))"
    else
        info "    TensorRT: ❌ 未找到"
    fi
    if [ -n "$EXISTING_CUDNN_TAR" ]; then
        info "    cuDNN:    $(basename "$EXISTING_CUDNN_TAR")"
    else
        info "    cuDNN:    ❌ 未找到"
    fi
info ""
info "  GPU 最低要求 (SM $GPU_SM):"
info "    CUDA:     ≥ v$REC_MIN_CUDA  (推荐 13.3)"
info "    TensorRT: ≥ v$REC_MIN_TRT   (推荐 11.0)"
info "    cuDNN:    ≥ v${REC_MIN_CUDNN:-8.9}  (推荐匹配 CUDA 大版本)"
info ""

if $NEED_TRT_UPGRADE || $NEED_CUDA_UPGRADE; then
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    warn "⚠ 离线包与 GPU 不兼容"
    warn ""
    warn "  当前配置的包版本:"
    warn "    CUDA:     v${CUDA_VERSION}"
    warn "    TensorRT: v${TENSORRT_VERSION}"
    warn ""
    warn "  此 GPU (SM $GPU_SM) 最低要求:"
    warn "    CUDA:     ≥ v${REC_MIN_CUDA}"
    warn "    TensorRT: ≥ v${REC_MIN_TRT}"
    warn ""
    warn "  TensorRT 8.x 不支持 SM ≥ 100 (RTX 50 系列)"
    warn "  TensorRT ≥ 10.0 需要 CUDA ≥ 12.4"
    warn ""
    warn "  推荐下载最新版本 (也可用其他 >= 最低要求的版本):"
    warn "    CUDA 13.3 + TensorRT 11.0 + cuDNN 9.23  ← 推荐"
    warn "    CUDA 12.6 + TensorRT 10.7 + cuDNN 9.6    ← 备选"
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    info "━━━ 下载地址 (需 NVIDIA Developer 免费账号) ━━━"
    info "  CUDA (WSL-Ubuntu):"
    info "    https://developer.nvidia.com/cuda-downloads"
    info "    选择: Linux → x86_64 → WSL-Ubuntu → deb (local)"
    info "    可选版本: 13.3 (最新) / 12.6 (稳定)"
    info ""
    info "  TensorRT:"
    info "    https://developer.nvidia.com/tensorrt/download"
    info "    可选版本: 11.0 (最新) / 10.7 (稳定)"
    info ""
    info "  cuDNN:"
    info "    https://developer.nvidia.com/cudnn"
    info "    选择与 CUDA 大版本匹配的 cuDNN"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if $DRY_RUN; then
        warn "Dry-run: 以上兼容性问题需在实际运行前解决"
        warn "  请下载兼容包放入 Downloads/ 后运行 (不加 --dry-run)"
    else
        echo "  请选择:"
        echo "    1) 退出，我去下载兼容的离线包后重新运行"
        echo "       (推荐: CUDA 13.3 + TensorRT 11.0 + cuDNN 9.23)"
        echo "    2) 仅生成 ONNX 模型，跳过 TensorRT engine 编译"
        echo "       (ONNX 模型可用 ONNX Runtime 推理，速度较慢)"
        echo ""
        read -r -p "  请输入 [1-2] (默认 1): " _pkg_choice
        _pkg_choice="${_pkg_choice:-1}"

        case "$_pkg_choice" in
            1)
                info "已退出。下载兼容包并放入 $DOWNLOAD_DIR/ 后重新运行。"
                info "放入后无需修改脚本，脚本会自动匹配版本。"
                exit 0
                ;;
            2)
                warn "跳过 TensorRT engine 编译"
                warn "将仅生成 ONNX 模型，推理使用 ONNX Runtime"
                SKIP_TENSORRT_ENGINE=true
                ;;
            *)
                info "已退出。"
                exit 0
                ;;
        esac
    fi
else
    if [ "$HAS_COMPAT_TRT" = true ] && [ -n "$EXISTING_TRT_TAR" ]; then
        info "已匹配 Downloads/ 中的 TensorRT v$TENSORRT_VERSION ✅"
    fi
    if [ "$HAS_COMPAT_CUDA" = true ] && [ -n "$EXISTING_CUDA_DEB" ]; then
        info "已匹配 Downloads/ 中的 CUDA v$CUDA_VERSION ✅"
    fi
    if [ "$HAS_COMPAT_CUDNN" = true ] && [ -n "$EXISTING_CUDNN_TAR" ]; then
        info "已匹配 Downloads/ 中的 cuDNN v$CUDNN_VERSION ✅"
    fi
    info "所有离线包与 GPU 架构兼容 ✅"
fi

SKIP_TENSORRT_ENGINE="${SKIP_TENSORRT_ENGINE:-false}"
# --------------------------------------------------------------------

info "更新系统并安装基础依赖..."
dry "  → sudo apt update && sudo apt install (基础工具链)"
if ! $DRY_RUN; then
    sudo apt update -qq
    sudo apt install -y -qq htop build-essential autoconf libtool pkg-config \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
        git wget curl cmake python3-dev libcurl4-openssl-dev ninja-build
fi
success "系统环境检查完成"

# ============================================================================
# 1. 安装 CUDA
# ============================================================================
step "1. CUDA ${CUDA_VERSION} 安装"

if $SKIP_CUDA; then
    warn "跳过 CUDA 安装 (--skip-cuda)"
elif has_cmd nvcc && nvcc --version 2>/dev/null | grep -q "release ${CUDA_VERSION}"; then
    success "CUDA ${CUDA_VERSION} 已安装，跳过"
elif $IS_WSL; then
    info "WSL 环境，使用 WSL-Ubuntu 专用 CUDA 包"

    PIN_SRC="$DOWNLOAD_DIR/$CUDA_PIN"
    PIN_DST="/etc/apt/preferences.d/cuda-repository-pin-600"
    if [ -f "$PIN_SRC" ] && [ ! -f "$PIN_DST" ]; then
        run sudo cp "$PIN_SRC" "$PIN_DST"
        info "CUDA pin 文件已配置"
    fi

    CUDA_DEB_PATH="$DOWNLOAD_DIR/$CUDA_DEB"
    if [ -f "$CUDA_DEB_PATH" ]; then
        if ! has_deb cuda-repo-wsl-ubuntu-12-2-local; then
            run sudo dpkg -i "$CUDA_DEB_PATH"
            info "CUDA repo 已安装"
        fi
    else
        error "找不到 $CUDA_DEB_PATH，请将离线包放入 Downloads/"
    fi

    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
        run sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
    fi

    LIBTINFO5_PATH="$DOWNLOAD_DIR/$LIBTINFO5_DEB"
    if [ -f "$LIBTINFO5_PATH" ] && ! has_deb libtinfo5; then
        run sudo dpkg -i "$LIBTINFO5_PATH"
    fi

    run sudo apt-get update -qq
    run sudo apt-get install -y -qq cuda-toolkit-12-2
else
    info "原生 Ubuntu 环境，请使用 runfile 或 apt 网络安装 CUDA"
    warn "参考: https://developer.nvidia.com/cuda-downloads"
    warn "或使用 --skip-cuda 跳过此步骤"
fi

# 环境变量持久化
if ! grep -q "/usr/local/cuda-12.2/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    info "CUDA 环境变量已写入 ~/.bashrc"
fi
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
success "CUDA ${CUDA_VERSION} 配置完成"

# ============================================================================
# 2. 安装 TensorRT
# ============================================================================
step "2. TensorRT ${TENSORRT_VERSION} 安装"

TENSORRT_INSTALL_DIR="$AI_LIBS_DIR/TensorRT-${TENSORRT_VERSION}"

if $SKIP_TENSORRT; then
    warn "跳过 TensorRT 安装 (--skip-tensorrt)"
elif [ -d "$TENSORRT_INSTALL_DIR" ] && [ -f "$TENSORRT_INSTALL_DIR/lib/libnvinfer.so" ]; then
    success "TensorRT ${TENSORRT_VERSION} 已存在，跳过"
else
    TENSORRT_PATH="$DOWNLOAD_DIR/$TENSORRT_TAR"
    [ -f "$TENSORRT_PATH" ] || error "找不到 $TENSORRT_PATH，请将离线包放入 Downloads/"

    run mkdir -p "$AI_LIBS_DIR"
    info "解压 TensorRT (可能需要几分钟)..."
    run tar -xzvf "$TENSORRT_PATH" -C "$AI_LIBS_DIR" >/dev/null 2>&1
    success "TensorRT 解压完成"
fi

export LD_LIBRARY_PATH="$TENSORRT_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$TENSORRT_INSTALL_DIR/bin${PATH:+:$PATH}"

if ! grep -q "TensorRT-${TENSORRT_VERSION}" ~/.bashrc 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=\"$TENSORRT_INSTALL_DIR/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "export PATH=\"$TENSORRT_INSTALL_DIR/bin:\$PATH\"" >> ~/.bashrc
    info "TensorRT 环境变量已写入 ~/.bashrc"
fi
success "TensorRT ${TENSORRT_VERSION} 配置完成"

# ============================================================================
# 3. 安装 cuDNN
# ============================================================================
step "3. cuDNN ${CUDNN_VERSION} 安装"

if $SKIP_CUDNN; then
    warn "跳过 cuDNN 安装 (--skip-cudnn)"
elif [ -f "/usr/local/cuda-12.2/include/cudnn_version.h" ] && \
     [ -f "/usr/local/cuda-12.2/lib64/libcudnn.so" ]; then
    success "cuDNN ${CUDNN_VERSION} 已安装，跳过"
else
    CUDNN_PATH="$DOWNLOAD_DIR/$CUDNN_TAR"
    [ -f "$CUDNN_PATH" ] || error "找不到 $CUDNN_PATH，请将离线包放入 Downloads/"

    run mkdir -p "$AI_LIBS_DIR/cudnn"
    info "解压 cuDNN..."
    run tar -xf "$CUDNN_PATH" -C "$AI_LIBS_DIR/cudnn" >/dev/null 2>&1

    # 获取 tar 内顶层目录名（|| true 避免 head -1 触发 SIGPIPE → pipefail）
    if $DRY_RUN; then
        CUDNN_EXTRACT_DIR="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${CUDNN_CUDA_COMPAT}-archive"
    else
        CUDNN_EXTRACT_DIR=$(tar -tf "$CUDNN_PATH" 2>/dev/null | head -1 | cut -f1 -d"/") || true
    fi
    [ -z "$CUDNN_EXTRACT_DIR" ] && error "无法确定 cuDNN 解压目录名"

    info "安装 cuDNN 头文件和库文件..."
    run sudo cp "$AI_LIBS_DIR/cudnn/$CUDNN_EXTRACT_DIR/include/cudnn"*.h /usr/local/cuda-12.2/include/
    run sudo cp -P "$AI_LIBS_DIR/cudnn/$CUDNN_EXTRACT_DIR/lib/libcudnn"* /usr/local/cuda-12.2/lib64/
    run sudo chmod a+r /usr/local/cuda-12.2/include/cudnn*.h /usr/local/cuda-12.2/lib64/libcudnn*
fi
success "cuDNN ${CUDNN_VERSION} 配置完成"

# ============================================================================
# 4. Python 环境初始化 (uv)
# ============================================================================
step "4. Python 环境初始化 (uv + Python ${PYTHON_VERSION})"

if $SKIP_PYTHON; then
    warn "跳过 Python 环境初始化 (--skip-python)"
else
    if ! has_cmd uv; then
        info "安装 uv..."
        UV_TARBALL="$DOWNLOAD_DIR/uv-x86_64-unknown-linux-gnu.tar.gz"

        if [ -f "$UV_TARBALL" ]; then
            # 离线安装: 从 Downloads/ 中的 tar.gz 解压安装
            info "发现离线 uv 包，使用离线安装..."
            if $DRY_RUN; then
                dry "tar xzf $UV_TARBALL && cp uv ~/.local/bin/"
            else
                TMP_UV=$(mktemp -d)
                tar xzf "$UV_TARBALL" -C "$TMP_UV"
                mkdir -p "$HOME/.local/bin"
                cp "$TMP_UV"/uv-x86_64-unknown-linux-gnu/uv "$HOME/.local/bin/uv"
                cp "$TMP_UV"/uv-x86_64-unknown-linux-gnu/uvx "$HOME/.local/bin/uvx" 2>/dev/null || true
                chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" 2>/dev/null || true
                rm -rf "$TMP_UV"
                success "uv 离线安装完成: $($HOME/.local/bin/uv --version 2>/dev/null || echo 'ok')"
            fi
        else
            # 在线安装
            info "未发现离线 uv 包，尝试在线安装..."
            if $DRY_RUN; then
                dry "curl -LsSf https://astral.sh/uv/install.sh | sh"
            else
                if curl --connect-timeout 30 --max-time 300 -LsSf https://astral.sh/uv/install.sh | sh; then
                    success "uv 在线安装完成"
                else
                    warn "uv 在线安装失败（网络问题？），请手动下载并放入 Downloads/:"
                    warn "  https://github.com/astral-sh/uv/releases/download/0.11.19/uv-x86_64-unknown-linux-gnu.tar.gz"
                    warn "  保存为: $UV_TARBALL"
                    warn "  然后重新运行本脚本"
                    error "uv 安装失败 (在线下载不可用，请准备离线包)"
                fi
            fi
        fi
        export PATH="$HOME/.local/bin:$PATH"
        hash -r
        has_cmd uv || error "uv 安装失败"
    else
        info "uv 已存在: $(uv --version 2>/dev/null || echo 'unknown')"
    fi

    if ! uv python find "${PYTHON_VERSION}" &> /dev/null; then
        info "安装 Python ${PYTHON_VERSION} (via uv)..."
        run uv python install "${PYTHON_VERSION}"
    fi

    run mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"

    if [ ! -f "pyproject.toml" ]; then
        info "初始化 Python 项目..."
        run uv init --no-readme --no-workspace
    else
        warn "pyproject.toml 已存在，跳过 init"
    fi

    if [ ! -d ".venv" ]; then
        info "创建虚拟环境 (Python ${PYTHON_VERSION})..."
        run uv venv --python "${PYTHON_VERSION}"
    else
        warn "虚拟环境已存在，跳过创建"
    fi

    source .venv/bin/activate 2>/dev/null || true
    success "Python 环境初始化完成"
fi

# ============================================================================
# 5. VapourSynth 编译安装
# ============================================================================
step "5. VapourSynth 核心库编译安装"

if $SKIP_VAPOURSYNTH; then
    warn "跳过 VapourSynth 安装 (--skip-vapoursynth)"
elif ldconfig -p 2>/dev/null | grep -q libvapoursynth; then
    success "libvapoursynth.so 已安装，跳过编译"
else
    VS_SRC_DIR="$PROJECT_DIR/vapoursynth_headers"
    [ -d "$VS_SRC_DIR" ] || error "找不到 VapourSynth 源码: $VS_SRC_DIR"

    cd "$PROJECT_DIR"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    # dry-run 模式下虚拟环境未实际创建，设置占位路径避免 set -u 报错
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        VIRTUAL_ENV="$PROJECT_DIR/.venv"
    fi

    info "安装构建工具..."
    run uv pip install meson ninja cython

    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
    PYTHON_INCLUDE_DIR=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "/usr/include/python${PYTHON_VERSION}")
    PYTHON_LIB_DIR=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "/usr/lib")
    PYTHON_LIB_NAME="python${PYTHON_VERSION}"

    export PKG_CONFIG_PATH=""
    export PYTHON_CFLAGS="-I$PYTHON_INCLUDE_DIR"
    export PYTHON_LIBS="-L$PYTHON_LIB_DIR -l$PYTHON_LIB_NAME"

    cd "$VS_SRC_DIR"
    [ -d "build" ] && rm -rf build

    MESON="$VIRTUAL_ENV/bin/meson"
    NINJA="$VIRTUAL_ENV/bin/ninja"

    run $MESON setup build --prefix=/usr/local
    run $NINJA -C build -j$(nproc)
    run sudo $NINJA -C build install
    run sudo ldconfig

    cd "$PROJECT_DIR"
    success "VapourSynth 核心库编译安装完成"
fi

# ============================================================================
# 6. Python 依赖安装
# ============================================================================
step "6. Python 依赖安装"

if $SKIP_DEPS; then
    warn "跳过 Python 依赖安装 (--skip-deps)"
else
    cd "$PROJECT_DIR"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi

    info "安装核心依赖 (torch, onnx, spandrel, etc.)..."
    run uv add vapoursynth vsdpir vsrife tensorrt torch-tensorrt
    run uv add torch onnx spandrel onnxscript onnxruntime
    success "Python 依赖安装完成"
fi

# ============================================================================
# 7. vs-mlrt 插件编译
# ============================================================================
step "7. vs-mlrt (TensorRT 推理插件) 编译"

VSMLRT_SO="$PROJECT_DIR/lib/libvstrt.so"

if $SKIP_VSMLRT; then
    warn "跳过 vs-mlrt 编译 (--skip-vsmlrt)"
elif [ -f "$VSMLRT_SO" ]; then
    success "libvstrt.so 已存在，跳过编译"
else
    VS_HEADER_DIR="$PROJECT_DIR/vapoursynth_headers"

    if [ ! -d "$VS_HEADER_DIR/include" ]; then
        info "下载 VapourSynth 头文件..."
        run git clone --depth 1 https://github.com/vapoursynth/vapoursynth.git "$VS_HEADER_DIR"
    fi

    mkdir -p "$PROJECT_DIR/plugins"
    cd "$PROJECT_DIR/plugins"

    if [ ! -d "vs-mlrt" ]; then
        info "克隆 vs-mlrt 源码..."
        run git clone --recursive https://github.com/AmusementClub/vs-mlrt.git
    fi

    cd vs-mlrt/vstrt
    mkdir -p build && cd build
    rm -rf *

    info "配置 cmake..."
    CXXFLAGS="-I$VS_HEADER_DIR/include -I$VS_HEADER_DIR/include/vapoursynth -I$TENSORRT_INSTALL_DIR/include" \
    LDFLAGS="-L$TENSORRT_INSTALL_DIR/lib -L$TENSORRT_INSTALL_DIR/targets/x86_64-linux-gnu/lib" \
    # CUDA 12.2 已知的 SM 上限为 90，超过则使用 90 (驱动会 JIT 编译 PTX)
CMAKE_SM="${GPU_SM:-89}"
if [ "$CMAKE_SM" -gt 90 ]; then
    info "GPU SM ${GPU_SM} 超出 CUDA 12.2 支持范围，编译目标设为 SM 90 (运行时由驱动 JIT)"
    CMAKE_SM=90
fi
run cmake .. -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_SM" -DCMAKE_BUILD_TYPE=Release

    # 修复 TensorRT 版本兼容性
    if [ -f ../vs_tensorrt.cpp ]; then
        sed -i 's/NV_TENSORRT_VERSION >= 10100/NV_TENSORRT_MAJOR >= 10/g' ../vs_tensorrt.cpp
    fi

    info "编译 vs-mlrt..."
    run cmake --build . -j$(nproc)

    run mkdir -p "$PROJECT_DIR/lib"
    run cp libvstrt.so "$VSMLRT_SO"
    success "vs-mlrt 编译完成 → $VSMLRT_SO"
fi

# ============================================================================
# 8. FFMS2 插件编译
# ============================================================================
step "8. FFMS2 (视频源插件) 编译"

FFMS2_SO="$PROJECT_DIR/lib/libffms2.so"

if $SKIP_FFMS2; then
    warn "跳过 FFMS2 编译 (--skip-ffms2)"
elif [ -f "$FFMS2_SO" ]; then
    success "libffms2.so 已存在，跳过编译"
else
    cd "$PROJECT_DIR/plugins"

    if [ ! -d "ffms2" ]; then
        info "克隆 FFMS2 源码..."
        run git clone https://github.com/FFMS/ffms2.git
    fi

    cd ffms2
    run git checkout "${FFMS2_VERSION}"

    if [ ! -d "../../vapoursynth_headers/src/core" ]; then
        run git clone --depth 1 https://github.com/vapoursynth/vapoursynth.git ../../vapoursynth_headers
    fi

    info "配置并编译 FFMS2..."
    run ./autogen.sh
    run ./configure --enable-vapoursynth \
        VAPOURSYNTH_CFLAGS="-I$(pwd)/../../vapoursynth_headers/src/core" \
        VAPOURSYNTH_LIBS=" "
    run make -j$(nproc)

    # libtool 将编译产物放在 .libs/ 子目录下，优先查找
    # 文件可能带版本后缀如 libffms2.so.5.0.0
    SO_FILE=$(find . -name "libffms2.so*" -type f -print -quit)
    [ -z "$SO_FILE" ] && SO_FILE=$(find . -path '*/.libs/libffms2.so*' -type f -print -quit)
    if [ -n "$SO_FILE" ]; then
        run mkdir -p "$PROJECT_DIR/lib"
        run cp "$SO_FILE" "$FFMS2_SO"
        success "FFMS2 编译完成 → $FFMS2_SO"
    elif $DRY_RUN; then
        dry "libffms2.so 将在实际编译后生成"
    else
        error "未找到 libffms2.so，编译可能失败"
    fi
fi

# ============================================================================
# 9. 模型下载与转换
# ============================================================================
step "9. 模型下载与转换 (.pth → .onnx → .engine)"

if $SKIP_MODELS; then
    warn "跳过模型处理 (--skip-models)"
else
    cd "$PROJECT_DIR"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi

    # ---- RealESRGAN_x4plus: .pth 下载 ----
    PTH_FILE="$PROJECT_DIR/models/RealESRGAN_x4plus.pth"
    if [ ! -f "$PTH_FILE" ]; then
        info "下载 RealESRGAN_x4plus.pth ..."
        run mkdir -p models
        run wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/models/RealESRGAN_x4plus.pth -P models/
        success "RealESRGAN_x4plus.pth 下载完成"
    else
        info "RealESRGAN_x4plus.pth 已存在，跳过下载"
    fi

    # ---- RealESRGAN_x4plus: .pth → .onnx ----
    ONNX_FILE="$PROJECT_DIR/models/RealESRGAN_x4plus.onnx"
    if [ ! -f "$ONNX_FILE" ]; then
        info "转换 RealESRGAN_x4plus.pth → RealESRGAN_x4plus.onnx ..."

        CONVERT_SCRIPT="$PROJECT_DIR/src/convert_pth_to_onnx.py"
        if [ ! -f "$CONVERT_SCRIPT" ]; then
            mkdir -p src
            cat > "$CONVERT_SCRIPT" << 'PYEOF'
import torch
from spandrel import ModelLoader

wrapper = ModelLoader().load_from_file("models/RealESRGAN_x4plus.pth")
model = wrapper.model
model.eval()
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model,
    dummy,
    "models/RealESRGAN_x4plus.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"}
    },
    dynamo=False,
    export_params=True,
    do_constant_folding=True,
    verbose=False
)
print("✅ 转换成功: RealESRGAN_x4plus.onnx")
PYEOF
        fi

        run uv run python "$CONVERT_SCRIPT"
        success "ONNX 导出成功 → $ONNX_FILE"
    else
        warn "RealESRGAN_x4plus.onnx 已存在，跳过转换"
    fi

    # ---- RealESRGAN_x4plus: .onnx → .engine (TensorRT) ----
    ENGINE_FILE="$PROJECT_DIR/models/RealESRGAN_x4plus.engine"
    if [ ! -f "$ENGINE_FILE" ] && [ -f "$ONNX_FILE" ]; then
        if $SKIP_TENSORRT_ENGINE; then
            warn "已跳过 TensorRT engine 编译 (GPU 不兼容或用户选择跳过)"
            warn "ONNX 模型已生成，可在升级 TensorRT 后手动编译:"
            warn "  trtexec --onnx=$ONNX_FILE --saveEngine=$ENGINE_FILE --fp16"
        elif has_cmd trtexec; then
            info "编译 TensorRT 引擎 (RealESRGAN_x4plus)，可能需要几分钟..."
            mkdir -p logs
            if $DRY_RUN; then
                dry "trtexec --onnx=$ONNX_FILE --saveEngine=$ENGINE_FILE --fp16"
            elif trtexec --onnx="$ONNX_FILE" \
                --saveEngine="$ENGINE_FILE" \
                --minShapes=input:1x3x360x480 \
                --optShapes=input:1x3x540x960 \
                --maxShapes=input:1x3x1080x1920 \
                --fp16 --verbose 2>&1 | tee "logs/engine_build_real.log"; then
                success "TensorRT 引擎编译成功 → $ENGINE_FILE"
            else
                warn "TensorRT 引擎编译失败，详情见 logs/engine_build_real.log"
                warn "ONNX 模型可用 ONNX Runtime 作为替代推理引擎"
            fi
        else
            warn "trtexec 不可用，跳过 .engine 编译"
        fi
    elif [ -f "$ENGINE_FILE" ]; then
        warn "RealESRGAN_x4plus.engine 已存在，跳过编译"
    fi

    # ---- Waifu2x cunet 2x 降噪模型 (可选) ----
    WAIFU_ONNX="$PROJECT_DIR/models/downloaded/waifu2x/cunet/noise1_scale2.0x_model.onnx"
    WAIFU_ENGINE="$PROJECT_DIR/models/Waifu_cunet_x2n1.engine"
    if [ -f "$WAIFU_ONNX" ] && [ ! -f "$WAIFU_ENGINE" ]; then
        if $SKIP_TENSORRT_ENGINE; then
            warn "已跳过 Waifu TensorRT 引擎编译"
        elif has_cmd trtexec; then
            info "编译 TensorRT 引擎 (Waifu_cunet_x2n1)..."
            mkdir -p logs
            if $DRY_RUN; then
                dry "trtexec --onnx=$WAIFU_ONNX --saveEngine=$WAIFU_ENGINE --fp16"
            elif trtexec --onnx="$WAIFU_ONNX" \
                --saveEngine="$WAIFU_ENGINE" \
                --minShapes=input:1x3x360x480 \
                --optShapes=input:1x3x540x960 \
                --maxShapes=input:1x3x1080x1920 \
                --fp16 --verbose 2>&1 | tee "logs/engine_build_Waifu_cunet_x2n1.log"; then
                success "TensorRT 引擎编译成功 → $WAIFU_ENGINE"
            else
                warn "Waifu TensorRT 引擎编译失败"
            fi
        fi
    elif [ -f "$WAIFU_ENGINE" ]; then
        warn "Waifu_cunet_x2n1.engine 已存在，跳过编译"
    fi
fi

# ============================================================================
# 10. 环境持久化与收尾
# ============================================================================
step "10. 环境持久化"

cd "$PROJECT_DIR"

# 生成快捷激活脚本 env.sh
ENV_SH="$SCRIPT_DIR/env.sh"
cat > "$ENV_SH" << EOF
#!/bin/bash
# 视频超分环境快捷激活脚本
# 用法: source env.sh

export PATH="$PROJECT_DIR/.venv/bin:\$PATH"
export PATH="$TENSORRT_INSTALL_DIR/bin:\$PATH"
export PATH=/usr/local/cuda-12.2/bin:\$PATH
export LD_LIBRARY_PATH="$TENSORRT_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PROJECT_DIR/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH"

echo "✅ 视频超分环境已激活"
echo "  项目目录: $PROJECT_DIR"
echo "  插件目录: $PROJECT_DIR/lib"
echo "  模型目录: $PROJECT_DIR/models"
echo ""
echo "  用法示例:"
echo "    vspipe 脚本.vpy - | ffmpeg -i - output.mp4"
EOF
chmod +x "$ENV_SH"
info "快捷激活脚本已生成: $ENV_SH"

# 写入 ~/.bashrc
if ! grep -q "video-upscale environment" ~/.bashrc 2>/dev/null; then
    echo "source \"$ENV_SH\"  # video-upscale environment" >> ~/.bashrc
    info "环境配置已追加到 ~/.bashrc"
else
    warn "~/.bashrc 中已存在 video-upscale 环境配置，跳过"
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  🎉 部署完成！                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  项目目录:   $PROJECT_DIR"
echo "║  插件目录:   $PROJECT_DIR/lib"
echo "║  模型目录:   $PROJECT_DIR/models"
echo "║  日志文件:   $LOG_FILE"
echo "║  激活脚本:   $ENV_SH"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  快速开始:                                                  ║"
echo "║    source $ENV_SH"
echo "║    vspipe script.vpy - | ffmpeg -i - output.mp4             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

success "部署完成！详细日志: $LOG_FILE"
