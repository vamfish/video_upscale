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
# 版本配置 — 如需更换版本，修改此处即可
# ============================================================================
CUDA_VERSION="12.2"
TENSORRT_VERSION="8.6.1.6"
TENSORRT_CUDA_COMPAT="12.0"
CUDNN_VERSION="8.9.7.29"
CUDNN_CUDA_COMPAT="12"
PYTHON_VERSION="3.12"
FFMS2_VERSION="5.0"

# 离线包文件名
CUDA_DEB="cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb"
CUDA_PIN="cuda-wsl-ubuntu.pin"
TENSORRT_TAR="TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${TENSORRT_CUDA_COMPAT}.tar.gz"
CUDNN_TAR="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${CUDNN_CUDA_COMPAT}-archive.tar.xz"
LIBTINFO5_DEB="libtinfo5_6.3-2ubuntu0.1_amd64.deb"

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
dry()     { if $DRY_RUN; then echo -e "${YELLOW}[DRY-RUN]${NC}  ${1:-}"; return 0; else return 1; fi; }

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
    info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '未知')"
else
    error "未找到 nvidia-smi，请确认 NVIDIA 驱动已安装"
fi

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
        if $DRY_RUN; then
            dry "curl -LsSf https://astral.sh/uv/install.sh | sh"
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
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
    run cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release

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

    # find -print -quit 替代 head -1，避免 SIGPIPE
    SO_FILE=$(find . -name "libffms2.so" -type f -print -quit)
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
        if has_cmd trtexec; then
            info "编译 TensorRT 引擎 (RealESRGAN_x4plus)，可能需要几分钟..."
            mkdir -p logs
            run trtexec --onnx="$ONNX_FILE" \
                --saveEngine="$ENGINE_FILE" \
                --minShapes=input:1x3x360x480 \
                --optShapes=input:1x3x540x960 \
                --maxShapes=input:1x3x1080x1920 \
                --fp16 --verbose > "logs/engine_build_real.log" 2>&1
            success "TensorRT 引擎编译成功 → $ENGINE_FILE"
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
        if has_cmd trtexec; then
            info "编译 TensorRT 引擎 (Waifu_cunet_x2n1)..."
            run trtexec --onnx="$WAIFU_ONNX" \
                --saveEngine="$WAIFU_ENGINE" \
                --minShapes=input:1x3x360x480 \
                --optShapes=input:1x3x540x960 \
                --maxShapes=input:1x3x1080x1920 \
                --fp16 --verbose > "logs/engine_build_Waifu_cunet_x2n1.log" 2>&1
            success "TensorRT 引擎编译成功 → $WAIFU_ENGINE"
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
