# 🎬 视频超分环境一键部署

基于 **VapourSynth + TensorRT + RealESRGAN** 的 GPU 视频超分辨率处理环境，支持 WSL2 和原生 Ubuntu LTS，已适配 RTX 50 系列 (Blackwell)。

## 特性

- ✅ **一键部署** — `./setup_video_upscale.sh` 自动完成从 CUDA 到模型引擎的全流程安装
- ✅ **多 GPU 适配** — 自动检测 GPU 架构 (SM)，匹配最佳 TensorRT/CUDA 版本
- ✅ **可移植** — 整个目录可复制到任意机器离线部署，无需联网
- ✅ **自动兼容性检测** — 启动即检查 Downloads/ 中的离线包是否兼容当前 GPU
- ✅ **环境验证** — `./verify.sh` 一键检查 8 大类 23 项环境指标
- ✅ **模块化** — 支持 `--skip-*` 和 `--dry-run`，灵活控制部署流程

## 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04 / 24.04 LTS 或 WSL2 |
| GPU 架构 | ≥ Turing (SM 75) |
| NVIDIA 驱动 | ≥ 580 (CUDA 13.x)，或 ≥ 525 (CUDA 12.x) |
| 磁盘空间 | ~25 GB (含离线包和模型) |
| 内存 | ≥ 16 GB |

### 已验证 GPU

| GPU | 架构 | SM | 状态 |
|-----|------|-----|------|
| RTX 5080 | Blackwell | 120 | ✅ 已验证 |
| RTX 4090 | Ada Lovelace | 89 | ✅ 兼容 |
| RTX A5000 | Ampere | 86 | ✅ 兼容 |

## 目录结构

```
video_upscale/                       # 项目根目录 (git clone 得到)
│
├── setup_video_upscale.sh           # 一键部署脚本
├── verify.sh                        # 环境验证脚本 (8 类 23 项)
├── download_wheels.sh               # Python wheel 离线下载 (从 uv.lock 提取)
├── env.sh                           # 环境激活脚本 (部署后自动生成)
├── README.md                        # 本文件
├── .gitignore                       # Git 忽略规则
│
├── Downloads/                       # 离线包 (需手动放入，共 ~12 GB)
│   ├── cuda-repo-wsl-ubuntu-13-3-local_13.3.0-1_amd64.deb   (3.5 GB)
│   ├── cuda-wsl-ubuntu.pin                                  (<1 KB)
│   ├── nv-tensorrt-local-repo-ubuntu2404-11.0.0-cuda-13.2_1.0-1_amd64.deb  (1.5 GB)
│   ├── cudnn-local-repo-ubuntu2404-9.23.0_1.0-1_amd64.deb  (800 MB)
│   ├── libtinfo5_6.3-2ubuntu0.1_amd64.deb                  (100 KB)
│   └── offline_pypi/                # Python wheel 离线源 (~5 GB)
│       ├── torch-2.12.0-cp312-*.whl
│       ├── nvidia_cublas-13.1.*.whl
│       └── ... (约 120 个 .whl 文件)
│
├── video_upscale_project/           # 项目工作目录
│   ├── pyproject.toml               # Python 项目配置 (uv init 生成)
│   ├── uv.lock                      # uv 锁定文件 (精确版本，机器间一致)
│   ├── .venv/                       # Python 虚拟环境 (9.5 GB，部署后生成)
│   ├── src/
│   │   └── convert_pth_to_onnx.py   # .pth → .onnx 转换脚本
│   ├── models/
│   │   ├── RealESRGAN_x4plus.pth    # 原始 PyTorch 模型 (64 MB)
│   │   ├── RealESRGAN_x4plus.onnx   # ONNX 模型 (3 MB)
│   │   ├── RealESRGAN_x4plus.engine # TensorRT 引擎 (70 MB，部署后生成)
│   │   └── downloaded/              # 预下载 ONNX 模型 (waifu2x/cugan/rife/dpir)
│   ├── plugins/                     # VapourSynth 插件源码 (部署时自动 git clone)
│   ├── lib/                         # 编译好的 .so 插件
│   └── vapoursynth_headers/         # VapourSynth 头文件 (部署时自动 git clone)
│
└── ai_libs/                         # TensorRT 运行库 (仅旧版 tar.gz 格式)
```

## 新机器从头部署

### 准备工作（在一台已联网的机器上完成）

```bash
# 1. 下载系统离线包 (CUDA / TensorRT / cuDNN)
#    从 NVIDIA Developer 网站下载，放入 Downloads/

# 2. 下载 Python wheel 离线包
./download_wheels.sh
# → Downloads/offline_pypi/ 中生成约 120 个 .whl 文件 (~5 GB)
```

### 部署步骤（目标机器，可离线）

```bash
# 1. 克隆仓库
git clone https://github.com/vamfish/video_upscale.git
cd video_upscale

# 2. 将准备好的离线包放入 Downloads/
#    - 5 个系统 deb 文件
#    - 整个 offline_pypi/ 目录

# 3. 预览
./setup_video_upscale.sh --dry-run

# 4. 部署
./setup_video_upscale.sh

# 5. 验证
source env.sh && ./verify.sh
# 预期: 23/23 全部 PASS
```

### 多台机器部署流程

```
                  ┌─ 下载离线包 (一次性) ─┐
                  │                        │
    源机器 (有网)  ──→  Downloads/          │
    运行:              ├── *.deb (系统)     │
    download_wheels.sh └── offline_pypi/    │
                           (~120 .whl)      │
                  │                        │
                  └────────────────────────┘
                            │ 复制到每台机器
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
     RTX A5000          RTX 4090           RTX 5080
     (Ampere)           (Ada)              (Blackwell)
     SM 86              SM 89              SM 120
          │                 │                  │
          └─────────────────┼─────────────────┘
                            │
                  每台执行:
                  ./setup_video_upscale.sh
                  source env.sh && ./verify.sh
```

## 命令行选项

```
用法: ./setup_video_upscale.sh [选项]

选项:
  --help                 显示帮助信息
  --dry-run              仅显示将执行的步骤，不实际执行
  --skip-cuda            跳过 CUDA 安装
  --skip-tensorrt        跳过 TensorRT 安装
  --skip-cudnn           跳过 cuDNN 安装
  --skip-vapoursynth     跳过 VapourSynth 编译安装
  --skip-python          跳过 Python 环境初始化
  --skip-deps            跳过 Python 依赖安装
  --skip-vsmlrt          跳过 vs-mlrt 插件编译
  --skip-ffms2           跳过 FFMS2 插件编译
  --skip-models          跳过模型下载与转换
```

### 常用组合

```bash
# 仅重新编译 vs-mlrt + FFMS2（其他已安装）
./setup_video_upscale.sh --skip-cuda --skip-tensorrt --skip-cudnn --skip-python --skip-deps --skip-vapoursynth --skip-models

# 仅安装 Python 环境和依赖
./setup_video_upscale.sh --skip-cuda --skip-tensorrt --skip-cudnn --skip-vapoursynth --skip-vsmlrt --skip-ffms2 --skip-models
```

## 离线部署

部署需要两类离线包：**系统组件**（CUDA/TensorRT/cuDNN deb）和 **Python 依赖**（wheel）。

### 1. 系统组件离线包

放入 `Downloads/` 目录，**三张 GPU (A5000/4090/5080) 共用同一套**：

| 文件 | 大小 | 下载地址 |
|------|------|----------|
| `cuda-repo-wsl-ubuntu-13-3-local_13.3.0-1_amd64.deb` | ~3.5 GB | [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) → Linux → x86_64 → WSL-Ubuntu → 13.3 → deb (local) |
| `nv-tensorrt-local-repo-ubuntu2404-11.0.0-cuda-13.2_1.0-1_amd64.deb` | ~1.5 GB | [NVIDIA TensorRT Downloads](https://developer.nvidia.com/tensorrt/download) |
| `cudnn-local-repo-ubuntu2404-9.23.0_1.0-1_amd64.deb` | ~800 MB | [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn) |
| `cuda-wsl-ubuntu.pin` | <1 KB | CUDA WSL 包附带 |
| `libtinfo5_6.3-2ubuntu0.1_amd64.deb` | ~100 KB | `apt download libtinfo5` |

> **注意**: 全部需要 **NVIDIA Developer 免费账号**（邮箱注册，5 分钟）。

### 2. Python 依赖离线包

从已有环境的 `uv.lock` 提取并下载所有 wheel：

```bash
# 在一台已部署的机器上执行
./download_wheels.sh
```

脚本从 `video_upscale_project/uv.lock` 提取约 120 个 wheel URL，下载到 `Downloads/offline_pypi/`（约 5 GB）。

在其他机器上，部署脚本会自动检测 `Downloads/offline_pypi/` 并离线安装：

```bash
# 脚本内部逻辑（无需手动执行）
uv sync --find-links Downloads/offline_pypi/ --no-index
```

> `uv sync` 会严格按 `uv.lock` 锁定版本，确保所有机器上 Python 包版本完全一致。

## 环境验证

部署完成后运行 `./verify.sh`，检查 8 大类 23 项：

```
╔══════════════════════════════════════════════════════════════╗
║       视频超分环境验证 v2.1                                  ║
╠══════════════════════════════════════════════════════════════╣
║  GPU: NVIDIA GeForce RTX 5080 (SM 120)                     ║
║  CUDA: 13.3         TensorRT: v11.0.0                      ║
╚══════════════════════════════════════════════════════════════╝

━━━ 1. NVIDIA GPU & CUDA ━━━     nvidia-smi / nvcc / cuDNN
━━━ 2. TensorRT ━━━              libnvinfer / trtexec
━━━ 3. VapourSynth ━━━           libvapoursynth / vspipe
━━━ 4. Python 环境 ━━━           uv / torch+CUDA / onnx / spandrel / vapoursynth
━━━ 5. VapourSynth 插件 ━━━      libvstrt.so / libffms2.so
━━━ 6. AI 模型文件 ━━━           .pth / .onnx / .engine / waifu2x
━━━ 7. 环境变量 ━━━              TensorRT 库 / CUDA PATH
━━━ 8. 功能自检 ━━━              uv run vspipe 执行测试
```

## 预置模型

`models/downloaded/` 目录中包含以下预下载的 ONNX 模型：

### 超分辨率模型

| 模型系列 | 架构 | 版本 |
|----------|------|------|
| **Waifu2x** | cunet, anime_style_art, photo, upresnet10, upconv_7 | 2x/3x, 多降噪级别 |
| **CUGAN** | pro, up2x-latest, up3x-latest | conservative, denoise, no-denoise |
| **RealESRGAN** | v2-animevideo-xsx2/x4, animevideo-v3 | 2x/4x |

### 其他模型

| 用途 | 模型 | 版本 |
|------|------|------|
| 插帧 | **RIFE** | v4.0 - v4.10 |
| 降噪/去块 | **DPIR** | color, grayscale, deblocking |

## 技术栈

```
┌──────────────────────────────────────────────┐
│           VapourSynth 脚本 (.vpy)              │
├──────────────────────────────────────────────┤
│  FFMS2 (视频源)   │   vs-mlrt (TensorRT 推理)  │
├──────────────────────────────────────────────┤
│        VapourSynth 核心框架 (R76+)             │
├──────────────────────────────────────────────┤
│  TensorRT 11.0   │  CUDA 13.3  │  cuDNN 9.23 │
├──────────────────────────────────────────────┤
│          NVIDIA GPU (SM 75-120)               │
└──────────────────────────────────────────────┘
```

## VapourSynth 脚本示例

```python
# upscale_example.vpy
import vapoursynth as vs
core = vs.core

# 加载视频
clip = core.ffms2.Source("input.mp4")

# RealESRGAN 4x 超分
clip = core.trt.Model(
    clip,
    engine_path="models/RealESRGAN_x4plus.engine",
    num_streams=3,
)

# 输出
clip.set_output()
```

```bash
# 运行
uv run vspipe upscale_example.vpy -c y4m - | ffmpeg -i - -c:v libx264 -crf 18 output.mp4
```

## GPU 版本对应关系

脚本自动检测 GPU 并匹配 Downloads/ 中的离线包：

| GPU SM | GPU 系列 | 最低 CUDA | 最低 TensorRT |
|--------|----------|-----------|---------------|
| 75-79 | RTX 20 (Turing) | 12.2 | 7.0 |
| 80-86 | RTX 30/A5000 (Ampere) | 12.2 | 8.0 |
| 89 | RTX 40 (Ada) | 12.2 | 8.6 |
| **100-120** | **RTX 50 (Blackwell)** | **12.4** | **10.0** |

> **统一推荐**: CUDA 13.3 + TensorRT 11.0 + cuDNN 9.23 兼容全部 GPU (SM 75-120)

## FAQ

### Q: 非 WSL 环境如何安装 CUDA？

非 WSL 环境建议使用 `--skip-cuda` 跳过，手动通过 NVIDIA 官方 runfile 安装。

### Q: 需要换版本怎么办？

直接下载新版离线包放入 `Downloads/`，脚本自动检测并匹配，无需手动修改变量。

### Q: 三台不同 GPU 的机器能共用同一套包吗？

可以。CUDA 13.3 + TensorRT 11.0 兼容 SM 75-120 全部架构。TensorRT engine 文件会在每台机器上自动根据其 GPU 编译。

### Q: 部署失败如何排查？

```bash
# 查看完整日志
cat setup.log

# 查看引擎编译日志
cat video_upscale_project/logs/engine_build_real.log

# 运行环境验证
source env.sh && ./verify.sh
```

### Q: 如何测试性能？

在三台机器上用相同的 `.vpy` 脚本，记录 vspipe 输出的 fps：
```bash
uv run vspipe upscale_example.vpy -c y4m --progress .
```

## License

本项目中各组件遵循其各自的许可证：
- VapourSynth: LGPL 2.1+
- FFMS2: MIT
- vs-mlrt: MIT
- RealESRGAN: BSD 3-Clause
