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

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/vamfish/video_upscale.git
cd video_upscale

# 2. 将离线包放入 Downloads/ 目录（见下方离线部署说明）

# 3. 预览部署步骤
./setup_video_upscale.sh --dry-run

# 4. 执行完整部署
./setup_video_upscale.sh

# 5. 激活环境
source env.sh

# 6. 验证环境
./verify.sh

# 7. 运行视频超分
uv run vspipe your_script.vpy -c y4m - | ffmpeg -i - output.mp4
```

## 目录结构

```
video_upscale/
├── setup_video_upscale.sh      # 一键部署脚本
├── verify.sh                   # 环境验证脚本 (23 项检查)
├── env.sh                      # 环境激活脚本（部署后自动生成）
├── README.md                   # 本文件
├── Downloads/                  # 离线安装包（需自行准备）
│   ├── cuda-repo-wsl-ubuntu-13-3-local_13.3.0-1_amd64.deb
│   ├── cuda-wsl-ubuntu.pin
│   ├── nv-tensorrt-local-repo-ubuntu2404-11.0.0-cuda-13.2_1.0-1_amd64.deb
│   ├── cudnn-local-repo-ubuntu2404-9.23.0_1.0-1_amd64.deb
│   └── libtinfo5_6.3-2ubuntu0.1_amd64.deb
├── video_upscale_project/      # 项目工作目录
│   ├── src/                    # Python 脚本
│   ├── models/                 # AI 模型文件
│   │   └── downloaded/         # 预下载的 ONNX 模型
│   ├── plugins/                # VapourSynth 插件源码（部署时自动克隆）
│   ├── lib/                    # 编译好的 .so 插件
│   └── vapoursynth_headers/    # VapourSynth 头文件（部署时自动克隆）
└── ai_libs/                    # TensorRT 运行库（仅旧版 tar 格式）
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

将以下离线包放入 `Downloads/` 目录即可离线安装。**三张 GPU (A5000/4090/5080) 共用同一套包**：

| 文件 | 大小 | 下载地址 |
|------|------|----------|
| `cuda-repo-wsl-ubuntu-13-3-local_13.3.0-1_amd64.deb` | ~3.5 GB | [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) → Linux → x86_64 → WSL-Ubuntu → 13.3 → deb (local) |
| `nv-tensorrt-local-repo-ubuntu2404-11.0.0-cuda-13.2_1.0-1_amd64.deb` | ~1.5 GB | [NVIDIA TensorRT Downloads](https://developer.nvidia.com/tensorrt/download) |
| `cudnn-local-repo-ubuntu2404-9.23.0_1.0-1_amd64.deb` | ~800 MB | [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn) |
| `cuda-wsl-ubuntu.pin` | <1 KB | CUDA WSL 包附带 |
| `libtinfo5_6.3-2ubuntu0.1_amd64.deb` | ~100 KB | `apt download libtinfo5` |

> **注意**: 全部需要 **NVIDIA Developer 免费账号**（邮箱注册，5 分钟搞定）。脚本启动时自动检测 GPU 架构并验证下载包兼容性。

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
