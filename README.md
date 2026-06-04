# 🎬 视频超分环境一键部署

基于 **VapourSynth + TensorRT + RealESRGAN** 的 GPU 视频超分辨率处理环境，支持 WSL2 和原生 Ubuntu LTS。

## 特性

- ✅ **一键部署** — 运行 `setup_video_upscale.sh` 自动完成全部安装
- ✅ **可移植** — 整个项目目录可复制到任意机器，离线部署
- ✅ **GPU 加速** — 基于 TensorRT 8.6 + CUDA 12.2 的 FP16 推理引擎
- ✅ **丰富模型** — 预置 RealESRGAN、Waifu2x、CUGAN、RIFE、DPIR 等多款 ONNX 模型
- ✅ **模块化** — 支持 `--skip-*` 选项跳过任意步骤
- ✅ **Dry-run** — 支持 `--dry-run` 预览将执行的操作

## 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04 / 24.04 LTS 或 WSL2 |
| NVIDIA 驱动 | ≥ 525 (支持 CUDA 12.2) |
| GPU 架构 | ≥ Turing (SM 75)，推荐 Ada (SM 89) |
| 磁盘空间 | ~20 GB (含离线包和模型) |
| 内存 | ≥ 16 GB |

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/video-upscale.git
cd video-upscale

# 2. 将离线包放入 Downloads/ 目录（见下方离线部署说明）

# 3. 预览部署步骤
./setup_video_upscale.sh --dry-run

# 4. 执行完整部署
./setup_video_upscale.sh

# 5. 激活环境
source env.sh

# 6. 运行视频超分
vspipe your_script.vpy - | ffmpeg -i - output.mp4
```

## 目录结构

```
video_upscale/
├── setup_video_upscale.sh      # 一键部署脚本
├── env.sh                      # 环境激活脚本（部署后自动生成）
├── README.md                   # 本文件
├── Downloads/                  # 离线安装包（需自行准备）
│   ├── cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
│   ├── cuda-wsl-ubuntu.pin
│   ├── TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
│   ├── cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
│   └── libtinfo5_6.3-2ubuntu0.1_amd64.deb
├── video_upscale_project/      # 项目工作目录
│   ├── src/                    # Python 脚本
│   ├── models/                 # AI 模型文件
│   │   └── downloaded/         # 预下载的 ONNX 模型
│   ├── plugins/                # VapourSynth 插件源码
│   ├── lib/                    # 编译好的 .so 插件
│   └── vapoursynth_headers/    # VapourSynth 头文件
└── ai_libs/                    # TensorRT 运行库（部署后生成）
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
# 仅重新编译插件（其他已安装）
./setup_video_upscale.sh --skip-cuda --skip-tensorrt --skip-cudnn --skip-python --skip-deps --skip-models

# 仅安装 Python 环境和依赖
./setup_video_upscale.sh --skip-cuda --skip-tensorrt --skip-cudnn --skip-vapoursynth --skip-vsmlrt --skip-ffms2 --skip-models
```

## 离线部署

将以下离线包放入 `Downloads/` 目录即可离线安装：

| 文件 | 大小 | 下载地址 |
|------|------|----------|
| `cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb` | ~3.2 GB | [NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-12-2-0-download-archive) |
| `TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz` | ~1.3 GB | [NVIDIA TensorRT Archive](https://developer.nvidia.com/tensorrt) |
| `cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz` | ~700 MB | [NVIDIA cuDNN Archive](https://developer.nvidia.com/cudnn) |
| `cuda-wsl-ubuntu.pin` | <1 KB | CUDA WSL 包附带 |
| `libtinfo5_6.3-2ubuntu0.1_amd64.deb` | ~100 KB | `apt download libtinfo5` |

> **注意**: CUDA 和 cuDNN 下载需要 NVIDIA Developer 账号（免费注册）。

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
│               VapourSynth 脚本 (.vpy)          │
├──────────────────────────────────────────────┤
│  FFMS2 (视频源)   │   vs-mlrt (TensorRT 推理)  │
├──────────────────────────────────────────────┤
│          VapourSynth 核心框架 (R70+)           │
├──────────────────────────────────────────────┤
│  TensorRT 8.6    │  CUDA 12.2  │  cuDNN 8.9  │
├──────────────────────────────────────────────┤
│             NVIDIA GPU (SM 75+)               │
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
vspipe upscale_example.vpy -c y4m - | ffmpeg -i - -c:v libx264 -crf 18 output.mp4
```

## FAQ

### Q: 非 WSL 环境如何安装 CUDA？

非 WSL 环境建议使用 `--skip-cuda` 跳过，手动通过 NVIDIA 官方 runfile 安装 CUDA 12.2。

### Q: GPU 架构不匹配怎么办？

修改脚本中 `CMAKE_CUDA_ARCHITECTURES` 为你的 GPU 对应的 SM 版本：
- RTX 20 系列: 75
- RTX 30 系列: 86
- RTX 40 系列: 89
- RTX 50 系列: 100

### Q: 部署失败如何排查？

```bash
# 查看完整日志
cat setup.log

# 查看分步骤日志
cat video_upscale_project/logs/engine_build_real.log
```

## License

本项目中各组件遵循其各自的许可证：
- VapourSynth: LGPL 2.1+
- FFMS2: MIT
- vs-mlrt: MIT
- RealESRGAN: BSD 3-Clause
