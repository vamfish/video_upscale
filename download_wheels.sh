#!/bin/bash
# =============================================================================
# 从 uv.lock 提取 wheel 下载 URL 并下载到指定目录
# 不依赖 pip, 纯 shell + curl/wget
# 用法: ./download_wheels.sh [--dest ./offline_pypi/]
# =============================================================================
set -euo pipefail

DEST="${1:-Downloads/offline_pypi}"
LOCK_FILE="video_upscale_project/uv.lock"

mkdir -p "$DEST"

echo "📦 从 uv.lock 提取下载 URL..."
echo "   目标目录: $DEST"
echo ""

# 提取当前平台 (linux x86_64, Python 3.12) 的 wheel URL
# cp312 = Python 3.12 / py3-none = 通用 Python 3 / manylinux/linux = Linux x86_64
grep -oP 'url\s*=\s*"\K[^"]+' "$LOCK_FILE" \
    | grep -iE '(cp312|cp312m|py3-none|py2\.py3-none).*(manylinux|linux).*x86_64.*\.whl$' \
    | grep -vi 'cp31[3-9]\|cp3[2-9][0-9]\|aarch64\|win_\|macosx' \
    | sort -u > /tmp/uv_wheel_urls.txt

# 额外: 纯 Python 通用 wheel (py3-none-any.whl)
grep -oP 'url\s*=\s*"\K[^"]+' "$LOCK_FILE" \
    | grep -E 'py3-none-any\.whl$|py2\.py3-none-any\.whl$' \
    | sort -u >> /tmp/uv_wheel_urls.txt

# 去重
sort -u -o /tmp/uv_wheel_urls.txt /tmp/uv_wheel_urls.txt

TOTAL=$(wc -l < /tmp/uv_wheel_urls.txt)
echo "   找到 $TOTAL 个 wheel 文件"
echo ""

COUNT=0
SKIPPED=0
DOWNLOADED=0

while IFS= read -r url; do
    COUNT=$((COUNT + 1))
    filename=$(basename "$url" | sed 's/[?#].*//')
    dest_path="$DEST/$filename"

    if [ -f "$dest_path" ]; then
        SKIPPED=$((SKIPPED + 1))
        printf "\r[%3d/%3d] ⏭  (已存在) %s" "$COUNT" "$TOTAL" "$filename"
        continue
    fi

    printf "\r[%3d/%3d] ⬇  %s" "$COUNT" "$TOTAL" "$filename"
    if curl -sSL --connect-timeout 30 --max-time 300 -o "$dest_path" "$url" 2>/dev/null; then
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        # curl 失败时尝试 wget
        if wget -q --timeout=30 --tries=3 -O "$dest_path" "$url" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo ""
            echo "   ⚠ 下载失败: $filename (跳过)"
            rm -f "$dest_path"
        fi
    fi
done < /tmp/uv_wheel_urls.txt
rm -f /tmp/uv_wheel_urls.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  完成: $DOWNLOADED 下载, $SKIPPED 跳过 (共 $TOTAL)"
ls -lh "$DEST" | tail -5
echo "  总计: $(du -sh "$DEST" | cut -f1)"
