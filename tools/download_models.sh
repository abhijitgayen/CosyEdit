#!/usr/bin/env bash
set -e

MODEL_DIR="${1:-pretrained_models/CosyEdit}"
SOURCE="${2:-huggingface}" # options: huggingface, modelscope

echo "========================================================="
echo "  CosyEdit Pretrained Model Downloader"
echo "  Target Directory: ${MODEL_DIR}"
echo "  Source:           ${SOURCE}"
echo "========================================================="

mkdir -p "${MODEL_DIR}"

if [ "${SOURCE}" = "huggingface" ]; then
    REPO_URL="https://huggingface.co/CJY/CosyEdit"
    echo "Downloading CosyEdit models from HuggingFace (${REPO_URL})..."

    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download CJY/CosyEdit --local-dir "${MODEL_DIR}"
    elif command -v git-lfs &> /dev/null || command -v git &> /dev/null; then
        git clone https://huggingface.co/CJY/CosyEdit "${MODEL_DIR}_tmp"
        cp -r "${MODEL_DIR}_tmp"/* "${MODEL_DIR}/"
        rm -rf "${MODEL_DIR}_tmp"
    else
        echo "Error: Neither huggingface-cli nor git is installed."
        echo "Please install huggingface-cli via: pip install huggingface_hub"
        exit 1
    fi

elif [ "${SOURCE}" = "modelscope" ]; then
    echo "Downloading CosyEdit models from ModelScope (CJY1018/CosyEdit)..."

    if command -v modelscope &> /dev/null || python3 -c "import modelscope" &> /dev/null; then
        python3 -c "from modelscope import snapshot_download; snapshot_download('CJY1018/CosyEdit', local_dir='${MODEL_DIR}')"
    elif command -v git &> /dev/null; then
        git clone https://www.modelscope.cn/CJY1018/CosyEdit.git "${MODEL_DIR}_tmp"
        cp -r "${MODEL_DIR}_tmp"/* "${MODEL_DIR}/"
        rm -rf "${MODEL_DIR}_tmp"
    else
        echo "Error: Neither modelscope SDK nor git is installed."
        exit 1
    fi
else
    echo "Invalid source specified. Use 'huggingface' or 'modelscope'."
    exit 1
fi

echo "========================================================="
echo "  Successfully downloaded model assets to ${MODEL_DIR}"
echo "  Model directory contents:"
ls -lh "${MODEL_DIR}"
echo "========================================================="
