#!/usr/bin/env bash
set -e

MODEL_DIR="${1:-pretrained_models/CosyEdit}"
SOURCE="${2:-huggingface}"    # options: huggingface, modelscope
MODEL_FILE="${3:-all_onnx}"   # options: campplus.onnx, speech_tokenizer_v2.onnx, flow.decoder.estimator.fp32.onnx, all_onnx, or all

echo "========================================================="
echo "  CosyEdit Pretrained Model Downloader"
echo "  Target Directory: ${MODEL_DIR}"
echo "  Source:           ${SOURCE}"
echo "  Model File(s):    ${MODEL_FILE}"
echo "========================================================="

mkdir -p "${MODEL_DIR}"

# List of standard ONNX models required for Rust runtime
ONNX_FILES=("campplus.onnx" "speech_tokenizer_v2.onnx" "flow.decoder.estimator.fp32.onnx")

download_single_hf() {
    local file_name="$1"
    echo "Downloading '${file_name}' from HuggingFace (CJY/CosyEdit)..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download CJY/CosyEdit "${file_name}" --local-dir "${MODEL_DIR}"
    else
        local url="https://huggingface.co/CJY/CosyEdit/resolve/main/${file_name}"
        echo "huggingface-cli not found, falling back to curl from ${url}..."
        curl -L -o "${MODEL_DIR}/${file_name}" "${url}"
    fi
}

download_single_ms() {
    local file_name="$1"
    echo "Downloading '${file_name}' from ModelScope (CJY1018/CosyEdit)..."
    local url="https://www.modelscope.cn/models/CJY1018/CosyEdit/repo?Revision=master&FilePath=${file_name}"
    if command -v python3 &> /dev/null && python3 -c "import modelscope" &> /dev/null; then
        python3 -c "from modelscope.hub.file_download import modelscope_file_download; modelscope_file_download('CJY1018/CosyEdit', '${file_name}', local_dir='${MODEL_DIR}')"
    else
        curl -L -o "${MODEL_DIR}/${file_name}" "${url}"
    fi
}

if [ "${SOURCE}" = "huggingface" ]; then
    if [ "${MODEL_FILE}" = "all_onnx" ]; then
        echo "Downloading only ONNX models for Rust runtime..."
        for file in "${ONNX_FILES[@]}"; do
            download_single_hf "${file}"
        done
    elif [ "${MODEL_FILE}" = "all" ]; then
        echo "Downloading full repo from HuggingFace..."
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download CJY/CosyEdit --local-dir "${MODEL_DIR}"
        else
            git clone https://huggingface.co/CJY/CosyEdit "${MODEL_DIR}_tmp"
            cp -r "${MODEL_DIR}_tmp"/* "${MODEL_DIR}/"
            rm -rf "${MODEL_DIR}_tmp"
        fi
    else
        download_single_hf "${MODEL_FILE}"
    fi

elif [ "${SOURCE}" = "modelscope" ]; then
    if [ "${MODEL_FILE}" = "all_onnx" ]; then
        echo "Downloading only ONNX models for Rust runtime..."
        for file in "${ONNX_FILES[@]}"; do
            download_single_ms "${file}"
        done
    elif [ "${MODEL_FILE}" = "all" ]; then
        echo "Downloading full repository from ModelScope..."
        if command -v python3 &> /dev/null && python3 -c "import modelscope" &> /dev/null; then
            python3 -c "from modelscope import snapshot_download; snapshot_download('CJY1018/CosyEdit', local_dir='${MODEL_DIR}')"
        else
            git clone https://www.modelscope.cn/CJY1018/CosyEdit.git "${MODEL_DIR}_tmp"
            cp -r "${MODEL_DIR}_tmp"/* "${MODEL_DIR}/"
            rm -rf "${MODEL_DIR}_tmp"
        fi
    else
        download_single_ms "${MODEL_FILE}"
    fi
else
    echo "Invalid source specified. Use 'huggingface' or 'modelscope'."
    exit 1
fi

echo "========================================================="
echo "  Successfully downloaded model asset(s) to ${MODEL_DIR}"
echo "  Model directory contents:"
ls -lh "${MODEL_DIR}"
echo "========================================================="
