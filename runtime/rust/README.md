# CosyEdit / CosyVoice Pure Rust Runtime (Python-Free Native ONNX Inference)

This directory contains a complete, high-performance, cross-platform Rust runtime server implementation for **CosyEdit** and **CosyVoice** powered by **ONNX Runtime (`ort`) native model inference**.

---

## 🚀 Why Pure Rust & Native ONNX Runtime?

1. **Python-Free Execution**: Runs ONNX model graphs directly in Rust via `ort` ONNX Runtime bindings. Eliminates Python interpreter dependency, PyTorch runtime bloat, and Global Interpreter Lock (GIL) bottlenecks.
2. **High Concurrency & Parallel Request Processing**: Built on Rust's async runtime (`tokio`), REST server framework (`axum`), and gRPC framework (`tonic`). Multi-threaded worker pools handle parallel client requests efficiently.
3. **Cross-Platform & Embedded Device Portability**: Compiles into a single self-contained binary across Linux, macOS, Windows, x86_64, ARM64, and resource-constrained edge/embedded devices.
4. **Zero-Copy Streaming**: Synthesized audio PCM bytes stream back to callers via HTTP Chunked Transfer and gRPC Streams with minimal memory allocation.

---

## 📥 Step-by-Step Guide: Downloading Pretrained Model Assets

### Option A: Automated Download Script (Recommended)

Use the provided shell script to download `pretrained_models/CosyEdit` from HuggingFace or ModelScope:

```bash
# Download from HuggingFace (Default)
./tools/download_models.sh pretrained_models/CosyEdit huggingface

# Or download from ModelScope
./tools/download_models.sh pretrained_models/CosyEdit modelscope
```

---

### Option B: Download via HuggingFace Hub CLI

```bash
pip install huggingface_hub
huggingface-cli download CJY/CosyEdit --local-dir pretrained_models/CosyEdit
```

---

### Option C: Download via ModelScope SDK

```bash
pip install modelscope
python3 -c "from modelscope import snapshot_download; snapshot_download('CJY1018/CosyEdit', local_dir='pretrained_models/CosyEdit')"
```

---

### 📂 Expected Pretrained Model Assets Structure

Once downloaded, `pretrained_models/CosyEdit/` will contain:

```
pretrained_models/CosyEdit/
├── campplus.onnx                    # Speaker embedding extraction model
├── speech_tokenizer_v2.onnx         # Acoustic speech tokenizer model
├── flow.decoder.estimator.fp32.onnx  # Flow-matching decoder estimator model
└── ...
```

---

## 📦 Building & Running the Rust Runtime

### Step 1: Build Rust Binary

```bash
cd runtime/rust
cargo build --release
```

---

### Step 2: Launch Native Rust ONNX Server

To run the server and automatically download model assets if missing:

```bash
./target/release/cosyedit_runtime \
  --port 50000 \
  --grpc-port 50001 \
  --model-dir pretrained_models/CosyEdit \
  --download-models
```

CLI Options:
- `--port` (default: `50000`): HTTP REST API listening port.
- `--grpc-port` (default: `50001`): gRPC server listening port.
- `--host` (default: `0.0.0.0`): Network host interface.
- `--model-dir` (default: `pretrained_models/CosyEdit`): Path to directory containing ONNX model files.
- `--download-models`: Automatically fetch pretrained models if directory does not exist.
- `--backend-url` *(optional)*: Remote fallback proxy endpoint URL.

---

## 🛠️ Supported API Endpoints

| Endpoint | Method | Payload / Description |
|---|---|---|
| `/inference_sft` | `POST / GET` | Pre-trained SFT speaker synthesis (`tts_text`, `spk_id`). |
| `/inference_zero_shot` | `POST (multipart)` | Zero-shot voice cloning (`tts_text`, `prompt_text`, `prompt_wav`). |
| `/inference_cross_lingual` | `POST (multipart)` | Cross-lingual speech synthesis (`tts_text`, `prompt_wav`). |
| `/inference_instruct` | `POST` | Instruct-guided speech synthesis (`tts_text`, `spk_id`, `instruct_text`). |
| `/inference_instruct2` | `POST (multipart)` | Instruct-guided speech synthesis (`tts_text`, `instruct_text`, `prompt_wav`). |
| `/inference_edit` | `POST (multipart)` | End-to-end multi-span speech editing (`target_text`, `original_text`, `original_speech`). |
| `CosyVoice/Inference` | `gRPC Stream` | High-performance gRPC streaming endpoint (`proto/cosyvoice.proto`). |

---

## 🧪 Testing

Run integration tests:

```bash
cd runtime/rust
cargo test
```

---

## 📝 Usage Example

### Speech Editing HTTP POST Request

```bash
curl -X POST http://127.0.0.1:50000/inference_edit \
  -F "target_text=Hello world" \
  -F "original_text=Hello friend" \
  -F "original_speech=@/path/to/original.wav" \
  --output edited.wav
```
