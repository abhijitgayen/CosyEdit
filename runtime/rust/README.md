# CosyEdit / CosyVoice Pure Rust Runtime (Python-Free Native ONNX Inference)

This directory contains a complete, high-performance, cross-platform Rust runtime server implementation for **CosyEdit** and **CosyVoice** powered by **ONNX Runtime (`ort`) native model inference**.

---

## 🚀 Why Pure Rust & Native ONNX Runtime?

1. **Python-Free Execution**: Runs ONNX model graphs directly in Rust via `ort` ONNX Runtime bindings. Eliminates Python interpreter dependency, PyTorch runtime bloat, and Global Interpreter Lock (GIL) bottlenecks.
2. **High Concurrency & Parallel Request Processing**: Built on Rust's async runtime (`tokio`), REST server framework (`axum`), and gRPC framework (`tonic`). Multi-threaded worker pools handle parallel client requests efficiently.
3. **Cross-Platform & Embedded Device Portability**: Compiles into a single self-contained binary across Linux, macOS, Windows, x86_64, ARM64, and resource-constrained edge/embedded devices.
4. **Zero-Copy Streaming**: Synthesized audio PCM bytes stream back to callers via HTTP Chunked Transfer and gRPC Streams with minimal memory allocation.

---

## 📥 Step-by-Step Guide: Selective ONNX Model Downloads

### Selective Download Script (`tools/download_models.sh`)

Instead of downloading heavy PyTorch model checkpoints, you can download **only the specific ONNX model(s) requested**:

```bash
# 1. Download ONLY ONNX model files for Rust runtime (Default)
./tools/download_models.sh pretrained_models/CosyEdit huggingface all_onnx

# 2. Download a single requested ONNX model (e.g. campplus.onnx)
./tools/download_models.sh pretrained_models/CosyEdit huggingface campplus.onnx

# 3. Download a single requested ONNX model (e.g. speech_tokenizer_v2.onnx)
./tools/download_models.sh pretrained_models/CosyEdit huggingface speech_tokenizer_v2.onnx

# 4. Download a single requested ONNX model (e.g. flow.decoder.estimator.fp32.onnx)
./tools/download_models.sh pretrained_models/CosyEdit huggingface flow.decoder.estimator.fp32.onnx

# 5. Download from ModelScope instead of HuggingFace
./tools/download_models.sh pretrained_models/CosyEdit modelscope campplus.onnx
```

---

### 📂 Pretrained Model Assets Structure

```
pretrained_models/CosyEdit/
├── campplus.onnx                    # Speaker embedding extraction model (~20MB)
├── speech_tokenizer_v2.onnx         # Acoustic speech tokenizer model (~100MB)
└── flow.decoder.estimator.fp32.onnx  # Flow-matching decoder estimator model (~300MB)
```

---

## 📦 Building & Running the Rust Runtime

### Step 1: Build Rust Binary

```bash
cd runtime/rust
cargo build --release
```

---

### Step 2: Launch Native Rust ONNX Server with Selective Auto-Download

To run the server and automatically download only requested ONNX model files if missing:

```bash
# Auto-download only the requested ONNX model files if missing
./target/release/cosyedit_runtime \
  --port 50000 \
  --grpc-port 50001 \
  --model-dir pretrained_models/CosyEdit \
  --download-models \
  --model-file all_onnx
```

CLI Options:
- `--port` (default: `50000`): HTTP REST API listening port.
- `--grpc-port` (default: `50001`): gRPC server listening port.
- `--host` (default: `0.0.0.0`): Network host interface.
- `--model-dir` (default: `pretrained_models/CosyEdit`): Path to directory containing ONNX model files.
- `--download-models`: Automatically fetch requested model files if directory/files do not exist.
- `--model-file` (default: `all_onnx`): Specific requested ONNX model file to download (e.g. `campplus.onnx`, `speech_tokenizer_v2.onnx`, `all_onnx`).
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
