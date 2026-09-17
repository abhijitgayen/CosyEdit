# CosyEdit / CosyVoice Pure Rust Runtime (Python-Free Native ONNX Inference)

This directory contains a complete, high-performance, cross-platform Rust runtime server implementation for **CosyEdit** and **CosyVoice** powered by **ONNX Runtime (`ort`) native model inference**.

---

## 🚀 Why Pure Rust & Native ONNX Runtime?

1. **Python-Free Execution**: Runs ONNX model graphs directly in Rust via `ort` ONNX Runtime bindings. Eliminates Python interpreter dependency, PyTorch runtime bloat, and Global Interpreter Lock (GIL) bottlenecks.
2. **High Concurrency & Parallel Request Processing**: Built on Rust's async runtime (`tokio`), REST server framework (`axum`), and gRPC framework (`tonic`). Multi-threaded worker pools handle parallel client requests efficiently.
3. **Cross-Platform & Embedded Device Portability**: Compiles into a single self-contained binary across Linux, macOS, Windows, x86_64, ARM64, and resource-constrained edge/embedded devices.
4. **Zero-Copy Streaming**: Synthesized audio PCM bytes stream back to callers via HTTP Chunked Transfer and gRPC Streams with minimal memory allocation.

---

## 🏗️ System Architecture

```
                       ┌────────────────────────┐
                       │   Client Application   │
                       └───────────┬────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │ HTTP REST (50000) / gRPC (50001) │
                     └─────────────┬─────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │    Rust Async Server    │
                      │ (Tokio / Axum / Tonic)  │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   Worker Pool Manager   │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │  Native Rust ONNX       │
                      │  ModelEngine (ort)      │
                      └────────────┬────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
│  campplus.onnx  │       │speech_tokenizer │       │  flow.decoder   │
│ (Spk Embedding) │       │   _v2.onnx      │       │ .estimator.onnx │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 📖 Step-by-Step Guide: Exporting & Loading ONNX Models

### Step 1: Export/Obtain ONNX Models

Place your exported ONNX model files inside a directory (e.g. `pretrained_models/CosyEdit/`):

- **`campplus.onnx`**: Speaker embedding extraction model.
  - *Input*: `speech` `[batch_size, num_samples]` (f32 waveform at 16kHz)
  - *Output*: `embedding` `[batch_size, 192]` (f32 speaker embedding vector)
- **`speech_tokenizer_v2.onnx`** or **`speech_tokenizer_v3.onnx`**: Acoustic speech tokenizer model.
  - *Input*: `speech` `[batch_size, num_samples]`
  - *Output*: `speech_token` `[batch_size, num_tokens]` (i64 token IDs)
- **`flow.decoder.estimator.fp32.onnx`**: Flow-matching decoder estimator model.
  - *Input*: `tokens` `[batch_size, num_tokens]`, `embedding` `[batch_size, 192]`
  - *Output*: `mel` `[batch_size, num_frames, num_mels]`

---

### Step 2: Build Rust Runtime Binary

```bash
cd runtime/rust
cargo build --release
```

---

### Step 3: Launch Native Rust ONNX Inference Server

To run the server in pure Rust ONNX execution mode loading ONNX models:

```bash
./target/release/cosyedit_runtime \
  --port 50000 \
  --grpc-port 50001 \
  --model-dir pretrained_models/CosyEdit
```

CLI options:
- `--port` (default: `50000`): HTTP REST API listening port.
- `--grpc-port` (default: `50001`): gRPC server listening port.
- `--host` (default: `0.0.0.0`): Network host interface.
- `--model-dir`: Directory path containing the `.onnx` model files.
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
