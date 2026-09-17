# CosyEdit / CosyVoice Lightweight Rust Runtime (Python-Free Native ONNX Execution)

This directory contains an optimized, lightweight, cross-platform Rust runtime server implementation for **CosyEdit** and **CosyVoice** with native **ONNX Runtime (`ort`) model inference**.

---

## 🚀 Why Rust Runtime & Native ONNX?

1. **Python-Free Execution**: Runs ONNX model graphs directly using Rust bindings to ONNX Runtime (`ort`), eliminating Python dependency, interpreter overhead, and Python GIL bottlenecks during inference.
2. **High Concurrency & Parallel Request Processing**: Built on Rust's async runtime (`tokio`), REST server framework (`axum`), and gRPC framework (`tonic`). Efficiently handles simultaneous client requests with parallel worker pools.
3. **Cross-Platform & Cross-Device Portability**: Easily compiles into a single, standalone binary for Linux, macOS, Windows, x86_64, ARM64, and resource-constrained edge/embedded devices.
4. **Zero-Copy Streaming**: Synthesized audio PCM bytes stream back to callers via HTTP Chunked Transfer and gRPC Streams with minimal memory allocation and lower latency.
5. **Flexible Execution Modes**:
   - **Native ONNX Execution Mode**: Executes local ONNX model graphs directly in Rust.
   - **High-Throughput Proxy Mode**: Option to proxy/load-balance requests to upstream Triton or Python worker backends.

---

## 🏗️ Architecture

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
               ┌───────────────────┴───────────────────┐
               │                                       │
     ┌─────────▼─────────┐                   ┌─────────▼─────────┐
     │  Native Rust ONNX │                   │ Forwarding Proxy  │
     │  ModelEngine (ort)│                   │ (Triton / Python) │
     └───────────────────┘                   └───────────────────┘
```

---

## 🛠️ Supported Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/inference_sft` | `POST / GET` | Pre-trained SFT speaker synthesis. |
| `/inference_zero_shot` | `POST (multipart)` | Zero-shot voice cloning with prompt text & audio. |
| `/inference_cross_lingual` | `POST (multipart)` | Cross-lingual speech synthesis with prompt audio. |
| `/inference_instruct` | `POST` | Instruct-guided speech synthesis with speaker ID. |
| `/inference_instruct2` | `POST (multipart)` | Instruct-guided speech synthesis with prompt audio. |
| `/inference_edit` | `POST (multipart)` | End-to-end multi-span speech editing (target text, original text, original speech). |
| `CosyVoice/Inference` | `gRPC Stream` | High-performance gRPC streaming endpoint (`cosyvoice.proto`). |

---

## 📦 Building & Running

### Prerequisites

- **Rust toolchain** (`cargo` / `rustc` 1.70+)
- **Protobuf compiler** (`protoc`)

### Build

```bash
cd runtime/rust
cargo build --release
```

### Run Native Rust ONNX Inference Mode

```bash
./target/release/cosyedit-runtime \
  --port 50000 \
  --grpc-port 50001 \
  --model-dir pretrained_models/CosyEdit
```

### Run in Backend Proxy Mode

```bash
./target/release/cosyedit-runtime \
  --port 50000 \
  --grpc-port 50001 \
  --backend-url http://127.0.0.1:8000
```

---

## 🧪 Testing

Run integration tests:

```bash
cd runtime/rust
cargo test
```

---

## 📝 Examples

### Speech Editing HTTP Request

```bash
curl -X POST http://127.0.0.1:50000/inference_edit \
  -F "target_text=Hello world" \
  -F "original_text=Hello friend" \
  -F "original_speech=@/path/to/original.wav" \
  --output edited.wav
```
