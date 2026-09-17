use axum::extract::DefaultBodyLimit;
use clap::Parser;
use std::net::SocketAddr;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use tonic::transport::Server as GrpcServer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use cosyedit_runtime::engine::WorkerPool;
use cosyedit_runtime::grpc::{cosyvoice_proto::cosy_voice_server::CosyVoiceServer, CosyVoiceGrpcService};
use cosyedit_runtime::http::create_router;

#[derive(Parser, Debug)]
#[command(author, version, about = "CosyEdit & CosyVoice High-Performance Rust Runtime Server")]
struct Args {
    /// Port to listen on for HTTP REST API
    #[arg(short, long, default_value_t = 50000)]
    port: u16,

    /// Port to listen on for gRPC
    #[arg(long, default_value_t = 50001)]
    grpc_port: u16,

    /// Host IP address to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Local directory containing ONNX models (e.g., pretrained_models/CosyEdit)
    #[arg(long, default_value = "pretrained_models/CosyEdit")]
    model_dir: String,

    /// Automatically download requested ONNX model files if not found locally
    #[arg(long, default_value_t = false)]
    download_models: bool,

    /// Specific ONNX model file to download (e.g., campplus.onnx, all_onnx)
    #[arg(long, default_value = "all_onnx")]
    model_file: String,

    /// Optional upstream Python/Triton inference backend URL for forwarding requests
    #[arg(long)]
    backend_url: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let http_addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let grpc_addr: SocketAddr = format!("{}:{}", args.host, args.grpc_port).parse()?;

    tracing::info!("Starting CosyEdit Rust Runtime Server");
    tracing::info!("  HTTP REST server: http://{}", http_addr);
    tracing::info!("  gRPC server:      grpc://{}", grpc_addr);

    let model_path = Path::new(&args.model_dir);
    if (!model_path.exists() || args.download_models) && args.download_models {
        tracing::info!("Triggering selective ONNX model download for '{}'...", args.model_file);
        let status = Command::new("bash")
            .arg("../../tools/download_models.sh")
            .arg(&args.model_dir)
            .arg("huggingface")
            .arg(&args.model_file)
            .status();
        match status {
            Ok(s) if s.success() => tracing::info!("Selective ONNX model download finished successfully."),
            _ => tracing::warn!("Selective model download failed. You can run tools/download_models.sh manually."),
        }
    } else if !model_path.exists() {
        tracing::info!("Model path '{}' not found. Run 'tools/download_models.sh' or pass '--download-models --model-file campplus.onnx' to fetch selectively.", args.model_dir);
    }

    if let Some(ref backend) = args.backend_url {
        tracing::info!("Configured backend proxy endpoint: {}", backend);
    } else {
        tracing::info!("Running in native Rust ONNX inference mode");
    }

    let worker_pool = Arc::new(WorkerPool::new(args.backend_url, Some(&args.model_dir)));

    // HTTP Server task
    let http_pool = Arc::clone(&worker_pool);
    let http_app = create_router(http_pool).layer(DefaultBodyLimit::max(50 * 1024 * 1024));
    let http_listener = tokio::net::TcpListener::bind(http_addr).await?;

    let http_handle = tokio::spawn(async move {
        axum::serve(http_listener, http_app).await.unwrap();
    });

    // gRPC Server task
    let grpc_service = CosyVoiceGrpcService::new(Arc::clone(&worker_pool));
    let grpc_handle = tokio::spawn(async move {
        GrpcServer::builder()
            .add_service(CosyVoiceServer::new(grpc_service))
            .serve(grpc_addr)
            .await
            .unwrap();
    });

    let _ = tokio::join!(http_handle, grpc_handle);

    Ok(())
}
