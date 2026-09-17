use axum::{
    extract::{Multipart, State},
    response::{IntoResponse, Response},
    routing::post,
    Form, Router,
};
use axum::body::Body;
use serde::Deserialize;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use crate::engine::{InferenceTask, WorkerPool};

#[derive(Deserialize)]
pub struct SftForm {
    pub tts_text: String,
    pub spk_id: String,
}

#[derive(Deserialize)]
pub struct InstructForm {
    pub tts_text: String,
    pub spk_id: String,
    pub instruct_text: String,
}

pub fn create_router(pool: Arc<WorkerPool>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/inference_sft", post(inference_sft).get(inference_sft_get))
        .route("/inference_zero_shot", post(inference_zero_shot))
        .route("/inference_cross_lingual", post(inference_cross_lingual))
        .route("/inference_instruct", post(inference_instruct))
        .route("/inference_instruct2", post(inference_instruct2))
        .route("/inference_edit", post(inference_edit))
        .layer(cors)
        .with_state(pool)
}

async fn inference_sft(
    State(pool): State<Arc<WorkerPool>>,
    Form(payload): Form<SftForm>,
) -> Response {
    let task = InferenceTask::Sft {
        tts_text: payload.tts_text,
        spk_id: payload.spk_id,
    };
    handle_task(pool, task).await
}

async fn inference_sft_get(
    State(pool): State<Arc<WorkerPool>>,
    axum::extract::Query(payload): axum::extract::Query<SftForm>,
) -> Response {
    let task = InferenceTask::Sft {
        tts_text: payload.tts_text,
        spk_id: payload.spk_id,
    };
    handle_task(pool, task).await
}

async fn inference_zero_shot(
    State(pool): State<Arc<WorkerPool>>,
    mut multipart: Multipart,
) -> Response {
    let mut tts_text = String::new();
    let mut prompt_text = String::new();
    let mut prompt_audio = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "tts_text" {
            if let Ok(text) = field.text().await {
                tts_text = text;
            }
        } else if name == "prompt_text" {
            if let Ok(text) = field.text().await {
                prompt_text = text;
            }
        } else if name == "prompt_wav" || name == "prompt_speech" {
            if let Ok(bytes) = field.bytes().await {
                prompt_audio = bytes.to_vec();
            }
        }
    }

    let task = InferenceTask::ZeroShot {
        tts_text,
        prompt_text,
        prompt_audio,
    };
    handle_task(pool, task).await
}

async fn inference_cross_lingual(
    State(pool): State<Arc<WorkerPool>>,
    mut multipart: Multipart,
) -> Response {
    let mut tts_text = String::new();
    let mut prompt_audio = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "tts_text" {
            if let Ok(text) = field.text().await {
                tts_text = text;
            }
        } else if name == "prompt_wav" || name == "prompt_speech" {
            if let Ok(bytes) = field.bytes().await {
                prompt_audio = bytes.to_vec();
            }
        }
    }

    let task = InferenceTask::CrossLingual {
        tts_text,
        prompt_audio,
    };
    handle_task(pool, task).await
}

async fn inference_instruct(
    State(pool): State<Arc<WorkerPool>>,
    Form(payload): Form<InstructForm>,
) -> Response {
    let task = InferenceTask::Instruct {
        tts_text: payload.tts_text,
        spk_id: payload.spk_id,
        instruct_text: payload.instruct_text,
    };
    handle_task(pool, task).await
}

async fn inference_instruct2(
    State(pool): State<Arc<WorkerPool>>,
    mut multipart: Multipart,
) -> Response {
    let mut tts_text = String::new();
    let mut instruct_text = String::new();
    let mut prompt_audio = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "tts_text" {
            if let Ok(text) = field.text().await {
                tts_text = text;
            }
        } else if name == "instruct_text" {
            if let Ok(text) = field.text().await {
                instruct_text = text;
            }
        } else if name == "prompt_wav" || name == "prompt_speech" {
            if let Ok(bytes) = field.bytes().await {
                prompt_audio = bytes.to_vec();
            }
        }
    }

    let task = InferenceTask::Instruct2 {
        tts_text,
        instruct_text,
        prompt_audio,
    };
    handle_task(pool, task).await
}

async fn inference_edit(
    State(pool): State<Arc<WorkerPool>>,
    mut multipart: Multipart,
) -> Response {
    let mut target_text = String::new();
    let mut original_text = String::new();
    let mut original_speech = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "target_text" {
            if let Ok(text) = field.text().await {
                target_text = text;
            }
        } else if name == "original_text" {
            if let Ok(text) = field.text().await {
                original_text = text;
            }
        } else if name == "original_speech" || name == "prompt_wav" {
            if let Ok(bytes) = field.bytes().await {
                original_speech = bytes.to_vec();
            }
        }
    }

    let task = InferenceTask::Edit {
        target_text,
        original_text,
        original_speech,
    };
    handle_task(pool, task).await
}

async fn handle_task(pool: Arc<WorkerPool>, task: InferenceTask) -> Response {
    match pool.process_task(task).await {
        Ok(stream) => {
            let body = Body::from_stream(stream);
            Response::builder()
                .header("Content-Type", "audio/x-wav")
                .body(body)
                .unwrap_or_else(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to create response").into_response())
        }
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error processing request: {}", err),
        )
            .into_response(),
    }
}
