use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use cosyedit_runtime::engine::WorkerPool;
use cosyedit_runtime::http::create_router;

#[tokio::test]
async fn test_http_sft_endpoint() {
    let pool = Arc::new(WorkerPool::new(None));
    let app = create_router(pool);

    let req: Request<Body> = Request::builder()
        .method("POST")
        .uri("/inference_sft")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from("tts_text=Hello+world&spk_id=spk01"))
        .unwrap();

    let response = app.oneshot(req).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!body_bytes.is_empty());
}

#[tokio::test]
async fn test_http_edit_endpoint() {
    let pool = Arc::new(WorkerPool::new(None));
    let app = create_router(pool);

    let boundary = "------------------------boundary123456";
    let body_data = format!(
        "--{0}\r\nContent-Disposition: form-data; name=\"target_text\"\r\n\r\nHello edited\r\n\
         --{0}\r\nContent-Disposition: form-data; name=\"original_text\"\r\n\r\nHello original\r\n\
         --{0}\r\nContent-Disposition: form-data; name=\"original_speech\"; filename=\"orig.wav\"\r\nContent-Type: audio/wav\r\n\r\nfakeaudio\r\n\
         --{0}--\r\n",
        boundary
    );

    let req: Request<Body> = Request::builder()
        .method("POST")
        .uri("/inference_edit")
        .header("content-type", format!("multipart/form-data; boundary={}", boundary))
        .body(Body::from(body_data))
        .unwrap();

    let response = app.oneshot(req).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!body_bytes.is_empty());
}
