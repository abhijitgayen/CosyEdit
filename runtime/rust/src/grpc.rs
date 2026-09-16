use tonic::{Request, Response, Status};
use tokio_stream::Stream;
use std::pin::Pin;
use std::sync::Arc;
use futures_util::StreamExt;

use crate::engine::{InferenceTask, WorkerPool};

pub mod cosyvoice_proto {
    tonic::include_proto!("cosyvoice");
}

use cosyvoice_proto::cosy_voice_server::CosyVoice;
use cosyvoice_proto::request::RequestPayload;
use cosyvoice_proto::{Request as ProtoRequest, Response as ProtoResponse};

pub struct CosyVoiceGrpcService {
    worker_pool: Arc<WorkerPool>,
}

impl CosyVoiceGrpcService {
    pub fn new(worker_pool: Arc<WorkerPool>) -> Self {
        Self { worker_pool }
    }
}

#[tonic::async_trait]
impl CosyVoice for CosyVoiceGrpcService {
    type InferenceStream = Pin<Box<dyn Stream<Item = Result<ProtoResponse, Status>> + Send + 'static>>;

    async fn inference(
        &self,
        request: Request<ProtoRequest>,
    ) -> Result<Response<Self::InferenceStream>, Status> {
        let req = request.into_inner();
        let payload = req.request_payload.ok_or_else(|| Status::invalid_argument("Missing request payload"))?;

        let task = match payload {
            RequestPayload::SftRequest(sft) => InferenceTask::Sft {
                tts_text: sft.tts_text,
                spk_id: sft.spk_id,
            },
            RequestPayload::ZeroShotRequest(zs) => InferenceTask::ZeroShot {
                tts_text: zs.tts_text,
                prompt_text: zs.prompt_text,
                prompt_audio: zs.prompt_audio,
            },
            RequestPayload::CrossLingualRequest(cl) => InferenceTask::CrossLingual {
                tts_text: cl.tts_text,
                prompt_audio: cl.prompt_audio,
            },
            RequestPayload::InstructRequest(inst) => InferenceTask::Instruct {
                tts_text: inst.tts_text,
                spk_id: inst.spk_id,
                instruct_text: inst.instruct_text,
            },
        };

        let audio_stream = self
            .worker_pool
            .process_task(task)
            .await
            .map_err(|e| Status::internal(e))?;

        let grpc_stream = audio_stream.map(|chunk_res| match chunk_res {
            Ok(bytes) => Ok(ProtoResponse {
                tts_audio: bytes.to_vec(),
            }),
            Err(e) => Err(Status::internal(e)),
        });

        Ok(Response::new(Box::pin(grpc_stream)))
    }
}
