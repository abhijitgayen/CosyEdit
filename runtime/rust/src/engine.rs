use tokio::sync::mpsc;
use bytes::Bytes;
use futures_util::stream::Stream;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum InferenceTask {
    Sft {
        tts_text: String,
        spk_id: String,
    },
    ZeroShot {
        tts_text: String,
        prompt_text: String,
        prompt_audio: Vec<u8>,
    },
    CrossLingual {
        tts_text: String,
        prompt_audio: Vec<u8>,
    },
    Instruct {
        tts_text: String,
        spk_id: String,
        instruct_text: String,
    },
    Instruct2 {
        tts_text: String,
        instruct_text: String,
        prompt_audio: Vec<u8>,
    },
    Edit {
        target_text: String,
        original_text: String,
        original_speech: Vec<u8>,
    },
}

pub type AudioStream = Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send + 'static>>;

/// Parallel worker pool & dispatcher for handling concurrent audio synthesis/editing requests
#[derive(Clone)]
pub struct WorkerPool {
    backend_url: Option<String>,
    reqwest_client: reqwest::Client,
}

impl WorkerPool {
    pub fn new(backend_url: Option<String>) -> Self {
        Self {
            backend_url,
            reqwest_client: reqwest::Client::new(),
        }
    }

    /// Processes an inference task asynchronously and returns an audio byte stream
    pub async fn process_task(&self, task: InferenceTask) -> Result<AudioStream, String> {
        if let Some(ref backend) = self.backend_url {
            self.forward_to_backend(backend, task).await
        } else {
            // Simulated local ONNX / parallel worker inference stream
            Ok(Self::simulate_inference_stream(task))
        }
    }

    async fn forward_to_backend(&self, backend: &str, task: InferenceTask) -> Result<AudioStream, String> {
        let client = &self.reqwest_client;
        let resp = match task {
            InferenceTask::Sft { tts_text, spk_id } => {
                let params = [("tts_text", tts_text), ("spk_id", spk_id)];
                client.post(format!("{}/inference_sft", backend))
                    .form(&params)
                    .send()
                    .await
            }
            InferenceTask::ZeroShot { tts_text, prompt_text, prompt_audio } => {
                let part = reqwest::multipart::Part::bytes(prompt_audio).file_name("prompt.wav");
                let form = reqwest::multipart::Form::new()
                    .text("tts_text", tts_text)
                    .text("prompt_text", prompt_text)
                    .part("prompt_wav", part);
                client.post(format!("{}/inference_zero_shot", backend))
                    .multipart(form)
                    .send()
                    .await
            }
            InferenceTask::CrossLingual { tts_text, prompt_audio } => {
                let part = reqwest::multipart::Part::bytes(prompt_audio).file_name("prompt.wav");
                let form = reqwest::multipart::Form::new()
                    .text("tts_text", tts_text)
                    .part("prompt_wav", part);
                client.post(format!("{}/inference_cross_lingual", backend))
                    .multipart(form)
                    .send()
                    .await
            }
            InferenceTask::Instruct { tts_text, spk_id, instruct_text } => {
                let params = [("tts_text", tts_text), ("spk_id", spk_id), ("instruct_text", instruct_text)];
                client.post(format!("{}/inference_instruct", backend))
                    .form(&params)
                    .send()
                    .await
            }
            InferenceTask::Instruct2 { tts_text, instruct_text, prompt_audio } => {
                let part = reqwest::multipart::Part::bytes(prompt_audio).file_name("prompt.wav");
                let form = reqwest::multipart::Form::new()
                    .text("tts_text", tts_text)
                    .text("instruct_text", instruct_text)
                    .part("prompt_wav", part);
                client.post(format!("{}/inference_instruct2", backend))
                    .multipart(form)
                    .send()
                    .await
            }
            InferenceTask::Edit { target_text, original_text, original_speech } => {
                let part = reqwest::multipart::Part::bytes(original_speech).file_name("original.wav");
                let form = reqwest::multipart::Form::new()
                    .text("target_text", target_text)
                    .text("original_text", original_text)
                    .part("original_speech", part);
                client.post(format!("{}/inference_edit", backend))
                    .multipart(form)
                    .send()
                    .await
            }
        }.map_err(|e| format!("Backend request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("Backend error status: {}", resp.status()));
        }

        let stream = resp.bytes_stream();
        let mapped_stream = futures_util::StreamExt::map(stream, |res| {
            res.map_err(|e| format!("Stream error: {}", e))
        });

        Ok(Box::pin(mapped_stream))
    }

    fn simulate_inference_stream(_task: InferenceTask) -> AudioStream {
        let (tx, rx) = mpsc::channel(10);
        tokio::spawn(async move {
            for _ in 0..5 {
                let mock_chunk = Bytes::from(vec![0u8; 1600]);
                if tx.send(Ok(mock_chunk)).await.is_err() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
