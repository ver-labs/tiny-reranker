mod protocol;
mod worker;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use protocol::{MAX_FRAME, RankedDocument, Request, validate_ranking};
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, watch};
use worker::Worker;

#[derive(Clone, Debug)]
struct Failure {
    status: StatusCode,
    code: &'static str,
    detail: String,
    request_id: Option<String>,
}
impl Failure {
    fn new(
        status: StatusCode,
        code: &'static str,
        detail: impl Into<String>,
        id: Option<String>,
    ) -> Self {
        Self {
            status,
            code,
            detail: detail.into(),
            request_id: id,
        }
    }
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.status,Json(serde_json::json!({"code":self.code,"detail":self.detail,"request_id":self.request_id}))).into_response()
    }
}

#[derive(Debug, Serialize)]
struct RankingResponse {
    request_id: String,
    backend: String,
    ranking: Vec<RankedDocument>,
    queue_ms: f64,
    processing_ms: f64,
}
struct Job {
    request: Request,
    reply: oneshot::Sender<Result<RankingResponse, Failure>>,
    queued: Instant,
}
#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<Job>,
    ready: Arc<AtomicBool>,
    metadata: Arc<serde_json::Value>,
    request_timeout: Duration,
}

async fn health(State(state): State<AppState>) -> Response {
    let ready = state.ready.load(Ordering::Acquire);
    let mut data = (*state.metadata).clone();
    data["ready"] = ready.into();
    data["queue_depth"] = (state.tx.max_capacity() - state.tx.capacity()).into();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(data),
    )
        .into_response()
}

async fn rerank(
    State(state): State<AppState>,
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Json<RankingResponse>, Failure> {
    let Json(request) = body.map_err(|e| {
        Failure::new(
            if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            },
            "invalid_request",
            e.body_text(),
            None,
        )
    })?;
    let id = Some(request.request_id.clone());
    request.validate().map_err(|e| {
        Failure::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            e,
            id.clone(),
        )
    })?;
    if !state.ready.load(Ordering::Acquire) {
        return Err(Failure::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_unavailable",
            "Model worker is unavailable",
            id,
        ));
    }
    let (reply, receiver) = oneshot::channel();
    state
        .tx
        .try_send(Job {
            request,
            reply,
            queued: Instant::now(),
        })
        .map_err(|e| {
            let full = matches!(e, mpsc::error::TrySendError::Full(_));
            Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                if full {
                    "queue_full"
                } else {
                    "worker_unavailable"
                },
                if full {
                    "Rerank queue is full"
                } else {
                    "Model worker is unavailable"
                },
                id.clone(),
            )
        })?;
    match tokio::time::timeout(state.request_timeout, receiver).await {
        Err(_) => Err(Failure::new(
            StatusCode::GATEWAY_TIMEOUT,
            "deadline_exceeded",
            "Rerank deadline exceeded",
            id,
        )),
        Ok(Err(_)) => Err(Failure::new(
            StatusCode::BAD_GATEWAY,
            "worker_failed",
            "Model worker stopped",
            id,
        )),
        Ok(Ok(value)) => value.map(Json),
    }
}

async fn manager(
    mut worker: Worker,
    mut rx: mpsc::Receiver<Job>,
    ready: Arc<AtomicBool>,
    mut shutdown: watch::Receiver<bool>,
    worker_timeout: Duration,
) {
    let mut seq = 0;
    let mut closing = false;
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    loop {
        let job = tokio::select! {
            biased;
            _=shutdown.changed(),if !closing=>{closing=true;ready.store(false,Ordering::Release);rx.close();continue;},
            job=rx.recv()=>match job {Some(job)=>job,None=>break},
            _=interval.tick(),if !closing=>{
                if !worker.alive() {eprintln!("Model worker exited while idle");break;}
                continue;
            }
        };
        if job.reply.is_closed() {
            continue;
        }
        seq += 1;
        let started = Instant::now();
        let response = tokio::time::timeout(worker_timeout, worker.infer(seq, &job.request)).await;
        let mut fatal = false;
        let result = match response {
            Ok(Ok(response)) if response.error.is_some() => Err(Failure::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                "Model inference failed",
                Some(job.request.request_id.clone()),
            )),
            Ok(Ok(response)) => {
                let ranking = response.ranking.unwrap();
                match validate_ranking(&job.request, &ranking) {
                    Ok(()) => Ok(RankingResponse {
                        request_id: job.request.request_id.clone(),
                        backend: worker.ready.backend.clone(),
                        ranking,
                        queue_ms: (started - job.queued).as_secs_f64() * 1000.,
                        processing_ms: started.elapsed().as_secs_f64() * 1000.,
                    }),
                    Err(e) => {
                        fatal = true;
                        Err(Failure::new(
                            StatusCode::BAD_GATEWAY,
                            "worker_protocol_error",
                            e,
                            Some(job.request.request_id.clone()),
                        ))
                    }
                }
            }
            other => {
                fatal = true;
                let detail = match other {
                    Ok(Err(e)) => e,
                    Err(_) => "Worker execution timeout".into(),
                    _ => unreachable!(),
                };
                Err(Failure::new(
                    StatusCode::BAD_GATEWAY,
                    "worker_failed",
                    detail,
                    Some(job.request.request_id.clone()),
                ))
            }
        };
        let _ = job.reply.send(result);
        // Even if the client timed out, finish reading this response before advancing.
        if fatal {
            break;
        }
    }
    ready.store(false, Ordering::Release);
    rx.close();
    while let Some(job) = rx.recv().await {
        let _ = job.reply.send(Err(Failure::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_unavailable",
            "Model worker is unavailable",
            Some(job.request.request_id),
        )));
    }
    worker.close().await;
}

struct Config {
    root: PathBuf,
    python: PathBuf,
    script: Option<PathBuf>,
    backend: String,
    port: u16,
    batch: usize,
    queue: usize,
    request_timeout: Duration,
    worker_timeout: Duration,
}
impl Config {
    fn parse() -> Result<Self, String> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .to_path_buf();
        let mut c = Self {
            python: root.join(".venv/bin/python"),
            root,
            script: None,
            backend: "mlx".into(),
            port: 8100,
            batch: 8,
            queue: 32,
            request_timeout: Duration::from_secs(30),
            worker_timeout: Duration::from_secs(60),
        };
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            if key == "--help" {
                println!(
                    "--backend mlx|pytorch --port 8100 --pair-batch-size 8 --queue-size 32 --timeout 30 --worker-timeout 60 --python PATH --root PATH"
                );
                std::process::exit(0);
            }
            let value = args.next().ok_or(format!("Missing value for {key}"))?;
            match key.as_str() {
                "--root" => c.root = value.into(),
                "--python" => c.python = value.into(),
                "--worker-script" => c.script = Some(value.into()),
                "--backend" => c.backend = value,
                "--port" => c.port = value.parse().map_err(|_| "Invalid port")?,
                "--pair-batch-size" => c.batch = value.parse().map_err(|_| "Invalid batch")?,
                "--queue-size" => c.queue = value.parse().map_err(|_| "Invalid queue size")?,
                "--timeout" | "--worker-timeout" => {
                    let seconds: f64 = value.parse().map_err(|_| "Invalid timeout")?;
                    if !seconds.is_finite() || seconds <= 0. || seconds > 3600. {
                        return Err("Timeout must be in (0,3600]".into());
                    }
                    if key == "--timeout" {
                        c.request_timeout = Duration::from_secs_f64(seconds);
                    } else {
                        c.worker_timeout = Duration::from_secs_f64(seconds);
                    }
                }
                _ => return Err(format!("Unknown argument {key}")),
            }
        }
        if !["mlx", "pytorch"].contains(&c.backend.as_str())
            || !(1..=64).contains(&c.batch)
            || c.queue == 0
            || c.queue > 1024
            || c.worker_timeout < c.request_timeout
        {
            return Err("Invalid backend, batch, queue, or timeout settings".into());
        }
        Ok(c)
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::select! {_=tokio::signal::ctrl_c()=>{},_=term.recv()=>{}}
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse().map_err(std::io::Error::other)?;
    let worker = Worker::spawn(
        &config.python,
        &config.root,
        config.script.as_deref(),
        &config.backend,
        config.batch,
    )
    .await
    .map_err(std::io::Error::other)?;
    let ready = Arc::new(AtomicBool::new(true));
    let (tx, rx) = mpsc::channel(config.queue);
    let metadata = Arc::new(
        serde_json::json!({"coordinator":"rust","backend":worker.ready.backend,
        "pair_batch_size":worker.ready.pair_batch_size,"max_length":worker.ready.max_length,
        "checkpoint_sha256":worker.ready.checkpoint_sha256,"worker_pid":worker.pid(),"queue_capacity":config.queue}),
    );
    let state = AppState {
        tx,
        ready: ready.clone(),
        metadata,
        request_timeout: config.request_timeout,
    };
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let mut http_shutdown = shutdown_rx.clone();
    let worker_task = tokio::spawn(manager(
        worker,
        rx,
        ready,
        shutdown_rx,
        config.worker_timeout,
    ));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", config.port)).await?;
    eprintln!("Rust serving ready at http://{}", listener.local_addr()?);
    let app = Router::new()
        .route("/health", get(health))
        .route("/rerank", post(rerank))
        .layer(DefaultBodyLimit::max(MAX_FRAME - 1024))
        .with_state(state);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = http_shutdown.changed().await;
        })
        .await?;
    worker_task.await?;
    Ok(())
}
