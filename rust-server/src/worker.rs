use crate::protocol::{MAX_FRAME, Ready, Request, WorkerResponse};
use std::{path::Path, process::Stdio};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};

pub struct Worker {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    pub ready: Ready,
}

async fn read_frame(reader: &mut BufReader<ChildStdout>) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    (&mut *reader)
        .take((MAX_FRAME + 1) as u64)
        .read_until(b'\n', &mut data)
        .await
        .map_err(|e| e.to_string())?;
    if data.is_empty() {
        return Err("Worker exited or closed stdout".into());
    }
    if data.len() > MAX_FRAME || data.last() != Some(&b'\n') {
        return Err("Invalid or oversized worker frame".into());
    }
    Ok(data)
}

impl Worker {
    pub async fn spawn(
        python: &Path,
        root: &Path,
        script: Option<&Path>,
        backend: &str,
        batch: usize,
    ) -> Result<Self, String> {
        let mut cmd = Command::new(python);
        cmd.arg("-u");
        if let Some(script) = script {
            cmd.arg(script);
        } else {
            cmd.args(["-m", "experiments.E12.worker"]);
        }
        cmd.arg("--backend")
            .arg(backend)
            .arg("--pair-batch-size")
            .arg(batch.to_string())
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| e.to_string())?;
        let stdin = child.stdin.take().ok_or("No worker stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("No worker stdout")?);
        let wire =
            tokio::time::timeout(std::time::Duration::from_secs(60), read_frame(&mut stdout))
                .await
                .map_err(|_| "Worker readiness timed out")??;
        let ready: Ready = serde_json::from_slice(&wire).map_err(|e| e.to_string())?;
        if ready.kind != "ready"
            || ready.protocol_version != 1
            || ready.backend != backend
            || ready.pair_batch_size != batch
            || ready.max_length != 128
            || ready.checkpoint_sha256.len() != 64
            || !ready
                .checkpoint_sha256
                .chars()
                .all(|c| c.is_ascii_hexdigit())
        {
            return Err("Worker readiness metadata mismatch".into());
        }
        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout,
            ready,
        })
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub async fn infer(&mut self, seq: u64, request: &Request) -> Result<WorkerResponse, String> {
        let payload = serde_json::json!({"type":"rerank","seq":seq,"request":request});
        let mut wire = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
        wire.push(b'\n');
        if wire.len() > MAX_FRAME {
            return Err("Request exceeds worker frame limit".into());
        }
        let stdin = self.stdin.as_mut().ok_or("Worker stdin closed")?;
        stdin.write_all(&wire).await.map_err(|e| e.to_string())?;
        stdin.flush().await.map_err(|e| e.to_string())?;
        let bytes = read_frame(&mut self.stdout).await?;
        let response: WorkerResponse = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if response.kind != "result"
            || response.seq != seq
            || response.request_id != request.request_id
            || (response.error.is_some() == response.ranking.is_some())
        {
            return Err("Worker response sequence/ID/type mismatch".into());
        }
        Ok(response)
    }

    pub async fn close(&mut self) {
        drop(self.stdin.take());
        if tokio::time::timeout(std::time::Duration::from_secs(5), self.child.wait())
            .await
            .is_err()
        {
            let _ = self.child.kill().await;
            let _ = self.child.wait().await;
        }
    }
}
