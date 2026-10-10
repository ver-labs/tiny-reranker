use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const MAX_FRAME: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    pub document_id: String,
    pub text: String,
}

fn default_top_k() -> usize {
    10
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Request {
    pub request_id: String,
    pub query: String,
    pub documents: Vec<Document>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

impl Request {
    pub fn validate(&self) -> Result<(), &'static str> {
        fn length(s: &str, max: usize) -> bool {
            let n = s.chars().count();
            n > 0 && n <= max
        }
        if !length(&self.request_id, 256)
            || !length(&self.query, 2000)
            || self.documents.is_empty()
            || self.documents.len() > 256
            || self.top_k == 0
            || self.top_k > 256
        {
            return Err("Invalid request ID, query, document count, or top_k");
        }
        let mut ids = HashSet::new();
        for d in &self.documents {
            if !length(&d.document_id, 256)
                || !length(&d.text, 20000)
                || !ids.insert(&d.document_id)
            {
                return Err("Invalid or duplicate document ID, or invalid document text");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RankedDocument {
    pub document_id: String,
    pub score: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Ready {
    #[serde(rename = "type")]
    pub kind: String,
    pub protocol_version: u32,
    pub backend: String,
    pub pair_batch_size: usize,
    pub max_length: usize,
    pub checkpoint_sha256: String,
}

#[derive(Debug, Deserialize)]
pub struct WorkerResponse {
    #[serde(rename = "type")]
    pub kind: String,
    pub seq: u64,
    pub request_id: String,
    pub ranking: Option<Vec<RankedDocument>>,
    pub error: Option<String>,
}

pub fn validate_ranking(request: &Request, ranked: &[RankedDocument]) -> Result<(), String> {
    if ranked.len() != request.top_k.min(request.documents.len()) {
        return Err("Incorrect worker ranking length".into());
    }
    let valid: HashSet<_> = request
        .documents
        .iter()
        .map(|d| d.document_id.as_str())
        .collect();
    let mut seen = HashSet::new();
    for d in ranked {
        if !d.score.is_finite()
            || !valid.contains(d.document_id.as_str())
            || !seen.insert(&d.document_id)
        {
            return Err("Invalid worker score or document ID".into());
        }
    }
    for pair in ranked.windows(2) {
        if pair[0].score < pair[1].score
            || (pair[0].score == pair[1].score && pair[0].document_id > pair[1].document_id)
        {
            return Err("Worker ranking is not sorted".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_and_nonfinite_rankings() {
        let mut r = Request {
            request_id: "id".into(),
            query: "q".into(),
            documents: vec![Document {
                document_id: "a".into(),
                text: "x".into(),
            }],
            top_k: 1,
        };
        assert!(r.validate().is_ok());
        assert!(
            validate_ranking(
                &r,
                &[RankedDocument {
                    document_id: "a".into(),
                    score: f64::NAN
                }]
            )
            .is_err()
        );
        r.documents.push(r.documents[0].clone());
        assert!(r.validate().is_err());
    }
}
