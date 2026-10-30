use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// Model identity, stored alongside vectors so a model swap invalidates them.
pub const MODEL_NAME: &str = "BGESmallENV15";
/// Bump when the text fed to the model changes (not when the model changes).
const TEXT_VERSION: u32 = 2;

/// BGE v1.5 recipe: passages are embedded as-is; only queries carry the
/// retrieval instruction. (E5-style `passage:`/`query:` prefixes are for a
/// different model family and cost retrieval quality here.)
const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

/// BGE-small-en-v1.5 output size. `db.rs` hardcodes this in the vec0 schema;
/// a different-dimension model needs a schema migration.
pub const DIM: usize = 384;

pub fn model_id() -> String {
    format!("{}:v{}", MODEL_NAME, TEXT_VERSION)
}

static MODEL: OnceLock<Mutex<Option<TextEmbedding>>> = OnceLock::new();
static UNAVAILABLE: AtomicBool = AtomicBool::new(false);

/// False when embeddings have been disabled for this process.
pub fn available() -> bool {
    !UNAVAILABLE.load(Ordering::Relaxed)
}

/// Records are capped at ~2000 chars at write time, so the default 512-token
/// window never truncates a memory. Writes are interactive (one record per
/// call), not bulk sweeps, so the extra compute is a non-issue.
const MAX_TOKENS: usize = 512;

fn init_model() -> Option<TextEmbedding> {
    let mut opts = TextInitOptions::new(EmbeddingModel::BGESmallENV15)
        .with_show_download_progress(false)
        .with_max_length(MAX_TOKENS);
    if let Some(cache) = cache_dir() {
        opts = opts.with_cache_dir(cache);
    }
    match TextEmbedding::try_new(opts) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("ams: embedding model unavailable ({}); using full-text search only", e);
            UNAVAILABLE.store(true, Ordering::Relaxed);
            None
        }
    }
}

/// `AMS_MODEL_CACHE` wins, then the platform cache dir.
fn cache_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("AMS_MODEL_CACHE")
        && !p.is_empty()
    {
        return Some(std::path::PathBuf::from(p));
    }
    dirs::cache_dir().map(|d| d.join("ams").join("models"))
}

fn with_model<R>(f: impl FnOnce(&mut TextEmbedding) -> R) -> Option<R> {
    if !available() {
        return None;
    }
    let slot = MODEL.get_or_init(|| Mutex::new(init_model()));
    let mut guard = slot.lock().ok()?;
    guard.as_mut().map(f)
}

/// Load the model now instead of on first use.
pub fn warm() -> bool {
    with_model(|_| ()).is_some()
}

/// Embed documents for indexing. BGE v1.5 uses no passage-side prefix.
pub fn embed_documents(texts: &[String]) -> Option<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Some(Vec::new());
    }
    with_model(|m| m.embed(texts.to_vec(), None).ok())?
}

/// Embed a single query, with BGE's retrieval instruction.
pub fn embed_query(text: &str) -> Option<Vec<f32>> {
    let prefixed = format!("{QUERY_INSTRUCTION}{text}");
    with_model(|m| m.embed(vec![prefixed], None).ok())?.and_then(|v| v.into_iter().next())
}

pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_roundtrip() {
        let v = vec![1.0f32, -0.5, 0.25, 3.0];
        assert_eq!(from_blob(&to_blob(&v)), v);
    }
}
