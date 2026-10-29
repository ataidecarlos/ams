use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;

use rusqlite::{params, Connection};

use crate::db;
use crate::embeddings;

const CURATOR: &str = "curator";
const LOCK_LEASE_SECS: i64 = 1800;
/// Consecutive failures before the loop backs off, so a dead API key does not
/// hammer the endpoint every minute.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

pub struct CuratorConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub batch: i64,
    pub interval_secs: i64,
    pub digest_max_chars: usize,
}

impl CuratorConfig {
    /// `None` when `AMS_ANTHROPIC_API_KEY` is unset: curation is disabled and
    /// memories stay provisional, still fully searchable.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("AMS_ANTHROPIC_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        Some(Self {
            api_key,
            model: std::env::var("AMS_CURATOR_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".into()),
            base_url: std::env::var("AMS_ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".into()),
            batch: env_i64("AMS_CURATE_BATCH", 20),
            interval_secs: env_i64("AMS_CURATE_INTERVAL_SECS", 7200),
            digest_max_chars: 2200,
        })
    }
}

fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Debug, Default)]
pub struct Report {
    pub skipped: bool,
    pub batch: usize,
    pub updated: usize,
    pub archived: usize,
    pub promoted: usize,
    pub digests: usize,
}

/// Run one curation pass. Returns `skipped` when another process holds the
/// lock. Nothing is advanced on failure: the same batch is retried next time.
pub fn run_once(conn: &Connection, cfg: &CuratorConfig) -> Result<Report, String> {
    let holder = format!(
        "pid{}@{}",
        std::process::id(),
        std::env::var("COMPUTERNAME").unwrap_or_else(|_| "host".into())
    );
    if !db::acquire_lock(conn, CURATOR, &holder, LOCK_LEASE_SECS)
        .map_err(|e| format!("lock failed: {e}"))?
    {
        return Ok(Report {
            skipped: true,
            ..Default::default()
        });
    }
    let result = run_locked(conn, cfg);
    let _ = db::release_lock(conn, CURATOR, &holder);
    result
}

fn run_locked(conn: &Connection, cfg: &CuratorConfig) -> Result<Report, String> {
    let pending = db::pending_raw(conn, cfg.batch).map_err(|e| e.to_string())?;
    if pending.is_empty() {
        db::set_consumer(conn, CURATOR, 0, db::now()).map_err(|e| e.to_string())?;
        return Ok(Report::default());
    }

    // Each raw record already has a provisional memory (searchable since the
    // moment it was written). Candidates add related existing memories so the
    // curator can merge instead of duplicating.
    let mut new_memories: Vec<db::Memory> = Vec::new();
    let mut candidates: HashMap<i64, db::Memory> = HashMap::new();
    for raw in &pending {
        let mems = db::memories_for_raw(conn, raw.id).map_err(|e| e.to_string())?;
        for m in mems {
            new_memories.push(m);
        }
    }
    for m in &new_memories {
        if let Some(v) = db::vector_for_memory(conn, m.id) {
            if let Ok(neighbors) = db::knn(conn, &v, &m.agent, None, 5) {
                for (id, _) in neighbors {
                    if id == m.id || candidates.contains_key(&id) {
                        continue;
                    }
                    if let Ok(Some(n)) = db::get_memory(conn, id) {
                        if n.status != "archived" {
                            candidates.insert(id, n);
                        }
                    }
                }
            }
        }
    }

    let batch_agents: HashSet<String> = pending.iter().map(|r| r.agent.clone()).collect();
    let mut current_digests: Vec<(String, String)> = Vec::new();
    for agent in &batch_agents {
        if let Ok(Some(d)) = db::get_digest(conn, agent) {
            current_digests.push((agent.clone(), d));
        }
    }

    let (system, user) = build_prompt(&pending, &new_memories, &candidates, &current_digests);
    let text = call_api(cfg, &system, &user)?;
    let output = parse_output(&text)?;

    let raw_ids: Vec<i64> = pending.iter().map(|r| r.id).collect();
    let max_raw_id = raw_ids.iter().copied().max().unwrap_or(0);
    let mut report = Report {
        batch: pending.len(),
        ..Default::default()
    };
    let mut reembed: Vec<(i64, String)> = Vec::new();
    let mut touched: HashSet<i64> = HashSet::new();

    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;

    for op in &output.operations {
        let Some(memory_id) = op.memory_id else { continue };
        let Ok(Some(current)) = db::get_memory(&tx, memory_id) else {
            continue; // unknown id: ignore rather than fail the whole batch
        };
        match op.op.as_str() {
            "update" => {
                let content = op
                    .content
                    .as_deref()
                    .map(str::trim)
                    .filter(|c| !c.is_empty())
                    .unwrap_or(&current.content)
                    .to_string();
                let tags = op.tags.clone().unwrap_or_else(|| current.tags.clone());
                let kind = op
                    .kind
                    .clone()
                    .filter(|k| is_kind(k))
                    .unwrap_or_else(|| current.kind.clone());
                let scope = op
                    .scope
                    .clone()
                    .filter(|s| is_scope(s))
                    .unwrap_or_else(|| current.scope.clone());
                let confidence = op
                    .confidence
                    .map(|c| c.clamp(0.0, 1.0))
                    .unwrap_or(current.confidence);
                let expires_at = match op.expires_at {
                    None => current.expires_at,
                    Some(None) => None,
                    Some(Some(v)) => Some(v),
                };
                let content_changed = content != current.content;
                db::update_memory(
                    &tx, memory_id, &content, &tags, &kind, &scope, confidence, expires_at,
                    "curator",
                )
                .map_err(|e| e.to_string())?;
                if content_changed {
                    reembed.push((memory_id, content));
                }
                touched.insert(memory_id);
                report.updated += 1;
            }
            "archive" => {
                db::archive_memory(&tx, memory_id, op.superseded_by)
                    .map_err(|e| e.to_string())?;
                touched.insert(memory_id);
                report.archived += 1;
            }
            other => eprintln!("ams: curator: ignoring unknown op {:?}", other),
        }
    }

    // Safety net: a provisional memory the model did not resolve becomes
    // canonical as written, so nothing is stuck provisioning forever.
    for m in &new_memories {
        if m.status == "provisional" && !touched.contains(&m.id) {
            let changed = tx
                .execute(
                    "UPDATE memories SET status = 'canonical', updated_at = ?2
                     WHERE id = ?1 AND status = 'provisional'",
                    params![m.id, db::now()],
                )
                .map_err(|e| e.to_string())?;
            report.promoted += changed as usize;
        }
    }

    db::mark_raw_consumed(&tx, &raw_ids).map_err(|e| e.to_string())?;
    db::set_consumer(&tx, CURATOR, max_raw_id, db::now()).map_err(|e| e.to_string())?;

    for d in &output.digests {
        let content = db::truncate_chars(d.content.trim(), cfg.digest_max_chars);
        if content.is_empty() {
            continue;
        }
        db::set_digest(&tx, &d.agent, &content).map_err(|e| e.to_string())?;
        report.digests += 1;
    }

    tx.commit().map_err(|e| e.to_string())?;

    // Embedding is slow and CPU-bound; keep it outside the transaction so
    // concurrent writers are never blocked by ONNX inference.
    if !reembed.is_empty() {
        let texts: Vec<String> = reembed.iter().map(|(_, c)| c.clone()).collect();
        if let Some(vectors) = embeddings::embed_documents(&texts) {
            for ((id, _), v) in reembed.iter().zip(vectors) {
                if let Err(e) = db::set_vector(conn, *id, &v) {
                    eprintln!("ams: curator: failed to update vector for {}: {e}", id);
                }
            }
        }
    }

    Ok(report)
}

pub fn is_kind(k: &str) -> bool {
    matches!(k, "troubleshoot" | "correction" | "fact" | "procedure")
}

pub fn is_scope(s: &str) -> bool {
    matches!(s, "own" | "shared")
}

// ---------------------------------------------------------------- prompt

const SYSTEM_PROMPT: &str = r#"You are the memory curator for AMS, a shared memory system used by AI coding agents.
You receive newly captured raw memories (each backed by an existing provisional memory record) plus related existing memories. You return JSON operations that refine and merge them into canonical memory.

Rules:
- Resolve every provisional memory in new_memories: keep it with an "update" op (improve wording if useful), or "archive" it because an existing memory covers it.
- Merge duplicates: one memory per fact. Keep one record, archive the other(s) with superseded_by set to the kept id.
- Human corrections (kind "correction") are authoritative. They supersede conflicting older memories: archive the conflicting one with superseded_by pointing at the correction.
- Set scope "shared" only for project-independent facts (user preferences, agent behavior, tooling) useful to every agent. Otherwise keep "own".
- If the same troubleshoot fix appears a second time, convert the kept memory to kind "procedure".
- Ephemeral content (temporary workarounds, one-off state) gets expires_at set to a unix timestamp about 30 days ahead. Durable facts get expires_at null, which clears any existing expiry.
- confidence: 0.9 human correction or verified fix, 0.7 agent-discovered fix, 0.5 uncertain.
- Digests: for each agent in the batch, rewrite the digest (max 2200 characters) so it reflects current memory: durable facts, active corrections, useful procedures. Dense prose or tight bullets. Drop stale details.
- Output ONLY a JSON object. No prose, no markdown fences.

JSON shape:
{"operations":[{"op":"update","memory_id":1,"content":"...","tags":"space separated","kind":"fact|troubleshoot|correction|procedure","scope":"own|shared","confidence":0.7,"expires_at":null},{"op":"archive","memory_id":2,"superseded_by":1}],"digests":[{"agent":"name","content":"..."}]}"#;

fn build_prompt(
    pending: &[db::RawRecord],
    new_memories: &[db::Memory],
    candidates: &HashMap<i64, db::Memory>,
    digests: &[(String, String)],
) -> (String, String) {
    let raw_by_id: HashMap<i64, &db::RawRecord> = pending.iter().map(|r| (r.id, r)).collect();
    let new: Vec<serde_json::Value> = new_memories
        .iter()
        .map(|m| {
            let raw = raw_by_id.get(&m.raw_id);
            serde_json::json!({
                "memory_id": m.id,
                "raw_id": m.raw_id,
                "agent": m.agent,
                "project": m.project,
                "kind": m.kind,
                "scope": m.scope,
                "content": db::truncate_chars(m.content.trim(), 1500),
                "source_session": raw.map(|r| r.source_session.as_str()).unwrap_or(""),
                "captured_at": m.created_at,
            })
        })
        .collect();
    let mut cands: Vec<&db::Memory> = candidates.values().collect();
    cands.sort_by_key(|m| m.id);
    let cand_json: Vec<serde_json::Value> = cands
        .iter()
        .map(|m| {
            serde_json::json!({
                "memory_id": m.id,
                "agent": m.agent,
                "project": m.project,
                "kind": m.kind,
                "scope": m.scope,
                "status": m.status,
                "content": db::truncate_chars(m.content.trim(), 600),
            })
        })
        .collect();
    let digest_json: Vec<serde_json::Value> = digests
        .iter()
        .map(|(a, c)| serde_json::json!({"agent": a, "content": c}))
        .collect();

    let user = serde_json::json!({
        "new_memories": new,
        "related_existing_memories": cand_json,
        "current_digests": digest_json,
    });
    (SYSTEM_PROMPT.to_string(), serde_json::to_string_pretty(&user).unwrap_or_default())
}

// ---------------------------------------------------------------- api

fn call_api(cfg: &CuratorConfig, system: &str, user: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "model": cfg.model,
        "max_tokens": 8000,
        "thinking": {"type": "adaptive"},
        "system": system,
        "messages": [{"role": "user", "content": user}],
    });
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(180)))
        .build()
        .into();
    let mut resp = agent
        .post(&format!("{}/v1/messages", cfg.base_url.trim_end_matches('/')))
        .header("x-api-key", &cfg.api_key)
        .header("anthropic-version", "2026-10-21")
        .send_json(&body)
        .map_err(|e| format!("anthropic request failed: {e}"))?;
    let v: serde_json::Value = resp
        .body_mut()
        .read_json()
        .map_err(|e| format!("anthropic response parse failed: {e}"))?;
    if let Some(err) = v.get("error") {
        return Err(format!("anthropic error: {err}"));
    }
    let text = v["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();
    if text.trim().is_empty() {
        return Err("anthropic returned no text content".into());
    }
    Ok(text)
}

// ---------------------------------------------------------------- parsing

#[derive(serde::Deserialize, Debug, Default)]
struct LlmOutput {
    #[serde(default)]
    operations: Vec<LlmOp>,
    #[serde(default)]
    digests: Vec<LlmDigest>,
}

#[derive(serde::Deserialize, Debug)]
struct LlmOp {
    op: String,
    #[serde(default)]
    memory_id: Option<i64>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tags: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    confidence: Option<f64>,
    /// Tri-state: absent = keep, null = clear, number = set. Outer `None` is
    /// "absent"; `Some(None)` is an explicit null.
    #[serde(default, deserialize_with = "double_option")]
    expires_at: Option<Option<i64>>,
    #[serde(default)]
    superseded_by: Option<i64>,
}

#[derive(serde::Deserialize, Debug)]
struct LlmDigest {
    agent: String,
    content: String,
}

/// Distinguishes an absent field (`None`) from an explicit `null` (`Some(None)`).
fn double_option<'de, D, T>(de: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    <Option<T> as serde::Deserialize>::deserialize(de).map(Some)
}

fn parse_output(text: &str) -> Result<LlmOutput, String> {
    let start = text.find('{').ok_or("no JSON object in response")?;
    let end = text.rfind('}').ok_or("no JSON object in response")?;
    if end < start {
        return Err("malformed JSON in response".into());
    }
    serde_json::from_str(&text[start..=end]).map_err(|e| format!("invalid curation JSON: {e}"))
}

// ---------------------------------------------------------------- loop

/// Background curation for the MCP process: runs when the interval elapsed or
/// enough raw records accumulated. Cross-process safety comes from the DB lock.
pub fn spawn(db_path: PathBuf, cfg: CuratorConfig) {
    std::thread::spawn(move || {
        let conn = match db::open_db_at(&db_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ams: curator thread cannot open db: {e}");
                return;
            }
        };
        let mut failures: u32 = 0;
        loop {
            std::thread::sleep(Duration::from_secs(60));
            let pending = db::pending_raw(&conn, cfg.batch)
                .map(|r| r.len() as i64)
                .unwrap_or(0);
            let last_run = db::get_consumer(&conn, CURATOR)
                .map(|(_, t)| t)
                .unwrap_or(0);
            let due = pending >= cfg.batch || db::now() - last_run >= cfg.interval_secs;
            if !due {
                continue;
            }
            match run_once(&conn, &cfg) {
                Ok(r) => {
                    failures = 0;
                    if r.batch > 0 {
                        eprintln!(
                            "ams: curated {} record(s): {} updated, {} archived, {} promoted, {} digest(s)",
                            r.batch, r.updated, r.archived, r.promoted, r.digests
                        );
                    }
                }
                Err(e) => {
                    failures += 1;
                    eprintln!("ams: curation failed ({} consecutive): {e}", failures);
                    if failures >= MAX_CONSECUTIVE_FAILURES {
                        eprintln!(
                            "ams: curator paused after {} failures; restart the MCP server to resume",
                            failures
                        );
                        return;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ops_and_digests() {
        let text = r#"Here is the result:
{"operations":[{"op":"update","memory_id":3,"content":"merged text","kind":"fact","scope":"shared","confidence":0.9,"expires_at":null,"tags":"t"},{"op":"archive","memory_id":4,"superseded_by":3}],"digests":[{"agent":"a","content":"digest"}]}"#;
        let out = parse_output(text).unwrap();
        assert_eq!(out.operations.len(), 2);
        assert_eq!(out.operations[0].memory_id, Some(3));
        assert_eq!(out.operations[1].superseded_by, Some(3));
        assert_eq!(out.digests[0].agent, "a");
    }

    #[test]
    fn rejects_non_json() {
        assert!(parse_output("sorry, I cannot").is_err());
    }

    #[test]
    fn expires_at_tri_state() {
        // Absent -> None (keep), null -> Some(None) (clear), number -> set.
        let absent: LlmOp = serde_json::from_str(r#"{"op":"update","memory_id":1}"#).unwrap();
        assert!(absent.expires_at.is_none());
        let cleared: LlmOp =
            serde_json::from_str(r#"{"op":"update","memory_id":1,"expires_at":null}"#).unwrap();
        assert!(matches!(cleared.expires_at, Some(None)));
        let set: LlmOp =
            serde_json::from_str(r#"{"op":"update","memory_id":1,"expires_at":123}"#).unwrap();
        assert_eq!(set.expires_at, Some(Some(123)));
    }
}
