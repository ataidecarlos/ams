use rusqlite::{params, Connection, OptionalExtension, Row};
use std::path::{Path, PathBuf};

use crate::embeddings::DIM;

pub fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Char-boundary-safe truncation; appends an ellipsis when shortened.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// FNV-1a, 64-bit. Only used for exact duplicate detection, not security.
pub fn content_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:016x}", h)
}

/// sqlite-vec is compiled into the binary; register its entry point as a
/// process-wide auto-extension once, before the first connection opens.
fn register_sqlite_vec() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

const SCHEMA_V1: &str = "
CREATE TABLE raw_memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent TEXT NOT NULL,
  project TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'own',
  content TEXT NOT NULL,
  source_session TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  consumed_at INTEGER);

CREATE INDEX idx_raw_pending ON raw_memories(consumed_at, id);

CREATE TABLE memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent TEXT NOT NULL,
  project TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'own',
  content TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'provisional',
  confidence REAL NOT NULL DEFAULT 0.6,
  raw_id INTEGER NOT NULL REFERENCES raw_memories(id),
  supersedes_id INTEGER REFERENCES memories(id),
  superseded_by INTEGER REFERENCES memories(id),
  version INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  expires_at INTEGER,
  access_count INTEGER NOT NULL DEFAULT 0,
  last_access_at INTEGER);

CREATE INDEX idx_mem_scope ON memories(agent, project, status);
CREATE INDEX idx_mem_hash ON memories(content_hash);

CREATE VIRTUAL TABLE memories_fts USING fts5(
  content, tags,
  content='memories', content_rowid='id',
  tokenize='porter unicode61');

CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES('delete', old.id, old.content, old.tags);
END;
CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES('delete', old.id, old.content, old.tags);
  INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;

CREATE VIRTUAL TABLE memories_vec USING vec0(
  memory_id integer primary key,
  embedding float[384] distance_metric=cosine);

CREATE TABLE digest (
  agent TEXT PRIMARY KEY,
  content TEXT NOT NULL DEFAULT '',
  updated_at INTEGER NOT NULL DEFAULT 0);

CREATE TABLE consumers (
  name TEXT PRIMARY KEY,
  last_raw_id INTEGER NOT NULL DEFAULT 0,
  last_run_at INTEGER NOT NULL DEFAULT 0);

CREATE TABLE memory_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id INTEGER NOT NULL,
  version INTEGER NOT NULL,
  content TEXT NOT NULL,
  op TEXT NOT NULL,
  at INTEGER NOT NULL);

CREATE INDEX idx_versions_memory ON memory_versions(memory_id, version);

CREATE TABLE locks (
  name TEXT PRIMARY KEY,
  holder TEXT NOT NULL DEFAULT '',
  acquired_at INTEGER NOT NULL DEFAULT 0);

CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL);
";

// Feedback loop: agents upvote memories they validated and downvote ones they
// found wrong or outdated. Retrieval never votes — only explicit use does.
const SCHEMA_V2: &str = "
ALTER TABLE memories ADD COLUMN upvotes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN downvotes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN last_confirmed_at INTEGER;

CREATE TABLE IF NOT EXISTS memory_votes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id INTEGER NOT NULL REFERENCES memories(id),
  agent TEXT NOT NULL,
  session TEXT NOT NULL DEFAULT '',
  vote INTEGER NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  at INTEGER NOT NULL,
  UNIQUE(memory_id, agent, session));

CREATE INDEX IF NOT EXISTS idx_votes_memory ON memory_votes(memory_id);
";

// v3: per-project digests, one non-archived memory per (agent, content hash),
// and a dedupe of any rows written before the index existed.
const SCHEMA_V3: &str = "
CREATE TABLE digest_project (
  agent TEXT NOT NULL,
  project TEXT NOT NULL,
  content TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (agent, project));
INSERT OR REPLACE INTO digest_project (agent, project, content, updated_at)
  SELECT agent, '', content, updated_at FROM digest;
DROP TABLE digest;
ALTER TABLE digest_project RENAME TO digest;

UPDATE memories SET status = 'archived'
 WHERE status != 'archived' AND id NOT IN (
   SELECT MIN(id) FROM memories WHERE status != 'archived' GROUP BY agent, content_hash);

CREATE UNIQUE INDEX idx_mem_unique_hash
  ON memories(agent, content_hash) WHERE status != 'archived';
";

pub fn config_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("AMS_DB")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    let dir = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.join("ams").join("memory.db")
}

pub fn open_db() -> rusqlite::Result<Connection> {
    open_db_at(&config_db_path())
}

/// Identifier for one process's session, used when a caller does not supply
/// one. Keeps the "one vote per agent per session" rule meaningful even when
/// the host gives agents no session id.
pub fn new_session_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("p{}-{:x}", std::process::id(), nanos)
}

pub fn open_db_at(path: &Path) -> rusqlite::Result<Connection> {
    register_sqlite_vec();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.busy_timeout(std::time::Duration::from_millis(5000))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    let mut version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version == 0 {
        conn.execute_batch(SCHEMA_V1)?;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('vector_model', ?1)",
            params![crate::embeddings::model_id()],
        )?;
        conn.pragma_update(None, "user_version", 1)?;
        version = 1;
    }
    if version < 2 {
        conn.execute_batch(SCHEMA_V2)?;
        conn.pragma_update(None, "user_version", 2)?;
        version = 2;
    }
    if version < 3 {
        conn.execute_batch(SCHEMA_V3)?;
        conn.pragma_update(None, "user_version", 3)?;
    }
    // A model or text-recipe swap makes every stored vector a different
    // vector space. Warn once; `ams reembed` rebuilds them.
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'vector_model'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if stored.as_deref() != Some(crate::embeddings::model_id().as_str()) {
        eprintln!(
            "ams: stored vectors use `{}`, current model is `{}`; run `ams reembed`",
            stored.unwrap_or_default(),
            crate::embeddings::model_id()
        );
    }
    Ok(conn)
}

// ---------------------------------------------------------------- records

#[derive(Debug, Clone)]
pub struct RawRecord {
    pub id: i64,
    pub source_session: String,
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub id: i64,
    pub agent: String,
    pub project: String,
    pub kind: String,
    pub scope: String,
    pub content: String,
    pub tags: String,
    pub status: String,
    pub confidence: f64,
    pub raw_id: i64,
    pub supersedes_id: Option<i64>,
    pub superseded_by: Option<i64>,
    pub version: i64,
    pub created_at: i64,
    pub updated_at: i64,
    pub expires_at: Option<i64>,
    pub access_count: i64,
    pub last_access_at: Option<i64>,
    pub upvotes: i64,
    pub downvotes: i64,
    pub last_confirmed_at: Option<i64>,
}

const MEMORY_COLS: &str = "id, agent, project, kind, scope, content, tags, status, confidence, \
                           raw_id, supersedes_id, superseded_by, version, created_at, updated_at, \
                           expires_at, access_count, last_access_at, upvotes, downvotes, \
                           last_confirmed_at";

fn memory_from_row(r: &Row) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: r.get(0)?,
        agent: r.get(1)?,
        project: r.get(2)?,
        kind: r.get(3)?,
        scope: r.get(4)?,
        content: r.get(5)?,
        tags: r.get(6)?,
        status: r.get(7)?,
        confidence: r.get(8)?,
        raw_id: r.get(9)?,
        supersedes_id: r.get(10)?,
        superseded_by: r.get(11)?,
        version: r.get(12)?,
        created_at: r.get(13)?,
        updated_at: r.get(14)?,
        expires_at: r.get(15)?,
        access_count: r.get(16)?,
        last_access_at: r.get(17)?,
        upvotes: r.get(18)?,
        downvotes: r.get(19)?,
        last_confirmed_at: r.get(20)?,
    })
}

pub fn insert_raw(
    conn: &Connection,
    agent: &str,
    project: &str,
    kind: &str,
    scope: &str,
    content: &str,
    source_session: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO raw_memories (agent, project, kind, scope, content, source_session, content_hash, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![agent, project, kind, scope, content, source_session, content_hash(content), now()],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Insert a memory row plus its first version row. Caller owns the transaction.
pub fn insert_memory(
    conn: &Connection,
    raw_id: i64,
    agent: &str,
    project: &str,
    kind: &str,
    scope: &str,
    content: &str,
    tags: &str,
    status: &str,
    confidence: f64,
    expires_at: Option<i64>,
) -> rusqlite::Result<i64> {
    let t = now();
    conn.execute(
        "INSERT INTO memories (agent, project, kind, scope, content, tags, content_hash, status, confidence,
          raw_id, version, created_at, updated_at, expires_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1, ?11, ?11, ?12)",
        params![agent, project, kind, scope, content, tags, content_hash(content), status, confidence, raw_id, t, expires_at],
    )?;
    let id = conn.last_insert_rowid();
    insert_version(conn, id, 1, content, "created")?;
    Ok(id)
}

pub fn insert_version(
    conn: &Connection,
    memory_id: i64,
    version: i64,
    content: &str,
    op: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO memory_versions (memory_id, version, content, op, at) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![memory_id, version, content, op, now()],
    )?;
    Ok(())
}

/// Replace a memory's mutable fields and bump its version. `op` lands in
/// `memory_versions` so restore/history can tell edits from restores.
pub fn update_memory(
    conn: &Connection,
    id: i64,
    content: &str,
    tags: &str,
    kind: &str,
    scope: &str,
    confidence: f64,
    expires_at: Option<i64>,
    op: &str,
) -> rusqlite::Result<()> {
    // No-op updates (curator restating an unchanged memory) must not churn the
    // version log. A provisional memory still needs the status promotion.
    if let Some(cur) = get_memory(conn, id)?
        && cur.status == "canonical"
        && cur.content == content
        && cur.tags == tags
        && cur.kind == kind
        && cur.scope == scope
        && (cur.confidence - confidence).abs() < 1e-9
        && cur.expires_at == expires_at
    {
        return Ok(());
    }
    let version: i64 = conn.query_row("SELECT version FROM memories WHERE id = ?1", params![id], |r| {
        r.get(0)
    })?;
    let v = version + 1;
    conn.execute(
        "UPDATE memories SET content = ?2, tags = ?3, kind = ?4, scope = ?5, confidence = ?6,
           expires_at = ?7, updated_at = ?8, version = ?9, status = 'canonical', content_hash = ?10
         WHERE id = ?1",
        params![id, content, tags, kind, scope, confidence, expires_at, now(), v, content_hash(content)],
    )?;
    insert_version(conn, id, v, content, op)?;
    Ok(())
}

pub fn archive_memory(
    conn: &Connection,
    id: i64,
    superseded_by: Option<i64>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE memories SET status = 'archived', superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
        params![id, superseded_by, now()],
    )?;
    Ok(())
}

// ---------------------------------------------------------------- votes

/// Fresh memories expire without reinforcement; an upvote resets the timer.
/// `AMS_MEMORY_TTL_DAYS` (default 90) applies to every kind except procedures.
pub fn ttl_secs() -> i64 {
    let days: i64 = std::env::var("AMS_MEMORY_TTL_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|d| *d > 0)
        .unwrap_or(90);
    days * 86_400
}

/// Laplace-smoothed vote ratio in (-1, 1). Zero votes scores 0 (neutral).
pub fn usefulness(up: i64, down: i64) -> f32 {
    (up - down) as f32 / (up + down + 2) as f32
}

#[derive(Debug, Clone, Copy)]
pub struct VoteCounts {
    pub up: i64,
    pub down: i64,
}

/// One vote per (memory, agent, session); a later vote replaces the earlier
/// one. An upvote confirms the memory and resets its expiry timer.
pub fn cast_vote(
    conn: &Connection,
    memory_id: i64,
    agent: &str,
    session: &str,
    vote: i64,
    note: &str,
) -> Result<VoteCounts, String> {
    let vote = if vote >= 0 { 1 } else { -1 };
    let note = truncate_chars(note, 500);
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    let status: Option<String> = tx
        .query_row(
            "SELECT status FROM memories WHERE id = ?1",
            params![memory_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let Some(status) = status else {
        return Err(format!("memory {memory_id} not found"));
    };
    if status == "archived" {
        return Err(format!(
            "memory {memory_id} is archived; write a correction memory instead"
        ));
    }
    tx.execute(
        "INSERT INTO memory_votes (memory_id, agent, session, vote, note, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(memory_id, agent, session)
         DO UPDATE SET vote = excluded.vote, note = excluded.note, at = excluded.at",
        params![memory_id, agent, session, vote, note, now()],
    )
    .map_err(|e| e.to_string())?;
    let counts = vote_counts(&tx, memory_id).map_err(|e| e.to_string())?;
    if vote > 0 {
        // Procedures never expire; everything else gets a fresh TTL.
        tx.execute(
            "UPDATE memories SET upvotes = ?2, downvotes = ?3, last_confirmed_at = ?4,
               expires_at = CASE WHEN kind = 'procedure' THEN NULL ELSE ?5 END
             WHERE id = ?1",
            params![memory_id, counts.up, counts.down, now(), now() + ttl_secs()],
        )
        .map_err(|e| e.to_string())?;
    } else {
        // A downvote changes the score but deliberately does not extend life.
        tx.execute(
            "UPDATE memories SET upvotes = ?2, downvotes = ?3 WHERE id = ?1",
            params![memory_id, counts.up, counts.down],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(counts)
}

pub fn vote_counts(conn: &Connection, memory_id: i64) -> rusqlite::Result<VoteCounts> {
    conn.query_row(
        "SELECT COALESCE(SUM(vote = 1), 0), COALESCE(SUM(vote = -1), 0)
         FROM memory_votes WHERE memory_id = ?1",
        params![memory_id],
        |r| {
            Ok(VoteCounts {
                up: r.get(0)?,
                down: r.get(1)?,
            })
        },
    )
}

/// Transfer votes when one memory supersedes another. Colliding (agent,
/// session) rows are dropped in favour of the survivor's own vote; counters
/// on the survivor are then recomputed from the merged rows.
pub fn merge_votes(conn: &Connection, from_id: i64, to_id: i64) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO memory_votes (memory_id, agent, session, vote, note, at)
         SELECT ?2, agent, session, vote, note, at FROM memory_votes WHERE memory_id = ?1",
        params![from_id, to_id],
    )?;
    conn.execute(
        "DELETE FROM memory_votes WHERE memory_id = ?1",
        params![from_id],
    )?;
    let c = vote_counts(conn, to_id)?;
    conn.execute(
        "UPDATE memories SET upvotes = ?2, downvotes = ?3 WHERE id = ?1",
        params![to_id, c.up, c.down],
    )?;
    Ok(())
}

#[derive(Debug, Default)]
pub struct ExpiredSweep {
    pub archived: usize,
    /// (agent, project) pairs whose digest may now be stale.
    pub targets: Vec<(String, String)>,
}

/// Archive memories whose reinforcement timer ran out, reporting which
/// agent/project digests need a refresh afterwards.
pub fn archive_expired(conn: &Connection) -> rusqlite::Result<ExpiredSweep> {
    let t = now();
    let targets: Vec<(String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT agent, project FROM memories
             WHERE status != 'archived' AND expires_at IS NOT NULL AND expires_at <= ?1",
        )?;
        let rows = stmt.query_map(params![t], |r| Ok((r.get(0)?, r.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let archived = conn.execute(
        "UPDATE memories SET status = 'archived', updated_at = ?1
         WHERE status != 'archived' AND expires_at IS NOT NULL AND expires_at <= ?1",
        params![t],
    )?;
    Ok(ExpiredSweep { archived, targets })
}

pub fn get_memory(conn: &Connection, id: i64) -> rusqlite::Result<Option<Memory>> {
    conn.query_row(
        &format!("SELECT {MEMORY_COLS} FROM memories WHERE id = ?1"),
        params![id],
        memory_from_row,
    )
    .optional()
}

/// Non-archived memory of this agent whose content is byte-identical.
pub fn find_duplicate(
    conn: &Connection,
    agent: &str,
    hash: &str,
) -> rusqlite::Result<Option<i64>> {
    conn.query_row(
        "SELECT id FROM memories WHERE agent = ?1 AND content_hash = ?2 AND status != 'archived'
         ORDER BY id LIMIT 1",
        params![agent, hash],
        |r| r.get(0),
    )
    .optional()
}

// ---------------------------------------------------------------- vectors

pub fn set_vector(conn: &Connection, memory_id: i64, vector: &[f32]) -> rusqlite::Result<()> {
    debug_assert_eq!(vector.len(), DIM);
    conn.execute("DELETE FROM memories_vec WHERE memory_id = ?1", params![memory_id])?;
    conn.execute(
        "INSERT INTO memories_vec (memory_id, embedding) VALUES (?1, ?2)",
        params![memory_id, crate::embeddings::to_blob(vector)],
    )?;
    Ok(())
}

pub fn vector_for_memory(conn: &Connection, memory_id: i64) -> Option<Vec<f32>> {
    conn.query_row(
        "SELECT embedding FROM memories_vec WHERE memory_id = ?1",
        params![memory_id],
        |r| r.get::<_, Vec<u8>>(0),
    )
    .optional()
    .ok()
    .flatten()
    .map(|b| crate::embeddings::from_blob(&b))
}

/// Top-k nearest memories under the query vector, restricted to what `agent`
/// may see. Over-fetching then filtering happens in the caller for now.
pub fn knn(
    conn: &Connection,
    query: &[f32],
    agent: &str,
    project: Option<&str>,
    k: i64,
) -> rusqlite::Result<Vec<(i64, f32)>> {
    let mut stmt = conn.prepare(
        "SELECT v.memory_id, v.distance
         FROM (SELECT memory_id, distance FROM memories_vec WHERE embedding MATCH ?1 AND k = ?2) v
         JOIN memories m ON m.id = v.memory_id
         WHERE m.status != 'archived'
           AND (m.scope = 'shared' OR m.agent = ?3)
           AND (?4 IS NULL OR m.project = ?4)
           AND (m.expires_at IS NULL OR m.expires_at > ?5)
           AND NOT (m.downvotes >= 3 AND (m.upvotes - m.downvotes) < 0)
         ORDER BY v.distance",
    )?;
    let rows = stmt.query_map(
        params![crate::embeddings::to_blob(query), k, agent, project, now()],
        |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)? as f32)),
    )?;
    rows.collect()
}

// ---------------------------------------------------------------- search

/// FTS5 MATCH over memory content and tags. Returns (memory id, snippet).
pub fn fts_search(
    conn: &Connection,
    match_expr: &str,
    agent: &str,
    project: Option<&str>,
    limit: i64,
) -> rusqlite::Result<Vec<(i64, String)>> {
    let mut stmt = conn.prepare(
        "SELECT m.id, snippet(memories_fts, 0, '>>>', '<<<', ' … ', 32)
         FROM memories_fts
         JOIN memories m ON m.id = memories_fts.rowid
         WHERE memories_fts MATCH ?1
           AND m.status != 'archived'
           AND (m.scope = 'shared' OR m.agent = ?2)
           AND (?3 IS NULL OR m.project = ?3)
           AND (m.expires_at IS NULL OR m.expires_at > ?5)
           AND NOT (m.downvotes >= 3 AND (m.upvotes - m.downvotes) < 0)
         ORDER BY bm25(memories_fts, 1.0, 4.0)
         LIMIT ?4",
    )?;
    let rows = stmt.query_map(params![match_expr, agent, project, limit, now()], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
    })?;
    rows.collect()
}

/// Creation timestamps for a set of ids, used as the recency ranking signal.
pub fn created_at_map(
    conn: &Connection,
    ids: &[i64],
) -> rusqlite::Result<std::collections::HashMap<i64, i64>> {
    let mut out = std::collections::HashMap::new();
    if ids.is_empty() {
        return Ok(out);
    }
    let placeholders = std::iter::repeat_n("?", ids.len()).collect::<Vec<_>>().join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT id, created_at FROM memories WHERE id IN ({placeholders})"
    ))?;
    let mut rows = stmt.query(rusqlite::params_from_iter(ids.iter()))?;
    while let Some(r) = rows.next()? {
        out.insert(r.get(0)?, r.get(1)?);
    }
    Ok(out)
}

pub fn bump_access(conn: &Connection, ids: &[i64]) -> rusqlite::Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let placeholders = std::iter::repeat_n("?", ids.len()).collect::<Vec<_>>().join(",");
    conn.execute(
        &format!(
            "UPDATE memories SET access_count = access_count + 1, last_access_at = ?1
             WHERE id IN ({placeholders})"
        ),
        rusqlite::params_from_iter(std::iter::once(&now()).chain(ids.iter())),
    )?;
    Ok(())
}

// ---------------------------------------------------------------- curation

pub fn pending_raw(
    conn: &Connection,
    limit: i64,
) -> rusqlite::Result<Vec<RawRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, source_session
         FROM raw_memories WHERE consumed_at IS NULL ORDER BY id LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit], |r| {
        Ok(RawRecord {
            id: r.get(0)?,
            source_session: r.get(1)?,
        })
    })?;
    rows.collect()
}

pub fn memories_for_raw(conn: &Connection, raw_id: i64) -> rusqlite::Result<Vec<Memory>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {MEMORY_COLS} FROM memories WHERE raw_id = ?1"
    ))?;
    let rows = stmt.query_map(params![raw_id], memory_from_row)?;
    rows.collect()
}

pub fn mark_raw_consumed(conn: &Connection, ids: &[i64]) -> rusqlite::Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let placeholders = std::iter::repeat_n("?", ids.len()).collect::<Vec<_>>().join(",");
    conn.execute(
        &format!("UPDATE raw_memories SET consumed_at = ?1 WHERE id IN ({placeholders})"),
        rusqlite::params_from_iter(std::iter::once(&now()).chain(ids.iter())),
    )?;
    Ok(())
}

pub fn get_consumer(conn: &Connection, name: &str) -> rusqlite::Result<(i64, i64)> {
    conn.query_row(
        "SELECT last_raw_id, last_run_at FROM consumers WHERE name = ?1",
        params![name],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
    .optional()
    .map(|o| o.unwrap_or((0, 0)))
}

pub fn set_consumer(
    conn: &Connection,
    name: &str,
    last_raw_id: i64,
    last_run_at: i64,
) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO consumers (name, last_raw_id, last_run_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(name) DO UPDATE SET last_raw_id = excluded.last_raw_id, last_run_at = excluded.last_run_at",
        params![name, last_raw_id, last_run_at],
    )?;
    Ok(())
}

pub fn get_digest(conn: &Connection, agent: &str, project: &str) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT content FROM digest WHERE agent = ?1 AND project = ?2",
        params![agent, project],
        |r| r.get(0),
    )
    .optional()
}

/// Every project digest for an agent, ordered by project.
pub fn digests_for_agent(conn: &Connection, agent: &str) -> rusqlite::Result<Vec<(String, String)>> {
    let mut stmt =
        conn.prepare("SELECT project, content FROM digest WHERE agent = ?1 ORDER BY project")?;
    let rows = stmt.query_map(params![agent], |r| Ok((r.get(0)?, r.get(1)?)))?;
    rows.collect()
}

pub fn set_digest(
    conn: &Connection,
    agent: &str,
    project: &str,
    content: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO digest (agent, project, content, updated_at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(agent, project) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
        params![agent, project, content, now()],
    )?;
    Ok(())
}

// ---------------------------------------------------------------- locks / audit

pub fn acquire_lock(
    conn: &Connection,
    name: &str,
    holder: &str,
    lease_secs: i64,
) -> rusqlite::Result<bool> {
    let t = now();
    conn.execute(
        "INSERT OR IGNORE INTO locks (name, holder, acquired_at) VALUES (?1, '', 0)",
        params![name],
    )?;
    let changed = conn.execute(
        "UPDATE locks SET holder = ?2, acquired_at = ?3
         WHERE name = ?1 AND (holder = '' OR acquired_at < ?4)",
        params![name, holder, t, t - lease_secs],
    )?;
    Ok(changed == 1)
}

pub fn release_lock(conn: &Connection, name: &str, holder: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE locks SET holder = '', acquired_at = 0 WHERE name = ?1 AND holder = ?2",
        params![name, holder],
    )?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct VersionRow {
    pub id: i64,
    pub memory_id: i64,
    pub version: i64,
    pub content: String,
    pub op: String,
    pub at: i64,
}

pub fn versions(conn: &Connection, memory_id: i64) -> rusqlite::Result<Vec<VersionRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, memory_id, version, content, op, at FROM memory_versions
         WHERE memory_id = ?1 ORDER BY version",
    )?;
    let rows = stmt.query_map(params![memory_id], |r| {
        Ok(VersionRow {
            id: r.get(0)?,
            memory_id: r.get(1)?,
            version: r.get(2)?,
            content: r.get(3)?,
            op: r.get(4)?,
            at: r.get(5)?,
        })
    })?;
    rows.collect()
}

/// Restore a memory's content to a stored version. The restore is itself
/// versioned, so nothing is lost.
pub fn restore_version(conn: &Connection, version_id: i64) -> rusqlite::Result<i64> {
    let (memory_id, content): (i64, String) = conn.query_row(
        "SELECT memory_id, content FROM memory_versions WHERE id = ?1",
        params![version_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let tx = conn.unchecked_transaction()?;
    let version: i64 = tx.query_row(
        "SELECT version FROM memories WHERE id = ?1",
        params![memory_id],
        |r| r.get(0),
    )?;
    let v = version + 1;
    tx.execute(
        "UPDATE memories SET content = ?2, updated_at = ?3, version = ?4, status = 'canonical',
           content_hash = ?5, superseded_by = NULL,
           expires_at = CASE WHEN kind = 'procedure' THEN NULL ELSE ?6 END
         WHERE id = ?1",
        params![memory_id, content, now(), v, content_hash(&content), now() + ttl_secs()],
    )?;
    insert_version(&tx, memory_id, v, &content, "restore")?;
    tx.commit()?;
    Ok(memory_id)
}

#[derive(Debug, Default)]
pub struct Stats {
    pub raw_total: i64,
    pub raw_pending: i64,
    pub memories: i64,
    pub provisional: i64,
    pub canonical: i64,
    pub archived: i64,
    pub agents: i64,
    pub votes: i64,
}

pub fn stats(conn: &Connection) -> rusqlite::Result<Stats> {
    let q = |sql: &str| -> rusqlite::Result<i64> { conn.query_row(sql, [], |r| r.get(0)) };
    Ok(Stats {
        raw_total: q("SELECT COUNT(*) FROM raw_memories")?,
        raw_pending: q("SELECT COUNT(*) FROM raw_memories WHERE consumed_at IS NULL")?,
        memories: q("SELECT COUNT(*) FROM memories")?,
        provisional: q("SELECT COUNT(*) FROM memories WHERE status = 'provisional'")?,
        canonical: q("SELECT COUNT(*) FROM memories WHERE status = 'canonical'")?,
        archived: q("SELECT COUNT(*) FROM memories WHERE status = 'archived'")?,
        agents: q("SELECT COUNT(DISTINCT agent) FROM memories")?,
        votes: q("SELECT COUNT(*) FROM memory_votes")?,
    })
}

/// All non-archived memories with their content, for a full re-embed.
pub fn active_memories(conn: &Connection) -> rusqlite::Result<Vec<(i64, String)>> {
    let mut stmt =
        conn.prepare("SELECT id, content FROM memories WHERE status != 'archived' ORDER BY id")?;
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    rows.collect()
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db(name: &str) -> Connection {
        let dir = std::env::temp_dir().join(format!(
            "ams_test_{}_{}",
            name,
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join("memory.db"));
        open_db_at(&dir.join("memory.db")).unwrap()
    }

    fn unit_vec(i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        v[i] = 1.0;
        v
    }

    #[test]
    fn schema_initializes() {
        let conn = temp_db("schema");
        let v: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(v, 3);
        // sqlite-vec is loaded and the vec0 table exists
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories_vec", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn insert_and_read_memory_with_vector() {
        let conn = temp_db("roundtrip");
        let raw = insert_raw(&conn, "a", "proj", "fact", "own", "hello", "").unwrap();
        let id = insert_memory(&conn, raw, "a", "proj", "fact", "own", "hello", "tag", "provisional", 0.6, None).unwrap();
        set_vector(&conn, id, &unit_vec(1)).unwrap();
        let m = get_memory(&conn, id).unwrap().unwrap();
        assert_eq!(m.content, "hello");
        assert_eq!(m.version, 1);
        assert_eq!(vector_for_memory(&conn, id).unwrap()[1], 1.0);
        assert_eq!(knn(&conn, &unit_vec(1), "a", None, 5).unwrap().len(), 1);
    }

    #[test]
    fn knn_respects_agent_scope() {
        let conn = temp_db("knn_scope");
        let r1 = insert_raw(&conn, "a", "", "fact", "own", "one", "").unwrap();
        let id1 = insert_memory(&conn, r1, "a", "", "fact", "own", "one", "", "canonical", 0.6, None).unwrap();
        set_vector(&conn, id1, &unit_vec(1)).unwrap();
        let r2 = insert_raw(&conn, "b", "", "fact", "own", "two", "").unwrap();
        let id2 = insert_memory(&conn, r2, "b", "", "fact", "own", "two", "", "canonical", 0.6, None).unwrap();
        set_vector(&conn, id2, &unit_vec(1)).unwrap();

        let hits = knn(&conn, &unit_vec(1), "a", None, 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id1);
    }

    #[test]
    fn fts_finds_and_updates() {
        let conn = temp_db("fts");
        let raw = insert_raw(&conn, "a", "", "fact", "own", "kubernetes rollout", "").unwrap();
        let id = insert_memory(&conn, raw, "a", "", "fact", "own", "kubernetes rollout", "", "canonical", 0.6, None).unwrap();
        let hits = fts_search(&conn, "\"kubernetes\"", "a", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);

        update_memory(&conn, id, "terraform apply", "", "fact", "own", 0.6, None, "curator").unwrap();
        assert!(fts_search(&conn, "\"kubernetes\"", "a", None, 10).unwrap().is_empty());
        assert_eq!(fts_search(&conn, "\"terraform\"", "a", None, 10).unwrap().len(), 1);
    }

    #[test]
    fn archiving_hides_from_search() {
        let conn = temp_db("archive");
        let raw = insert_raw(&conn, "a", "", "fact", "own", "temporary note", "").unwrap();
        let id = insert_memory(&conn, raw, "a", "", "fact", "own", "temporary note", "", "canonical", 0.6, None).unwrap();
        archive_memory(&conn, id, None).unwrap();
        assert!(fts_search(&conn, "\"temporary\"", "a", None, 10).unwrap().is_empty());
    }

    #[test]
    fn shared_memories_visible_to_everyone() {
        let conn = temp_db("shared");
        let raw = insert_raw(&conn, "a", "", "fact", "shared", "shared fact", "").unwrap();
        insert_memory(&conn, raw, "a", "", "fact", "shared", "shared fact", "", "canonical", 0.6, None).unwrap();
        assert_eq!(fts_search(&conn, "\"shared\"", "b", None, 10).unwrap().len(), 1);
    }

    #[test]
    fn version_restore_roundtrip() {
        let conn = temp_db("restore");
        let raw = insert_raw(&conn, "a", "", "fact", "own", "v1", "").unwrap();
        let id = insert_memory(&conn, raw, "a", "", "fact", "own", "v1", "", "canonical", 0.6, None).unwrap();
        update_memory(&conn, id, "v2", "", "fact", "own", 0.6, None, "curator").unwrap();
        let vers = versions(&conn, id).unwrap();
        assert_eq!(vers.len(), 2);
        restore_version(&conn, vers[0].id).unwrap();
        assert_eq!(get_memory(&conn, id).unwrap().unwrap().content, "v1");
        assert_eq!(versions(&conn, id).unwrap().len(), 3);
    }

    #[test]
    fn lock_is_exclusive_until_released() {
        let conn = temp_db("lock");
        assert!(acquire_lock(&conn, "curator", "p1", 100).unwrap());
        assert!(!acquire_lock(&conn, "curator", "p2", 100).unwrap());
        release_lock(&conn, "curator", "p1").unwrap();
        assert!(acquire_lock(&conn, "curator", "p2", 100).unwrap());
    }

    #[test]
    fn pending_raw_orders_and_marks() {
        let conn = temp_db("pending");
        let a = insert_raw(&conn, "a", "", "fact", "own", "one", "").unwrap();
        let b = insert_raw(&conn, "a", "", "fact", "own", "two", "").unwrap();
        let p = pending_raw(&conn, 10).unwrap();
        assert_eq!(p.iter().map(|r| r.id).collect::<Vec<_>>(), vec![a, b]);
        mark_raw_consumed(&conn, &[a]).unwrap();
        assert_eq!(pending_raw(&conn, 10).unwrap().len(), 1);
    }

    fn mem(conn: &Connection, agent: &str, content: &str, expires_at: Option<i64>) -> i64 {
        let raw = insert_raw(conn, agent, "", "fact", "own", content, "").unwrap();
        insert_memory(conn, raw, agent, "", "fact", "own", content, "", "canonical", 0.6, expires_at)
            .unwrap()
    }

    #[test]
    fn vote_is_one_per_agent_session_and_refreshes_expiry() {
        let conn = temp_db("votes");
        let id = mem(&conn, "a", "kubernetes rollback", Some(now() - 10));

        // Expired memory is invisible until an upvote revives it.
        assert!(fts_search(&conn, "\"kubernetes\"", "a", None, 10).unwrap().is_empty());

        let c = cast_vote(&conn, id, "a", "s1", 1, "").unwrap();
        assert_eq!((c.up, c.down), (1, 0));
        let m = get_memory(&conn, id).unwrap().unwrap();
        assert!(m.expires_at.unwrap() > now());
        assert!(m.last_confirmed_at.is_some());
        assert_eq!(fts_search(&conn, "\"kubernetes\"", "a", None, 10).unwrap().len(), 1);

        // Re-voting in the same session replaces the vote.
        let c = cast_vote(&conn, id, "a", "s1", -1, "").unwrap();
        assert_eq!((c.up, c.down), (0, 1));

        // A different session or agent adds a vote.
        let c = cast_vote(&conn, id, "a", "s2", 1, "").unwrap();
        assert_eq!((c.up, c.down), (1, 1));
        let c = cast_vote(&conn, id, "b", "s9", 1, "").unwrap();
        assert_eq!((c.up, c.down), (2, 1));

        // Downvotes never extend the timer.
        let before = get_memory(&conn, id).unwrap().unwrap().expires_at;
        cast_vote(&conn, id, "c", "s3", -1, "").unwrap();
        assert_eq!(get_memory(&conn, id).unwrap().unwrap().expires_at, before);
    }

    #[test]
    fn archived_memory_rejects_votes() {
        let conn = temp_db("vote_archived");
        let id = mem(&conn, "a", "old fact", None);
        archive_memory(&conn, id, None).unwrap();
        assert!(cast_vote(&conn, id, "a", "s1", 1, "").is_err());
    }

    #[test]
    fn three_downvotes_hide_a_memory() {
        let conn = temp_db("vote_hide");
        let id = mem(&conn, "a", "wrong rollback advice", None);
        for (i, agent) in ["a", "b", "c"].iter().enumerate() {
            cast_vote(&conn, id, agent, &format!("s{i}"), -1, "").unwrap();
        }
        assert!(fts_search(&conn, "\"rollback\"", "a", None, 10).unwrap().is_empty());
        assert!(get_memory(&conn, id).unwrap().is_some());
    }

    #[test]
    fn merge_votes_moves_counts_to_survivor() {
        let conn = temp_db("vote_merge");
        let old = mem(&conn, "a", "old duplicate", None);
        let new = mem(&conn, "a", "merged fact", None);
        cast_vote(&conn, old, "a", "s1", 1, "").unwrap();
        cast_vote(&conn, old, "b", "s2", 1, "").unwrap();
        cast_vote(&conn, new, "c", "s3", 1, "").unwrap();
        merge_votes(&conn, old, new).unwrap();
        let m = get_memory(&conn, new).unwrap().unwrap();
        assert_eq!((m.upvotes, m.downvotes), (3, 0));
        assert_eq!(vote_counts(&conn, old).unwrap().up, 0);
    }

    #[test]
    fn expired_sweep_archives_only_expired() {
        let conn = temp_db("expiry");
        let old = mem(&conn, "a", "stale", Some(now() - 1));
        let fresh = mem(&conn, "a", "fresh", Some(now() + 1000));
        let eternal = mem(&conn, "a", "eternal", None);
        let sweep = archive_expired(&conn).unwrap();
        assert_eq!(sweep.archived, 1);
        assert_eq!(sweep.targets, vec![("a".to_string(), String::new())]);
        assert_eq!(get_memory(&conn, old).unwrap().unwrap().status, "archived");
        assert_eq!(get_memory(&conn, fresh).unwrap().unwrap().status, "canonical");
        assert_eq!(get_memory(&conn, eternal).unwrap().unwrap().status, "canonical");
    }

    #[test]
    fn usefulness_is_neutral_at_zero_votes() {
        assert_eq!(usefulness(0, 0), 0.0);
        assert!(usefulness(3, 0) > 0.0);
        assert!(usefulness(0, 3) < 0.0);
    }

    #[test]
    fn v1_database_upgrades_to_latest() {
        let dir = std::env::temp_dir().join(format!("ams_test_upgrade_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.db");
        let _ = std::fs::remove_file(&path);
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(SCHEMA_V1).unwrap();
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('vector_model', ?1)",
                params![crate::embeddings::model_id()],
            )
            .unwrap();
            conn.pragma_update(None, "user_version", 1).unwrap();
        }
        let conn = open_db_at(&path).unwrap();
        let v: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(v, 3);
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory_votes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn duplicate_content_rejected_until_archived() {
        let conn = temp_db("unique_hash");
        let first = mem(&conn, "a", "same text", None);
        let raw = insert_raw(&conn, "a", "", "fact", "own", "same text", "").unwrap();
        let err = insert_memory(
            &conn, raw, "a", "", "fact", "own", "same text", "", "provisional", 0.6, None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            rusqlite::Error::SqliteFailure(e, _) if e.code == rusqlite::ErrorCode::ConstraintViolation
        ));
        // A different agent may hold the same text.
        mem(&conn, "b", "same text", None);
        // Archiving the original frees the hash for this agent.
        archive_memory(&conn, first, None).unwrap();
        let raw2 = insert_raw(&conn, "a", "", "fact", "own", "same text", "").unwrap();
        insert_memory(
            &conn, raw2, "a", "", "fact", "own", "same text", "", "provisional", 0.6, None,
        )
        .unwrap();
    }

    #[test]
    fn restore_clears_expiry_and_supersession() {
        let conn = temp_db("restore_expiry");
        let id = mem(&conn, "a", "restorable content", Some(now() - 100));
        archive_memory(&conn, id, Some(id)).unwrap();
        let vid = versions(&conn, id).unwrap()[0].id;
        restore_version(&conn, vid).unwrap();
        let m = get_memory(&conn, id).unwrap().unwrap();
        assert_eq!(m.status, "canonical");
        assert!(m.superseded_by.is_none());
        assert!(m.expires_at.unwrap() > now());
        assert_eq!(
            fts_search(&conn, "\"restorable\"", "a", None, 10).unwrap().len(),
            1
        );
    }

    #[test]
    fn digests_are_per_project() {
        let conn = temp_db("digest");
        set_digest(&conn, "a", "alpha", "alpha facts").unwrap();
        set_digest(&conn, "a", "beta", "beta facts").unwrap();
        assert_eq!(
            get_digest(&conn, "a", "alpha").unwrap().as_deref(),
            Some("alpha facts")
        );
        let all = digests_for_agent(&conn, "a").unwrap();
        assert_eq!(all.len(), 2);
    }
}
